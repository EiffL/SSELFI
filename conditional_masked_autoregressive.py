from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import affine_scalar
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.numeric import clip_by_value_preserve_gradient
from tensorflow_probability.python.bijectors import masked_dense

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = [
    'ConditionalMaskedAutoregressiveFlow',
    'masked_autoregressive_conditional_template'
]

class ConditionalMaskedAutoregressiveFlow(bijector_lib.Bijector):
  """ Conditional Affine MaskedAutoregressiveFlow bijector.
  """
  def __init__(self,
               shift_and_log_scale_fn=None,
               conditioning=None,
               bijector_fn=None,
               is_constant_jacobian=False,
               validate_args=False,
               unroll_loop=False,
               event_ndims=1,
               name=None):
    """Creates the MaskedAutoregressiveFlow bijector.
    Args:
      shift_and_log_scale_fn: Python `callable` which computes `shift` and
        `log_scale` from the inverse domain (`y`). Calculation must respect the
        'autoregressive property' (see class docstring). Suggested default
        `tfb.AutoregressiveNetwork(params=2, hidden_layers=...)`.
        Typically the function contains `tf.Variables`. Returning `None` for
        either (both) `shift`, `log_scale` is equivalent to (but more efficient
        than) returning zero. If `shift_and_log_scale_fn` returns a single
        `Tensor`, the returned value will be unstacked to get the `shift` and
        `log_scale`: `tf.unstack(shift_and_log_scale_fn(y), num=2, axis=-1)`.
      bijector_fn: Python `callable` which returns a `tfb.Bijector` which
        transforms event tensor with the signature
        `(input, **condition_kwargs) -> bijector`. The bijector must operate on
        scalar events and must not alter the rank of its input. The
        `bijector_fn` will be called with `Tensors` from the inverse domain
        (`y`). Calculation must respect the 'autoregressive property' (see
        class docstring).
      is_constant_jacobian: Python `bool`. Default: `False`. When `True` the
        implementation assumes `log_scale` does not depend on the forward domain
        (`x`) or inverse domain (`y`) values. (No validation is made;
        `is_constant_jacobian=False` is always safe but possibly computationally
        inefficient.)
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      unroll_loop: Python `bool` indicating whether the `tf.while_loop` in
        `_forward` should be replaced with a static for loop. Requires that
        the final dimension of `x` be known at graph construction time. Defaults
        to `False`.
      event_ndims: Python `integer`, the intrinsic dimensionality of this
        bijector. 1 corresponds to a simple vector autoregressive bijector as
        implemented by the `tfp.bijectors.AutoregressiveNetwork`, 2 might be
        useful for a 2D convolutional `shift_and_log_scale_fn` and so on.
      name: Python `str`, name given to ops managed by this object.
    Raises:
      ValueError: If both or none of `shift_and_log_scale_fn` and `bijector_fn`
          are specified.
    """
    name = name or 'conditional_masked_autoregressive_flow'
    self._unroll_loop = unroll_loop
    self._event_ndims = event_ndims
    if bool(shift_and_log_scale_fn) == bool(bijector_fn):
      raise ValueError('Exactly one of `shift_and_log_scale_fn` and '
                       '`bijector_fn` should be specified.')
    if shift_and_log_scale_fn:
      def _bijector_fn(x, **condition_kwargs):
        if conditioning is not None:
          print(x, conditioning)
          x = tf.concat([conditioning, x], axis=-1)
          cond_depth = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(conditioning.shape, 1)[-1])
        else:
          cond_depth = 0
        params = shift_and_log_scale_fn(x, **condition_kwargs)
        if tf.is_tensor(params):
          shift, log_scale = tf.unstack(params, num=2, axis=-1)
        else:
          shift, log_scale = params
        shift = shift[..., cond_depth:]
        log_scale = log_scale[..., cond_depth:]
        return affine_scalar.AffineScalar(shift=shift, log_scale=log_scale)

      bijector_fn = _bijector_fn

    if validate_args:
      bijector_fn = _validate_bijector_fn(bijector_fn)
    # Still do this assignment for variable tracking.
    self._shift_and_log_scale_fn = shift_and_log_scale_fn
    self._bijector_fn = bijector_fn
    super(ConditionalMaskedAutoregressiveFlow, self).__init__(
        forward_min_event_ndims=self._event_ndims,
        is_constant_jacobian=is_constant_jacobian,
        validate_args=validate_args,
        name=name)

  def _forward(self, x, **kwargs):
    static_event_size = tensorshape_util.num_elements(
        tensorshape_util.with_rank_at_least(
            x.shape, self._event_ndims)[-self._event_ndims:])

    if self._unroll_loop:
      if not static_event_size:
        raise ValueError(
            'The final {} dimensions of `x` must be known at graph '
            'construction time if `unroll_loop=True`. `x.shape: {!r}`'.format(
                self._event_ndims, x.shape))
      y = tf.zeros_like(x, name='y0')

      for _ in range(static_event_size):
        y = self._bijector_fn(y, **kwargs).forward(x)
      return y

    event_size = tf.reduce_prod(tf.shape(x)[-self._event_ndims:])
    y0 = tf.zeros_like(x, name='y0')
    # call the template once to ensure creation
    if not tf.executing_eagerly():
      _ = self._bijector_fn(y0, **kwargs).forward(y0)
    def _loop_body(index, y0):
      """While-loop body for autoregression calculation."""
      # Set caching device to avoid re-getting the tf.Variable for every while
      # loop iteration.
      with tf1.variable_scope(tf1.get_variable_scope()) as vs:
        if vs.caching_device is None and not tf.executing_eagerly():
          vs.set_caching_device(lambda op: op.device)
        bijector = self._bijector_fn(y0, **kwargs)
      y = bijector.forward(x)
      return index + 1, y
    # If the event size is available at graph construction time, we can inform
    # the graph compiler of the maximum number of steps. If not,
    # static_event_size will be None, and the maximum_iterations argument will
    # have no effect.
    _, y = tf.while_loop(
        cond=lambda index, _: index < event_size,
        body=_loop_body,
        loop_vars=(0, y0),
        maximum_iterations=static_event_size)
    return y

  def _inverse(self, y, **kwargs):
    bijector = self._bijector_fn(y, **kwargs)
    return bijector.inverse(y)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    return self._bijector_fn(y, **kwargs).inverse_log_det_jacobian(
        y, event_ndims=self._event_ndims)

def masked_autoregressive_conditional_template(hidden_layers,
                                            conditional_tensor,
                                           shift_only=False,
                                           activation=tf.nn.relu,
                                           log_scale_min_clip=-5.,
                                           log_scale_max_clip=3.,
                                           log_scale_clip_gradient=False,
                                           name=None,
                                           *args,  # pylint: disable=keyword-arg-before-vararg
                                           **kwargs):
  """Build the Masked Autoregressive Density Estimator (Germain et al., 2015).
  This will be wrapped in a make_template to ensure the variables are only
  created once. It takes the input and returns the `loc` ('mu' in [Germain et
  al. (2015)][1]) and `log_scale` ('alpha' in [Germain et al. (2015)][1]) from
  the MADE network.
  Warning: This function uses `masked_dense` to create randomly initialized
  `tf.Variables`. It is presumed that these will be fit, just as you would any
  other neural architecture which uses `tf.layers.dense`.
  #### About Hidden Layers
  Each element of `hidden_layers` should be greater than the `input_depth`
  (i.e., `input_depth = tf.shape(input)[-1]` where `input` is the input to the
  neural network). This is necessary to ensure the autoregressivity property.
  #### About Clipping
  This function also optionally clips the `log_scale` (but possibly not its
  gradient). This is useful because if `log_scale` is too small/large it might
  underflow/overflow making it impossible for the `MaskedAutoregressiveFlow`
  bijector to implement a bijection. Additionally, the `log_scale_clip_gradient`
  `bool` indicates whether the gradient should also be clipped. The default does
  not clip the gradient; this is useful because it still provides gradient
  information (for fitting) yet solves the numerical stability problem. I.e.,
  `log_scale_clip_gradient = False` means
  `grad[exp(clip(x))] = grad[x] exp(clip(x))` rather than the usual
  `grad[clip(x)] exp(clip(x))`.
  Args:
    hidden_layers: Python `list`-like of non-negative integer, scalars
      indicating the number of units in each hidden layer. Default: `[512, 512].
    shift_only: Python `bool` indicating if only the `shift` term shall be
      computed. Default: `False`.
    activation: Activation function (callable). Explicitly setting to `None`
      implies a linear activation.
    log_scale_min_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The minimum value to clip by. Default: -5.
    log_scale_max_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The maximum value to clip by. Default: 3.
    log_scale_clip_gradient: Python `bool` indicating that the gradient of
      `tf.clip_by_value` should be preserved. Default: `False`.
    name: A name for ops managed by this function. Default:
      'masked_autoregressive_default_template'.
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.
  Returns:
    shift: `Float`-like `Tensor` of shift terms (the 'mu' in
      [Germain et al.  (2015)][1]).
    log_scale: `Float`-like `Tensor` of log(scale) terms (the 'alpha' in
      [Germain et al. (2015)][1]).
  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.
  #### References
  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
  """
  name = name or 'masked_autoregressive_conditional_template'
  with tf.name_scope(name):

    def _fn(x):
      """MADE parameterized via `masked_autoregressive_default_template`."""
      # TODO(b/67594795): Better support of dynamic shape.
      input_depth = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(x.shape, 1)[-1])
      cond_depth = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(conditional_tensor.shape, 1)[-1])
      if input_depth is None:
        raise NotImplementedError(
            'Rightmost dimension must be known prior to graph execution.')
      input_shape = (
          np.int32(tensorshape_util.as_list(x.shape))
          if tensorshape_util.is_fully_defined(x.shape) else tf.shape(x))
      if tensorshape_util.rank(x.shape) == 1:
        x = x[tf.newaxis, ...]
      x = tf.concat([conditional_tensor, x],  axis=1)
      for i, units in enumerate(hidden_layers):
        x = masked_dense(
            inputs=x,
            units=units,
            num_blocks=input_depth,
            exclusive=True if i == 0 else False,
            activation=activation,
            *args,  # pylint: disable=keyword-arg-before-vararg
            **kwargs)
      x = masked_dense(
          inputs=x,
          units=(1 if shift_only else 2) * input_depth,
          num_blocks=input_depth,
          activation=None,
          *args,  # pylint: disable=keyword-arg-before-vararg
          **kwargs)
      if shift_only:
        x = x[:, cond_depth:]
        x = tf.reshape(x, shape=input_shape)
        return x, None
      else:
        x = x[:, 2*cond_depth:]
      x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
      shift, log_scale = tf.unstack(x, num=2, axis=-1)
      which_clip = (
          tf.clip_by_value
          if log_scale_clip_gradient else clip_by_value_preserve_gradient)
      log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
      return shift, log_scale

    return tf1.make_template(name, _fn)
