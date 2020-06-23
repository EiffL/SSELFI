import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from tfp_utils import RationalQuadraticSpline

__all__ = [
    'ConditionalNeuralSpline',
]

class ConditionalNeuralSpline(tf.Module):
  def __init__(self, conditional_tensor=None, nbins=32, hidden_layers=[256],
               activation='relu',name=None):
    self._nbins = nbins
    self._built = False
    self._bin_widths = None
    self._bin_heights = None
    self._knot_slopes = None
    self._layers= []
    self._activation = activation
    self._hidden_layers = hidden_layers
    self._conditional_tensor = conditional_tensor
    super(ConditionalNeuralSpline, self).__init__(name)

  def __call__(self, x, nunits):
    if not self._built:
      def _bin_positions(x):
        x = tf.reshape(x, [-1, nunits, self._nbins])
        return tf.math.softmax(x, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

      def _slopes(x):
        x = tf.reshape(x, [-1, nunits, self._nbins - 1])
        return tf.math.softplus(x) + 1e-2

      for i, units in enumerate(self._hidden_layers):
        self._layers.append(tf.keras.layers.Dense(units, activation=self._activation,
                                                  name='layer_%d'%i))
      self._bin_widths = tf.keras.layers.Dense(
          nunits * self._nbins, activation=_bin_positions, name='w')

      self._bin_heights = tf.keras.layers.Dense(
          nunits * self._nbins, activation=_bin_positions, name='h')

      self._knot_slopes = tf.keras.layers.Dense(
          nunits * (self._nbins - 1), activation=_slopes, name='s')
      self._built = True

    # If provided, we append the condition as an input to the network
    if self._conditional_tensor is not None:
      net = tf.concat([x, self._conditional_tensor], axis=-1)
    else:
      net = x

    # Apply hidden layers
    for layer in self._layers:
      net = layer(net)

    return RationalQuadraticSpline(
        bin_widths=self._bin_widths(net),
        bin_heights=self._bin_heights(net),
        knot_slopes=self._knot_slopes(net))
