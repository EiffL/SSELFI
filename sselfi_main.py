# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a ResNet-50 inference model on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Adds path to the TPU code repo
import sys
sys.path.append('tpu/models')

import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from conditional_masked_autoregressive import ConditionalMaskedAutoregressiveFlow, masked_autoregressive_conditional_template
from conditional_neural_spline_flow import ConditionalNeuralSpline
from tfp_utils import RealNVP
from des_sv_input import DESSVInput, image_serving_input_fn
# import tensorflow_hub as hub

from common import inference_warmup
from common import tpu_profiler_hook
from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import flags_to_params
from hyperparameters import params_dict
from official.resnet import lars_util
from official.resnet import resnet_model
from official.resnet.configs import resnet_config
from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'resnet_depth', default=34,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))

flags.DEFINE_integer(
    'num_train_images', default=166500, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=18432, help='Size of testing data set.')

flags.DEFINE_integer(
    'num_label_classes', default=2, help='Number of classes, at least 2')

flags.DEFINE_string(
    'data_format', default=None,
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))

flags.DEFINE_bool(
    'transpose_input', default=None,
    help='Use TPU double transpose optimization')

flags.DEFINE_bool(
    'use_cache', default=None, help=('Enable cache for training input.'))

flags.DEFINE_integer('image_size', 224, 'The input image size.')

flags.DEFINE_string(
    'dropblock_groups', None,
    help=('A string containing comma separated integers indicating ResNet '
          'block groups to apply DropBlock. `3,4` means to apply DropBlock to '
          'block groups 3 and 4. Use an empty string to not apply DropBlock to '
          'any block group.'))
flags.DEFINE_float(
    'dropblock_keep_prob', default=None,
    help=('keep_prob parameter of DropBlock. Will not be used if '
          'dropblock_groups is empty.'))
flags.DEFINE_integer(
    'dropblock_size', default=None,
    help=('size parameter of DropBlock. Will not be used if dropblock_groups '
          'is empty.'))

flags.DEFINE_integer(
    'profile_every_n_steps', default=0,
    help=('Number of steps between collecting profiles if larger than 0'))

flags.DEFINE_string(
    'mode', default='train',
    help='One of {"train_and_eval", "train", "eval"}.')

flags.DEFINE_integer(
    'steps_per_eval', default=1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_integer(
    'num_parallel_calls', default=None,
    help=('Number of parallel threads in CPU for the input pipeline.'
          ' Recommended value is the number of cores per CPU host.'))

flags.DEFINE_integer(
    'num_cores', default=None,
    help=('Number of TPU cores in total. For a single TPU device, this is 8'
          ' because each TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_bool(
    'export_to_tpu', default=False,
    help=('Whether to export additional metagraph with "serve, tpu" tags'
          ' in addition to "serve" only metagraph.'))

flags.DEFINE_float(
    'base_learning_rate', default=0.01,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=None,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'weight_decay', default=None,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.,
    help=('Label smoothing parameter used in the loss function'))

flags.DEFINE_bool('enable_lars',
                  default=None,
                  help=('Enable LARS optimizer for large batch training.'))

flags.DEFINE_float('poly_rate', default=None,
                   help=('Set LARS/Poly learning rate.'))

flags.DEFINE_bool(
    'use_async_checkpointing', default=None, help=('Enable async checkpoint'))

flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                     'which the global step information is logged.')

# Inference configuration.
flags.DEFINE_bool(
    'add_warmup_requests', False,
    'Whether to add warmup requests into the export saved model dir,'
    'especially for TPU inference.')
flags.DEFINE_string('model_name', 'resnet',
                    'Serving model name used for the model server.')
flags.DEFINE_multi_integer(
    'inference_batch_sizes', [8],
    'Known inference batch sizes used to warm up for each core.')


def get_lr_schedule(train_steps, num_train_images, train_batch_size):
  """learning rate schedule."""
  steps_per_epoch = np.floor(num_train_images / train_batch_size)
  train_epochs = train_steps / steps_per_epoch
  return [  # (multiplier, epoch to start) tuples
      (1.0, np.floor(5 / 90 * train_epochs)),
      (0.1, np.floor(30 / 90 * train_epochs)),
      (0.01, np.floor(60 / 90 * train_epochs)),
      (0.001, np.floor(80 / 90 * train_epochs))
  ]


def learning_rate_schedule(params, current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.

  Args:
    params: Python dict containing parameters for this run.
    current_epoch: `Tensor` for current epoch.

  Returns:
    A scaled `Tensor` for current learning rate.
  """
  scaled_lr = params['base_learning_rate'] * (
      params['train_batch_size'] / 256.0)

  lr_schedule = get_lr_schedule(
      train_steps=params['train_steps'],
      num_train_images=params['num_train_images'],
      train_batch_size=params['train_batch_size'])
  decay_rate = (scaled_lr * lr_schedule[0][0] *
                current_epoch / lr_schedule[0][1])
  for mult, start_epoch in lr_schedule:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate


def resnet_model_fn(features, labels, mode, params):
  """The model_fn for ResNet to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images. If transpose_input is enabled, it
        is transposed to device layout and reshaped to 1D tensor.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    features = features['feature']

  # In most cases, the default data format NCHW instead of NHWC should be
  # used for a significant performance boost on GPU/TPU. NHWC should be used
  # only if the network needs to be run on CPU since the pooling operations
  # are only supported on NHWC.
  if params['data_format'] == 'channels_first':
    assert not params['transpose_input']    # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])

  if params['transpose_input'] and mode != tf.estimator.ModeKeys.PREDICT:
    image_size = params['image_size']
    features = tf.reshape(features, [image_size, image_size, 1, -1])
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  # DropBlock keep_prob for the 4 block groups of ResNet architecture.
  # None means applying no DropBlock at the corresponding block group.
  dropblock_keep_probs = [None] * 4
  if params['dropblock_groups']:
    # Scheduled keep_prob for DropBlock.
    train_steps = tf.cast(params['train_steps'], tf.float32)
    current_step = tf.cast(tf.train.get_global_step(), tf.float32)
    current_ratio = current_step / train_steps
    dropblock_keep_prob = (1 - current_ratio * (
        1 - params['dropblock_keep_prob']))

    # Computes DropBlock keep_prob for different block groups of ResNet.
    dropblock_groups = [int(x) for x in params['dropblock_groups'].split(',')]
    for block_group in dropblock_groups:
      if block_group < 1 or block_group > 4:
        raise ValueError(
            'dropblock_groups should be a comma separated list of integers '
            'between 1 and 4 (dropblcok_groups: {}).'
            .format(params['dropblock_groups']))
      dropblock_keep_probs[block_group - 1] = 1 - (
          (1 - dropblock_keep_prob) / 4.0**(4 - block_group))

  # This nested function allows us to avoid duplicating the logic which
  # builds the network, for different values of --precision.
  def build_network():
    network = resnet_model.resnet_v1(
        resnet_depth=params['resnet_depth'],
        num_classes=params['num_label_classes'],
        dropblock_size=params['dropblock_size'],
        dropblock_keep_probs=dropblock_keep_probs,
        data_format=params['data_format'])
    return network(
        inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  # Compute the summary statistic
  if params['precision'] == 'bfloat16':
    with tf.tpu.bfloat16_scope():
      sum_stat = build_network()
    sum_stat = tf.cast(sum_stat, tf.float32)
  elif params['precision'] == 'float32':
    sum_stat = build_network()

  # Now build a conditional density estimator from this density
  # Defines the chain of bijective transforms
  n = params['num_label_classes']

  net = sum_stat

  # Below is the chain for a MAF
  chain = [ tfp.bijectors.MaskedAutoregressiveFlow(
               shift_and_log_scale_fn=masked_autoregressive_conditional_template(hidden_layers=[128,128],
                                                                                 conditional_tensor=net,
                                                                                 shift_only=False)),
            tfb.Permute(np.arange(n)[::-1]),
            tfp.bijectors.MaskedAutoregressiveFlow(
               shift_and_log_scale_fn=masked_autoregressive_conditional_template(hidden_layers=[128,128],
                                                                                 conditional_tensor=net,
                                                                                 shift_only=False)),
            tfb.Permute(np.arange(n)[::-1]),
            tfp.bijectors.MaskedAutoregressiveFlow(
               shift_and_log_scale_fn=masked_autoregressive_conditional_template(hidden_layers=[128,128],
                                                                                 conditional_tensor=net,
                                                                                 shift_only=True)),
            tfb.Permute(np.arange(n)[::-1]),
            tfp.bijectors.MaskedAutoregressiveFlow(
               shift_and_log_scale_fn=masked_autoregressive_conditional_template(hidden_layers=[128,128],
                                                                                 conditional_tensor=net,
                                                                                 shift_only=True)),
          ]

  # Below is a Neural Spline Flow
  # chain = [
  #   tfb.Affine(scale_identity_multiplier=10),
  #   RealNVP(n//2, bijector_fn=ConditionalNeuralSpline(conditional_tensor=sum_stat, hidden_layers=[128, 128],name='nsf_1')),
  #   tfb.Permute(np.arange(n)[::-1]),
  #   RealNVP(n//2, bijector_fn=ConditionalNeuralSpline(conditional_tensor=sum_stat, hidden_layers=[128, 128],name='nsf_2')),
  #   tfb.Affine(scale_identity_multiplier=0.1)]

  bij = tfb.Chain(chain)
  prior  = tfd.MultivariateNormalDiag(loc=tf.zeros(n), scale_identity_multiplier=1.0)
  distribution = tfd.TransformedDistribution(prior, bijector=bij)

  if mode == tf.estimator.ModeKeys.PREDICT:
    dummy = distribution.log_prob(sum_stat)
    predictions = {
        'dummy': dummy,
        'summary': sum_stat,
        'samples': distribution.sample(256) # TODO: find a better way to sample
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'inference': tf.estimator.export.PredictOutput(predictions)
        })

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  batch_size = params['batch_size']   # pylint: disable=unused-variable

  # Add a little bit of scatter to the labels to smooth out the distribution
  if (params['label_smoothing'] > 0.) and (mode == tf.estimator.ModeKeys.TRAIN):
    labels += params['label_smoothing']*tf.random_normal(shape=[batch_size, n])

  # Compute loss function with some L2 regularization
  loglik = - tf.reduce_mean(distribution.log_prob(labels),axis=0)

  # Add weight decay to the loss for non-batch-normalization variables.
  if params['enable_lars']:
    loss = loglik
  else:
    loss = loglik + params['weight_decay'] * tf.add_n([
        tf.nn.l2_loss(v)
        for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name
    ])

  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_global_step()
    steps_per_epoch = params['num_train_images'] / params['train_batch_size']
    current_epoch = (tf.cast(global_step, tf.float32) /
                     steps_per_epoch)
    # LARS is a large batch optimizer. LARS enables higher accuracy at batch 16K
    # and larger batch sizes.
    if params['enable_lars']:
      learning_rate = 0.0
      optimizer = lars_util.init_lars_optimizer(current_epoch, params)
    else:
      learning_rate = learning_rate_schedule(params, current_epoch)
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=params['momentum'],
          use_nesterov=True)
    if params['use_tpu']:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    if not params['skip_host_call']:
      def host_call_fn(gs, loss, lr, ce):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed params['iterations_per_loop'] times after
        # one TPU loop is finished, setting max_queue value to the same as
        # number of iterations will make the summary writer only flush the data
        # to storage once per loop.
        with tf2.summary.create_file_writer(
            FLAGS.model_dir,
            max_queue=params['iterations_per_loop']).as_default():
          with tf2.summary.record_if(True):
            tf2.summary.scalar('loss', loss[0], step=gs)
            tf2.summary.scalar('learning_rate', lr[0], step=gs)
            tf2.summary.scalar('current_epoch', ce[0], step=gs)

          return tf.summary.all_v2_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(loss, [1])
      lr_t = tf.reshape(learning_rate, [1])
      ce_t = tf.reshape(current_epoch, [1])

      host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

  else:
    train_op = None

  eval_metrics = None
  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics)


def main(unused_argv):
  params = params_dict.ParamsDict(
      resnet_config.RESNET_CFG, resnet_config.RESNET_RESTRICTIONS)
  params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=True)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)

  params = flags_to_params.override_params_from_input_flags(params, FLAGS)

  params.validate()
  params.lock()

  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  if params.use_async_checkpointing:
    save_checkpoints_steps = None
  else:
    save_checkpoints_steps = max(5000, params.iterations_per_loop)
  config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      log_step_count_steps=FLAGS.log_step_count_steps,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))),
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=params.iterations_per_loop,
          num_shards=params.num_cores,
          per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
          .PER_HOST_V2))  # pylint: disable=line-too-long

  resnet_classifier = tf.estimator.tpu.TPUEstimator(
      use_tpu=params.use_tpu,
      model_fn=resnet_model_fn,
      config=config,
      params=params.as_dict(),
      train_batch_size=params.train_batch_size,
      eval_batch_size=params.eval_batch_size,
      export_to_tpu=FLAGS.export_to_tpu)

  assert (params.precision == 'bfloat16' or
          params.precision == 'float32'), (
              'Invalid value for precision parameter; '
              'must be bfloat16 or float32.')
  tf.logging.info('Precision: %s', params.precision)
  use_bfloat16 = params.precision == 'bfloat16'

  tf.logging.info('Using dataset: %s', FLAGS.data_dir)
  imagenet_train, imagenet_eval = [
          DESSVInput(  # pylint: disable=g-complex-comprehension
          is_training=is_training,
          data_dir=FLAGS.data_dir,
          transpose_input=params.transpose_input,
          cache=params.use_cache and is_training,
          image_size=params.image_size,
          num_parallel_calls=params.num_parallel_calls,
          use_bfloat16=use_bfloat16)
      for is_training in [True, False]
  ]

  steps_per_epoch = params.num_train_images // params.train_batch_size

  if FLAGS.mode == 'eval':
    eval_steps = params.num_eval_images // params.eval_batch_size

    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d',
                        eval_results, elapsed_time)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= params.train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d', current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint', ckpt)

  else:   # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    try:
      current_step = tf.train.load_variable(FLAGS.model_dir,
                                            tf.GraphKeys.GLOBAL_STEP)
    except (TypeError, ValueError, tf.errors.NotFoundError):
      current_step = 0
    steps_per_epoch = params.num_train_images // params.train_batch_size
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.',
                    params.train_steps,
                    params.train_steps / steps_per_epoch,
                    current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      hooks = []
      if params.use_async_checkpointing:
        try:
          from tensorflow.contrib.tpu.python.tpu import async_checkpoint  # pylint: disable=g-import-not-at-top
        except ImportError as e:
          logging.exception(
              'Async checkpointing is not supported in TensorFlow 2.x')
          raise e

        hooks.append(
            async_checkpoint.AsyncCheckpointSaverHook(
                checkpoint_dir=FLAGS.model_dir,
                save_steps=max(5000, params.iterations_per_loop)))
      if FLAGS.profile_every_n_steps > 0:
        hooks.append(
            tpu_profiler_hook.TPUProfilerHook(
                save_steps=FLAGS.profile_every_n_steps,
                output_dir=FLAGS.model_dir, tpu=FLAGS.tpu)
            )
      resnet_classifier.train(
          input_fn=imagenet_train.input_fn,
          max_steps=params.train_steps,
          hooks=hooks)

    else:
      assert FLAGS.mode == 'train_and_eval'
      while current_step < params.train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              params.train_steps)
        resnet_classifier.train(
            input_fn=imagenet_train.input_fn, max_steps=int(next_checkpoint))
        current_step = next_checkpoint

        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=params.num_eval_images // params.eval_batch_size)
        tf.logging.info('Eval results at step %d: %s',
                        next_checkpoint, eval_results)

      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                      params.train_steps, elapsed_time)

    if FLAGS.export_dir is not None:
      # The guide to serve a exported TensorFlow model is at:
      #    https://www.tensorflow.org/serving/serving_basic
      tf.logging.info('Starting to export model.')
      # exporter = hub.LatestModuleExporter("tf_hub", image_serving_input_fn)
      # exporter.export(resnet_classifier, FLAGS.export_dir, resnet_classifier.latest_checkpoint())
      export_path = resnet_classifier.export_saved_model(
          export_dir_base=FLAGS.export_dir,
          serving_input_receiver_fn=image_serving_input_fn)
      if FLAGS.add_warmup_requests:
        inference_warmup.write_warmup_requests(
            export_path,
            FLAGS.model_name,
            params.image_size,
            batch_sizes=FLAGS.inference_batch_sizes,
            image_format='JPEG')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.disable_v2_behavior()
  app.run(main)
