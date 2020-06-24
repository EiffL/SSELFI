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
"""Efficient DES SV input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import os
from absl import logging
import tensorflow.compat.v1 as tf

def preprocess_image(image_bytes, is_training=False, use_bfloat16=False):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed image `Tensor`.
  """
  # TODO: Add data augmentation during training
  image = tf.io.decode_raw(image_bytes, out_type=tf.float32)
  image = tf.reshape(image, [256, 256, 1])
  if use_bfloat16:
    image = tf.cast(image, tf.bfloat16)
  # Apply central cropping to 224x224 size to avoid issues
  image = tf.image.resize_with_crop_or_pad(image, 224, 224)

  # Apply clipping to guard against crazy KS values
  image = tf.clip_by_value(image,-0.5, 0.5)
  return image

def image_serving_input_fn():
  """Serving input fn for raw images."""
  input_image = tf.placeholder(
      shape=[None, 224, 224, 1],
      dtype=tf.float32,
  )
  return tf.estimator.export.TensorServingInputReceiver(
      features=input_image, receiver_tensors=input_image)

class DESInput(object):
  """Base class for DES input_fn generator.

  Attributes:
    image_preprocessing_fn: function to preprocess images
    is_training: `bool` for whether the input is for training
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    transpose_input: 'bool' for whether to use the double transpose trick
    image_size: size of images
    num_parallel_calls: `int` for the number of parallel threads.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               image_size=224,
               transpose_input=False,
               num_parallel_calls=8):
    self.image_preprocessing_fn = preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.transpose_input = transpose_input
    self.image_size = image_size
    self.num_parallel_calls = num_parallel_calls

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_size])))
      images = tf.reshape(images, [-1])
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size, 2])))
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size, 2])))

    return images, labels

  def dataset_parser(self, value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    # First, let's define what fields we are expecting
    data_fields = {
        "KS/encoded": tf.io.FixedLenFeature((), tf.string),
        "KS/format": tf.io.FixedLenFeature((), tf.string),
        "power/encoded": tf.io.FixedLenFeature((), tf.string),
        "power/format": tf.io.FixedLenFeature((), tf.string),
        "peaks/encoded": tf.io.FixedLenFeature((), tf.string),
        "peaks/format": tf.io.FixedLenFeature((), tf.string),
        "params/om": tf.io.FixedLenFeature((), tf.float32),
        "params/sigma8": tf.io.FixedLenFeature((), tf.float32),
        "params/S8": tf.io.FixedLenFeature((), tf.float32),
    }

    parsed = tf.io.parse_single_example(value, data_fields)
    image_bytes = tf.reshape(parsed['KS/encoded'], shape=[])

    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        use_bfloat16=self.use_bfloat16)

    label = [parsed['params/om'], parsed['params/S8']]
    return image, label

  @abc.abstractmethod
  def make_source_dataset(self, index, num_hosts):
    """Makes dataset of serialized TFExamples.

    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.

    If self.is_training, the dataset should be infinite.

    Args:
      index: current host index.
      num_hosts: total number of hosts.

    Returns:
      A `tf.data.Dataset` object.
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """

    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.estimator.tpu.RunConfig for details.
    batch_size = params['batch_size']

    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      current_host = 0
      num_hosts = 1

    dataset = self.make_source_dataset(current_host, num_hosts)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self.dataset_parser,
            batch_size=batch_size,
            num_parallel_batches=self.num_parallel_calls,
            drop_remainder=True))

    # Transpose for performance on TPU
    if self.transpose_input:
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=self.num_parallel_calls)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class DESSVInput(DESInput):
  """Generates DES SV input_fn from a series of TFRecord files.
  """

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               data_dir,
               image_size=224,
               num_parallel_calls=8,
               cache=False,
               dataset_split=None,
               shuffle_shards=True):
    """Create an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data; if
        'null' (the literal string 'null') or implicitly False then construct a
        null pipeline, consisting of empty images and blank labels.
      image_size: `int` image height and width.
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if true, fill the dataset by repeating from its cache
      dataset_split: If provided, must be one of 'train' or 'validation' and
        specifies the dataset split to read, overriding the default set by
        is_training. In this case, is_training specifies whether the data is
        augmented.
      shuffle_shards: Whether to shuffle the dataset shards.
    """
    super(DESSVInput, self).__init__(
        is_training=is_training,
        image_size=image_size,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input)
    self.data_dir = data_dir
    # TODO(b/112427086):  simplify the choice of input source
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache
    self.dataset_split = dataset_split
    self.shuffle_shards = shuffle_shards

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).

    Args:
      data: element of a dataset, ignored in this method, since it produces the
        same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 1],
                    tf.bfloat16 if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(DESSVInput, self).dataset_parser(value)

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    if not self.data_dir:
      tf.logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    # Create the list of files for the dataset
    data_list = []
    if self.is_training:
      # For training we use 90% of the dataset
      for i in range(74):
        for n in range(9):
          data_list += [os.path.join(self.data_dir, 'training-%02d-%05d-of-00010'%(i, n))]
          from random import shuffle
          shuffle(data_list) # Just to try to mix it as best we can
    else:
      # For testing we use 10% of the dataset
      for i in range(74):
          data_list += [os.path.join(self.data_dir, 'training-%02d-%05d-of-00010'%(i, 9))]

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = tf.data.Dataset.list_files(
        data_list, shuffle=self.shuffle_shards)
    dataset = dataset.shard(num_hosts, index)

    if self.is_training and not self.cache:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            fetch_dataset, cycle_length=64, sloppy=True))

    if self.cache:
      dataset = dataset.cache().apply(
          tf.data.experimental.shuffle_and_repeat(1024 * 16))
    else:
      dataset = dataset.shuffle(1024)
    return dataset
