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
"""ImageNet preprocessing for ResNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
IMAGE_SIZE = 224 # At 1.34 arcmin, that's 5x5 deg^2

def _decode_rescale_and_crop(image_bytes, is_training=False):
  """Make a center crop of IMAGE_SIZE."""
  # First decode raw string into image
  image = tf.decode_raw(image_bytes, tf.float32, little_endian=False)

  # Make sure the image has the correct shape
  image = tf.reshape(image, (1, 448, 448, 1))

  # Dowsample input images with averaging
  # image = tf.image.resize_area(image, (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE))

  # During training, randomly extract 5x5 sq deg patch
  if is_training:
      image = tf.random_crop(image, size=(1,IMAGE_SIZE,IMAGE_SIZE,1))
  else:
      # Extract a 5x5 deg patch at the corner of the input image
      image = tf.image.crop_to_bounding_box(image,0,0,IMAGE_SIZE,IMAGE_SIZE)

  image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 1))
  return image

def _flip(image):
  """Random horizontal image flip."""
  # Random flips
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  return image

def preprocess_for_train(image_bytes):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_rescale_and_crop(image_bytes, is_training=True)
  image = _flip(image)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
  return image

def preprocess_for_eval(image_bytes):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_rescale_and_crop(image_bytes, is_training=False)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
  return image


def preprocess_image(image_bytes, is_training=False):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.

  Returns:
    A preprocessed image `Tensor`.
  """
  if is_training:
    return preprocess_for_train(image_bytes)
  else:
    return preprocess_for_eval(image_bytes)
