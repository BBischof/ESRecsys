# Copyright 2022 Hector Yee, Bryan Bischoff
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

from typing import Sequence, Tuple, Set

import numpy as np
import tensorflow as tf

def normalize_image(img):
  img = tf.cast(img, dtype=tf.float32)
  img = (img / 255.0) - 0.5
  return img

def process_image(x):
  x = tf.io.read_file(x)
  x = tf.io.decode_jpeg(x)
  x = normalize_image(x)
  return x

def process_triplet(x):
  x = (process_image(x[0]), process_image(x[1]), process_image(x[2]))
  return x

def create_dataset(
    triplet: Sequence[Tuple[str, str, str]],
    train: bool):
    """Creates train and test splits from a product_scene sequence and an all products set.

    Args:
      triplet: filenames of scene, positive product, negative product.
      train: if this is training or not.
    """
    ds = tf.data.Dataset.from_tensor_slices(triplet)
    ds = ds.map(process_triplet)
    return ds