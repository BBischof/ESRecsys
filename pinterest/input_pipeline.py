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

def read_and_decode_image(x):
  x = (tf.io.read_file(x[0]), tf.io.read_file(x[1]), tf.io.read_file(x[2]))
  x = (tf.io.decode_jpeg(x[0]), tf.io.decode_jpeg(x[1]), tf.io.decode_jpeg(x[2]))
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
    ds = ds.map(read_and_decode_image)
    return ds