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

import jax
import tensorflow as tf

def create_dataset(
    scene_product: Sequence[Tuple[str, str]],
    all_products: Set[str],
    train: bool):
    """Creates train and test splits from a product_scene sequence and an all products set.

    Args:
      scene_product: ID of scene and ID of product that goes with scene.
      all_products: ID of all products.

    """