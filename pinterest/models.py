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

"""Models for the shop the look content recommender."""

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

class CNN(nn.Module):
    """Simple CNN."""
    filters : Sequence[int]
    output_size : int

    @nn.compact
    def __call__(self, x, train: bool = True):
        for filter in self.filters:
            x = nn.Conv(filter, (3, 3), (2, 2))(x)
            x = nn.BatchNorm(use_running_average=not train, use_bias=False)(x)
            x = nn.swish(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.output_size, dtype=jnp.float32)(x)
        return x

class STLModel(nn.Module):
    """Shop the look model that takes in a scene and item and computes a similarity for them."""
    def setup(self):
        self.scene_cnn = CNN(filters=[8, 16, 32, 64], output_size=256)
        self.product_cnn = CNN(filters=[8, 16, 32, 64], output_size=256)

    def __call__(self, scene, product, train: bool = True):
        scene_embed = self.scene_cnn(scene, train)
        product_embed = self.product_cnn(product, train)
        result = scene_embed * product_embed
        result = jnp.sum(result, axis=-1)
        return result