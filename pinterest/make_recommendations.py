#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
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

"""
  Given embedding files makes recommendations.
"""

import json
import os
from typing import Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import flax
from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import wandb

import input_pipeline
import models
import pin_util

FLAGS = flags.FLAGS
_PRODUCT_EMBED_ = flags.DEFINE_string("product_embed", None, "Product embedding json.")
_SCENE_EMBED_ = flags.DEFINE_string("scene_embed", None, "Scene embedding json.")

# Required flag.
flags.mark_flag_as_required("product_embed")
flags.mark_flag_as_required("scene_embed")


def main(argv):
    """Main function."""
    del argv  # Unused.

    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.enable_eager_execution()
    with open(_PRODUCT_EMBED_.value, "r") as f:
      product_dict = json.load(f)
    with open(_SCENE_EMBED_.value, "r") as f:
      scene_dict = json.load(f)
    print(product_dict)

if __name__ == "__main__":
    app.run(main)
