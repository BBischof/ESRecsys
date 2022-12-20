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
  Generates embedding files given a model and a catalog.
"""

import random
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
_INPUT_FILE = flags.DEFINE_string("input_file", None, "Input cat json file.")
_IMAGE_DIRECTORY = flags.DEFINE_string(
    "image_dir",
    None,
    "Directory containing downloaded images from the shop the look dataset.")
_OUTDIR = flags.DEFINE_string("out_dir", "/tmp", "Output directory.")
_OUTPUT_SIZE = flags.DEFINE_integer("output_size", 64, "Size of embeddings.") 
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    None,
    "Model name.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8, "Batch size.")

# Required flag.
flags.mark_flag_as_required("model_name")
flags.mark_flag_as_required("image_dir")


def main(argv):
    """Main function."""
    del argv  # Unused.

    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.enable_eager_execution()
    scene_product = pin_util.get_valid_scene_product(_IMAGE_DIRECTORY.value, _INPUT_FILE.value)
    logging.info("Found %d valid scene product pairs." % len(scene_product))
    unique_scenes = set(x[0] for x in scene_product)
    unique_products = set(x[1] for x in scene_product)
    logging.info("Found %d unique scenes.", len(unique_scenes))
    logging.info("Found %d unique products.", len(unique_products))
    unique_scenes = np.array(list(unique_scenes))
    unique_products = np.array(list(unique_products))

    model = models.STLModel(output_size=_OUTPUT_SIZE.value)
    state = None
    logging.info("Attempting to read model %s", _MODEL_NAME.value)
    with open(_MODEL_NAME.value, "rb") as f:
        data = f.read()
        state = flax.serialization.from_bytes(model, data)
    assert(state != None)

    @jax.jit
    def get_scene_embed(x):
      return model.apply(state["params"], x, method=models.STLModel.get_scene_embed)
    @jax.jit
    def get_product_embed(x):
      return model.apply(state["params"], x, method=models.STLModel.get_product_embed)

    ds = tf.data.Dataset.from_tensor_slices(unique_scenes).map(input_pipeline.process_image_with_id)
    ds = ds.batch(_BATCH_SIZE.value, drop_remainder=True)
    it = ds.as_numpy_iterator()
    scene_dict = {}
    count = 0
    for id, image in it:
      count = count + 1
      if count % 100 == 0:
        logging.info("Created %d scene embeddings", count * _BATCH_SIZE.value)
      result = get_scene_embed(image)
      for i in range(_BATCH_SIZE.value):
        current_id = id[i].decode("utf-8")
        tmp = np.array(result[i])
        current_result = [float(tmp[j]) for j in range(tmp.shape[0])]
        scene_dict.update({current_id : current_result})
    scene_filename = os.path.join(_OUTDIR.value, "scene_embed.json")
    with open(scene_filename, "w") as scene_file:
      json.dump(scene_dict, scene_file)

    ds = tf.data.Dataset.from_tensor_slices(unique_products).map(input_pipeline.process_image_with_id)
    ds = ds.batch(_BATCH_SIZE.value, drop_remainder=True)
    it = ds.as_numpy_iterator()
    product_dict = {}
    count = 0
    for id, image in it:
      count = count + 1
      if count % 100 == 0:
        logging.info("Created %d product embeddings", count * _BATCH_SIZE.value)
      result = get_product_embed(image)
      for i in range(_BATCH_SIZE.value):
        current_id = id[i].decode("utf-8")
        tmp = np.array(result[i])
        current_result = [float(tmp[j]) for j in range(tmp.shape[0])]
        product_dict.update({current_id : current_result})
    product_filename = os.path.join(_OUTDIR.value, "product_embed.json")
    with open(product_filename, "w") as product_file:
      json.dump(product_dict, product_file)

if __name__ == "__main__":
    app.run(main)
