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
  Generates a html file of random product recommendations from a json catalog file.
"""

import random
import json
import os
from typing import Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import wandb

import input_pipeline
import models

FLAGS = flags.FLAGS
_INPUT_FILE = flags.DEFINE_string("input_file", None, "Input cat json file.")
_IMAGE_DIRECTORY = flags.DEFINE_string(
    "image_dir",
    None,
    "Directory containing downloaded images from the shop the look dataset.")
_NUM_NEG = flags.DEFINE_integer(
    "num_neg", 5, "How many negatives per positive."
)
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8, "Batch size.")
_SHUFFLE_SIZE = flags.DEFINE_integer("shuffle_size", 100, "Shuffle size.")
_LOG_EVERY_STEPS = flags.DEFINE_integer("log_every_steps", 100, "Log every this step.")
_CHECKPOINT_EVERY_STEPS = flags.DEFINE_integer("checkpoint_every_steps", 1000, "Checkpoint every this step.")
_MAX_STEPS = flags.DEFINE_integer("max_steps", 10000, "Max number of steps.")
_WORKDIR = flags.DEFINE_string("work_dir", "/tmp", "Work directory.")

# Required flag.
flags.mark_flag_as_required("input_file")
flags.mark_flag_as_required("image_dir")

def id_to_filename(id: str) -> str:
    filename = os.path.join(
        _IMAGE_DIRECTORY.value,
        id + ".jpg")
    return filename

def is_valid_file(fname):
    return os.path.exists(fname) and os.path.getsize(fname) > 0

def get_valid_scene_product(input_file: str) -> Sequence[Tuple[str, str]]:
    """
      Reads in the Shop the look json file and returns a pair of scene and matching products.
    """
    scene_product = []
    with open(input_file, "r") as f:
        data = f.readlines()
        for line in data:
            row = json.loads(line)
            scene = id_to_filename(row["scene"])
            product = id_to_filename(row["product"])
            if is_valid_file(scene) and is_valid_file(product):
                scene_product.append([scene, product])
    return scene_product

def generate_triplets(
    scene_product: Sequence[Tuple[str, str]],
    num_neg: int) -> Sequence[Tuple[str, str, str]]:
    """Generate positive and negative triplets."""
    count = len(scene_product)
    train = []
    test = []
    for i in range(count):
        scene, pos = scene_product[i]
        is_train = i % 10 != 0
        for j in range(num_neg):
            neg_idx = random.randint(0, count - 1)
            _, neg = scene_product[neg_idx]
            if is_train:
                train.append((scene, pos, neg))
            else:
                test.append((scene, pos, neg))
    return np.array(train), np.array(test)

def train_step(state, scene, pos_product, neg_product):
    def loss_fn(params):
        result, new_model_state = state.apply_fn(
            params,
            scene, pos_product, neg_product, True,
            mutable=['batch_stats'])
        loss = jnp.mean(nn.relu(1.0 - result[0] + result[1]))
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def main(argv):
    """Main function."""
    del argv  # Unused.
    run = wandb.init(
        config=FLAGS,
        project="recsys-pinterest"
    )

    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.enable_eager_execution()
    scene_product = get_valid_scene_product(_INPUT_FILE.value)
    logging.info("Found %d valid scene product pairs." % len(scene_product))

    train, test = generate_triplets(scene_product, _NUM_NEG.value)

    train_ds = input_pipeline.create_dataset(train).repeat()
    train_ds = train_ds.shuffle(_SHUFFLE_SIZE.value)
    train_ds = train_ds.batch(_BATCH_SIZE.value).prefetch(tf.data.AUTOTUNE)
    test_ds = input_pipeline.create_dataset(test)

    stl = models.STLModel()
    train_it = train_ds.as_numpy_iterator()
    x = next(train_it)
    params = stl.init(jax.random.PRNGKey(0), x[0], x[1], x[2])
    tx = optax.adam(learning_rate=_LEARNING_RATE.value)
    state = train_state.TrainState.create(
        apply_fn=stl.apply, params=params, tx=tx)
    state = checkpoints.restore_checkpoint(_WORKDIR.value, state)

    train_step_fn = jax.jit(train_step)

    losses = []
    init_step = state.step
    logging.info("Starting at step %d", init_step)
    for i in range(init_step, _MAX_STEPS.value):
        batch = next(train_it)
        scene = batch[0]
        pos_product = batch[1]
        neg_product = batch[2]

        state, loss = train_step_fn(state, scene, pos_product, neg_product)
        losses.append(loss)
        if i % _CHECKPOINT_EVERY_STEPS.value == 0:
            logging.info("Saving checkpoint")
            checkpoints.save_checkpoint(_WORKDIR.value, state, state.step, keep=3)
        if i % _LOG_EVERY_STEPS.value == 0:
            mean_loss = jnp.mean(jnp.array(losses))
            metrics = {
                "train_loss" : mean_loss,
                "step" : state.step
            }
            wandb.log(metrics)
            logging.info(metrics)



if __name__ == "__main__":
    app.run(main)
