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
import optax
import tensorflow as tf
import wandb

import input_pipeline
import models
import pin_util

FLAGS = flags.FLAGS
_INPUT_FILE = flags.DEFINE_string(
    "input_file",
    "STL-Dataset/fashion.json",
    "Input cat json file.")
_IMAGE_DIRECTORY = flags.DEFINE_string(
    "image_dir",
    "artifacts/shop_the_look:v1",
    "Directory containing downloaded images from the shop the look dataset.")
_NUM_NEG = flags.DEFINE_integer(
    "num_neg", 5, "How many negatives per positive."
)
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
_REGULARIZATION = flags.DEFINE_float("regularization", 0.1, "Regularization.")
_OUTPUT_SIZE = flags.DEFINE_integer("output_size", 32, "Size of output embedding.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 16, "Batch size.")
_LOG_EVERY_STEPS = flags.DEFINE_integer("log_every_steps", 100, "Log every this step.")
_EVAL_EVERY_STEPS = flags.DEFINE_integer("eval_every_steps", 2000, "Eval every this step.")
_CHECKPOINT_EVERY_STEPS = flags.DEFINE_integer("checkpoint_every_steps", 100000, "Checkpoint every this step.")
_MAX_STEPS = flags.DEFINE_integer("max_steps", 30000, "Max number of steps.")
_WORKDIR = flags.DEFINE_string("work_dir", "/tmp", "Work directory.")
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "pinterest_stl_model", "Model name.")
_RESTORE_CHECKPOINT = flags.DEFINE_bool("restore_checkpoint", False, "If true, restore.")


def generate_triplets(
    scene_product: Sequence[Tuple[str, str]],
    num_neg: int) -> Sequence[Tuple[str, str, str]]:
    """Generate positive and negative triplets."""
    count = len(scene_product)
    train = []
    test = []
    key = jax.random.PRNGKey(0)
    for i in range(count):
        scene, pos = scene_product[i]
        is_test = i % 10 == 0
        key, subkey = jax.random.split(key)
        neg_indices = jax.random.randint(subkey, [num_neg], 0, count - 1)
        for neg_idx in neg_indices:
            _, neg = scene_product[neg_idx]
            if is_test:
                test.append((scene, pos, neg))
            else:
                train.append((scene, pos, neg))
    return train, test

def train_step(state, scene, pos_product, neg_product, regularization, batch_size):
    def loss_fn(params):
        result, new_model_state = state.apply_fn(
            params,
            scene, pos_product, neg_product, True,
            mutable=['batch_stats'])
        triplet_loss = jnp.sum(nn.relu(1.0 + result[1] - result[0]))
        def reg_fn(embed):
            return nn.relu(jnp.sqrt(jnp.sum(jnp.square(embed), axis=-1)) - 1.0)
        reg_loss = reg_fn(result[2]) + reg_fn(result[3]) + reg_fn(result[4])
        reg_loss = jnp.sum(reg_loss)
        return (triplet_loss + regularization * reg_loss) / batch_size
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def eval_step(state, scene, pos_product, neg_product):
    def loss_fn(params):
        result, new_model_state = state.apply_fn(
            state.params,
            scene, pos_product, neg_product, True,
            mutable=['batch_stats'])
        # Use a fixed margin for the eval.
        triplet_loss = jnp.sum(nn.relu(1.0 + result[1] - result[0]))
        return triplet_loss
    
    loss = loss_fn(state.params)    
    return loss

def shuffle_array(key, x):
    """Deterministic string shuffle."""
    num = len(x)
    to_swap = jax.random.randint(key, [num], 0, num - 1)
    return [x[t] for t in to_swap]

def main(argv):
    """Main function."""
    del argv  # Unused.
    config = {
        "learning_rate" : _LEARNING_RATE.value,
        "regularization" : _REGULARIZATION.value,
        "output_size" : _OUTPUT_SIZE.value
    }

    run = wandb.init(
        config=config,
        project="recsys-pinterest"
    )

    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.enable_eager_execution()
    logging.info("Image dir %s, input file %s", _IMAGE_DIRECTORY.value, _INPUT_FILE.value)
    scene_product = pin_util.get_valid_scene_product(_IMAGE_DIRECTORY.value, _INPUT_FILE.value)
    logging.info("Found %d valid scene product pairs." % len(scene_product))

    train, test = generate_triplets(scene_product, _NUM_NEG.value)
    num_train = len(train)
    num_test = len(test)
    logging.info("Train triplets %d", num_train)
    logging.info("Test triplets %d", num_test)
    
     # Random shuffle the train.
    key = jax.random.PRNGKey(0)
    train = shuffle_array(key, train)
    test = shuffle_array(key, test)
    train = np.array(train)
    test = np.array(test)
 
    train_ds = input_pipeline.create_dataset(train).repeat()
    train_ds = train_ds.batch(_BATCH_SIZE.value).prefetch(tf.data.AUTOTUNE)

    test_ds = input_pipeline.create_dataset(test).repeat()
    test_ds = test_ds.batch(_BATCH_SIZE.value)

    stl = models.STLModel(output_size=wandb.config.output_size)
    train_it = train_ds.as_numpy_iterator()
    test_it = test_ds.as_numpy_iterator()
    x = next(train_it)
    key, subkey = jax.random.split(key)
    params = stl.init(subkey, x[0], x[1], x[2])
    tx = optax.adam(learning_rate=wandb.config.learning_rate)
    state = train_state.TrainState.create(
        apply_fn=stl.apply, params=params, tx=tx)
    if _RESTORE_CHECKPOINT.value:
        state = checkpoints.restore_checkpoint(_WORKDIR.value, state)

    train_step_fn = jax.jit(train_step)
    eval_step_fn = jax.jit(eval_step)

    losses = []
    init_step = state.step
    logging.info("Starting at step %d", init_step)
    regularization = wandb.config.regularization
    batch_size = _BATCH_SIZE.value
    eval_steps = int(num_test / batch_size)
    for i in range(init_step, _MAX_STEPS.value + 1):
        batch = next(train_it)
        scene = batch[0]
        pos_product = batch[1]
        neg_product = batch[2]

        state, loss = train_step_fn(
            state, scene, pos_product, neg_product, regularization, batch_size)
        losses.append(loss)
        if i % _CHECKPOINT_EVERY_STEPS.value == 0 and i > 0:
            logging.info("Saving checkpoint")
            checkpoints.save_checkpoint(_WORKDIR.value, state, state.step, keep=3)
        metrics = {
            "step" : state.step
        }
        if i % _EVAL_EVERY_STEPS.value == 0 and i > 0:
            eval_loss = []
            for j in range(eval_steps):
                ebatch = next(test_it)
                escene = ebatch[0]
                epos_product = ebatch[1]
                eneg_product = ebatch[2]
                loss = eval_step_fn(state, escene, epos_product, eneg_product)
                eval_loss.append(loss)
            eval_loss = jnp.mean(jnp.array(eval_loss)) / batch_size
            metrics.update({"eval_loss" : eval_loss})
        if i % _LOG_EVERY_STEPS.value == 0 and i > 0:
            mean_loss = jnp.mean(jnp.array(losses))
            losses = []
            metrics.update({"train_loss" : mean_loss})
            wandb.log(metrics)
            logging.info(metrics)

    logging.info("Saving as %s", _MODEL_NAME.value)
    data = flax.serialization.to_bytes(state)
    artifact = wandb.Artifact(name=_MODEL_NAME.value, type="model")
    with artifact.new_file("pinterest_stl.model", "wb") as f:
        f.write(data)
    run.log_artifact(artifact)


if __name__ == "__main__":
    app.run(main)
