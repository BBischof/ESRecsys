#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
# Copyright 2023 Hector Yee, Bryan Bischoff
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
  Trains a model for the Spotify million playlist data set.
"""

import json
import os
from typing import Sequence, Tuple
import random

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
#import wandb

import input_pipeline
import models

FLAGS = flags.FLAGS
_TRAIN_PATTERN = flags.DEFINE_string(
    "train_pattern",
    "data/training/00[1-9]??.tfrecord",
    "Training pattern.")
_TEST_PATTERN = flags.DEFINE_string(
    "test_pattern",
    "data/training/000??.tfrecord",
    "Training pattern.")
_ALL_TRACKS =  flags.DEFINE_string(
    "all_tracks",
    "data/training/all_tracks.json",
    "Location of track database.")
_DICTIONARY_PATH = flags.DEFINE_string("dictionaries", "data/dictionaries", "Dictionary path.")

_NUM_NEGATIVES = flags.DEFINE_integer("num_negatives", 64, "Number of negatives to sample.")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
_REGULARIZATION = flags.DEFINE_float("regularization", 10.0, "Regularization (max l2 norm squared).")
_FEATURE_SIZE = flags.DEFINE_integer("feature_size", 64, "Size of output embedding.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 16, "Batch size.")
_LOG_EVERY_STEPS = flags.DEFINE_integer("log_every_steps", 100, "Log every this step.")
_EVAL_EVERY_STEPS = flags.DEFINE_integer("eval_every_steps", 2000, "Eval every this step.")
_EVAL_STEPS = flags.DEFINE_integer("eval_steps", 1000, "Eval this number of entries.")
_CHECKPOINT_EVERY_STEPS = flags.DEFINE_integer("checkpoint_every_steps", 100000, "Checkpoint every this step.")
_MAX_STEPS = flags.DEFINE_integer("max_steps", 30000, "Max number of steps.")
_WORKDIR = flags.DEFINE_string("work_dir", "/tmp", "Work directory.")
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "spotify_mpl_model", "Model name.")
_RESTORE_CHECKPOINT = flags.DEFINE_bool("restore_checkpoint", False, "If true, restore.")


def train_step(state, x, regularization):
    def loss_fn(params):
        result = state.apply_fn(
            params,
            x["track_context"], x["album_context"], x["artist_context"],
            x["next_track"], x["next_album"], x["next_artist"],
            x["neg_track"], x["neg_album"], x["neg_artist"])
        pos_affinity, neg_affinity, all_embeddings_l2 = result
        neg_affinity = jnp.mean(neg_affinity)
        pos_affinity = jnp.mean(pos_affinity)        
        triplet_loss = nn.relu(1.0 + neg_affinity - pos_affinity)
        reg_loss = jnp.sum(nn.relu(all_embeddings_l2 - regularization))
        return triplet_loss + reg_loss
    
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

def sample_negative(x, key, num_negatives, num_tracks, num_albums, num_artists):
    """Generate random negatives."""
    key, subkey = jax.random.split(key)
    # It is unlikely for a random negative to be in the positves
    # so for simplicity just sample at random.
    x["neg_track"] = jax.random.randint(subkey, [num_negatives], 0, num_tracks - 1)
    x["neg_album"] = jax.random.randint(subkey, [num_negatives], 0, num_albums - 1)
    x["neg_artist"] = jax.random.randint(subkey, [num_negatives], 0, num_artists - 1)
    return key

def check_inputs(x, num_tracks, num_albums, num_artists):
    """Assert checks on the validity of the inputs."""    
    assert(jnp.max(x["track_context"]) <= num_tracks)
    assert(jnp.max(x["album_context"]) <= num_albums)
    assert(jnp.max(x["artist_context"]) <= num_artists)

def main(argv):
    """Main function."""
    del argv  # Unused.

    # Uncomment to debug nans.
    #jax.config.update("jax_debug_nans", True)

    track_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "track_uri_dict.json")
    num_tracks = len(track_uri_dict)
    print("%d tracks loaded" % num_tracks)
    album_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "album_uri_dict.json")
    num_albums = len(album_uri_dict)
    print("%d albums loaded" % num_albums)
    artist_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "artist_uri_dict.json")
    num_artists = len(artist_uri_dict)
    print("%d artists loaded" % num_artists)
    all_tracks_dict, all_tracks_features = input_pipeline.load_all_tracks(
        _ALL_TRACKS.value, track_uri_dict, album_uri_dict, artist_uri_dict)
    print("10 sample tracks")
    for i in range(10):
        print("Track %d" % i)
        print(all_tracks_dict[i])
        print(all_tracks_features[i])
    num_tracks = len(track_uri_dict)

    # Make the all tracks for the evaluation.
    all_tracks, all_albums, all_artists = input_pipeline.make_all_tracks_numpy(all_tracks_features)
    print("All tracks features top 10")
    print(all_tracks[:10])
    print(all_albums[:10])
    print(all_artists[:10])

    config = {
        "learning_rate" : _LEARNING_RATE.value,
        "regularization" : _REGULARIZATION.value,
        "feature_size" : _FEATURE_SIZE.value
    }

    #run = wandb.init(
    #    config=config,
    #    project="recsys-spotify"
    #)
    #config = wandb.config

    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.enable_eager_execution()
    
     # Random shuffle the train.
    key = jax.random.PRNGKey(0)

    train_ds = input_pipeline.create_dataset(_TRAIN_PATTERN.value).repeat()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = input_pipeline.create_dataset(_TEST_PATTERN.value).repeat()
    test_ds = test_ds

    spotify = models.SpotifyModel(feature_size=config["feature_size"])
    train_it = train_ds.as_numpy_iterator()
    test_it = test_ds.as_numpy_iterator()
    num_negatives = _NUM_NEGATIVES.value
    x = next(train_it)
    key = sample_negative(x, key, num_negatives, num_tracks, num_albums, num_artists)
    print("Sample input with negatives")
    print(x)
    key, subkey = jax.random.split(key)
    params = spotify.init(
        subkey,
        x["track_context"], x["album_context"], x["artist_context"],
        x["next_track"], x["next_album"], x["next_artist"],
        x["neg_track"], x["neg_album"], x["neg_artist"])
    print("Sample model call")
    result = spotify.apply(
        params,
        x["track_context"], x["album_context"], x["artist_context"],
        x["next_track"], x["next_album"], x["next_artist"],
        x["neg_track"], x["neg_album"], x["neg_artist"])
    print(result)

    tx = optax.sgd(learning_rate=config["learning_rate"])
    state = train_state.TrainState.create(
        apply_fn=spotify.apply, params=params, tx=tx)
    if _RESTORE_CHECKPOINT.value:
        state = checkpoints.restore_checkpoint(_WORKDIR.value, state)

    train_step_fn = jax.jit(train_step)
    eval_step_fn = jax.jit(eval_step)

    losses = []
    init_step = state.step
    logging.info("Starting at step %d", init_step)
    regularization = config["regularization"]
    batch_size = _BATCH_SIZE.value
    eval_steps = _EVAL_STEPS.value
    for i in range(init_step, _MAX_STEPS.value + 1):
        x = next(train_it)
        key = sample_negative(x, key, num_negatives, num_tracks, num_albums, num_artists)        
        state, loss = train_step_fn(
            state, x, regularization)

        losses.append(loss)        
        if i % _CHECKPOINT_EVERY_STEPS.value == 0 and i > 0:
            logging.info("Saving checkpoint")
            checkpoints.save_checkpoint(_WORKDIR.value, state, state.step, keep=3)
        metrics = {
            "step" : state.step
        }
        #if i % _EVAL_EVERY_STEPS.value == 0 and i > 0:
        #    eval_loss = []
        #    for j in range(eval_steps):
        #        ebatch = next(test_it)
        #        loss = eval_step_fn(state, )
        #        eval_loss.append(loss)
        #    eval_loss = jnp.mean(jnp.array(eval_loss)) / batch_size
        #    metrics.update({"eval_loss" : eval_loss})
        if i % _LOG_EVERY_STEPS.value == 0 and i > 0:
            mean_loss = jnp.mean(jnp.array(losses))
            losses = []
            metrics.update({"train_loss" : mean_loss})
            #wandb.log(metrics)
            logging.info(metrics)

    #logging.info("Saving as %s", _MODEL_NAME.value)
    #data = flax.serialization.to_bytes(state)
    #metadata = { "output_size" : wandb.config.output_size }
    #artifact = wandb.Artifact(
        #name=_MODEL_NAME.value,
        #metadata=metadata,
        #type="model")
    #with artifact.new_file("pinterest_stl.model", "wb") as f:
        #f.write(data)
    #run.log_artifact(artifact)


if __name__ == "__main__":
    app.run(main)
