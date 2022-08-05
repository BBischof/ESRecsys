#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Trains the co-occurrence matrix.
  See the GloVe paper for the math.
  https://nlp.stanford.edu/pubs/glove.pdf
"""

from ast import Attribute
import os

from absl import app
from absl import flags
from absl import logging
import flax
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import wandb

from cooccurrence_matrix import CooccurrenceGenerator
from models import Glove
from token_dictionary import TokenDictionary


FLAGS = flags.FLAGS
flags.DEFINE_string("train_input_pattern", None, "Input cooccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_integer("max_terms", 20, "Max terms per row to dump")
flags.DEFINE_integer("embedding_dim", 64,
                     "Embedding dimension.")
flags.DEFINE_integer("batch_size", 256,
                     "Batch size")
flags.DEFINE_integer("seed", 1701,
                     "Random number seed.")
flags.DEFINE_integer("shuffle_buffer_size", 5000000,
                     "Shuffle buffer size")
flags.DEFINE_string("terms", None, "CSV of terms to dump")
flags.DEFINE_string("checkpoint_dir", None, "Location to save checkpoints.")
flags.DEFINE_integer("checkpoint_every_epochs", 10, "Number of epochs to checkpoint.")
flags.DEFINE_string("resume_checkpoint", None, "If not None, resume from this checkpoint.")
flags.DEFINE_integer("steps_per_epoch", 100,
                     "Number of training steps per epoch")
flags.DEFINE_integer("num_epochs", 100,
                     "Number of epochs")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")

# Required flag.
flags.mark_flag_as_required("train_input_pattern")


@jax.jit
def apply_model(state, inputs, target):
    """Computes the gradients and loss for a single batch."""
    
    # Define glove loss.
    def glove_loss(params):
        """The GloVe weighted loss."""
        predicted = state.apply_fn({'params': params}, inputs)
        ones = jnp.ones_like(target)
        weight = jnp.minimum(ones, target / 100.0)
        weight = jnp.power(weight, 0.75)
        log_target = jnp.log10(1.0 + target)
        loss = jnp.mean(jnp.square(log_target - predicted) * weight)
        return loss

    grad_fn = jax.value_and_grad(glove_loss)
    loss, grads = grad_fn(state.params)
    
    return grads, loss

def find_knn(model, params, token):
    scores = model.apply(
        {'params' : params},
        token,
        method=Glove.score_all)
    indices = jnp.argsort(scores, axis=0)
    return scores, indices

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

def train_epoch(state, steps_per_epoch, train_it):
    """Trains for an epoch."""
    epoch_loss = []
    for i in range(steps_per_epoch):
        inputs, targets = next(train_it)
        grads, loss = apply_model(state, inputs, targets)
        state = update_model(state, grads)
        epoch_loss.append(loss)
    train_loss = np.mean(epoch_loss)
    return state, train_loss

def dump_knn(model, params, tokens, token_dictionary):
    """"Dumps the k-nearest neighbors of tokens."""
    scores, indices = find_knn(model, params, tokens)
    for i in range(tokens.shape[0]):
        token = tokens[i]
        query_word = token_dictionary.get_token_from_embedding_index(token)        
        knn = []
        for j in range(10):
            idx = indices[-j-1][i]
            word = token_dictionary.get_token_from_embedding_index(idx)
            score = scores[idx][i]
            knn.append("%s:%f" % (word, score))  
        logging.info("Nearest neighbors for %s: %s", query_word, " ".join(knn))


def save_state(state, step):
    """Saves the state of the model."""
    filename = os.path.join(FLAGS.checkpoint_dir, "checkpoint-%05d.flax" % step)
    with open(filename, "wb") as f:
        serialized = flax.serialization.to_bytes(state)
        f.write(serialized)


def main(argv):
    """Main function."""
    del argv  # Unused.
    init_config = dict(
        seed = FLAGS.seed,
        learning_rate = FLAGS.learning_rate)
    run = wandb.init(config=init_config)
    config = wandb.config

    # We are only using tensorflow for tf.data so disable GPU use.
    tf.config.set_visible_devices([], 'GPU')
    logging.info('JAX process: %d / %d',
                 jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())
  
    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    num_tokens = token_dictionary.get_embedding_dictionary_size()

    debug_tokens = []
    for word in FLAGS.terms.split(','):
        token = token_dictionary.get_embedding_index(word)
        debug_tokens.append(token)
    debug_tokens = jnp.array(debug_tokens, dtype=jnp.int32)

    model = Glove(num_embeddings=num_tokens, features=FLAGS.embedding_dim)

    train_data = CooccurrenceGenerator(FLAGS.train_input_pattern)

    train_iterator = train_data.get_dataset(FLAGS.batch_size, FLAGS.shuffle_buffer_size)
    train_iterator = train_iterator.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    key = jax.random.PRNGKey(config.seed)
    x, _ = next(train_iterator)
    params = model.init(key, x)
    out = model.apply(params, x)
    tx = optax.adagrad(config.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params["params"], tx=tx)
    if FLAGS.resume_checkpoint:
        logging.info("Resuming from %s", FLAGS.resume_checkpoint)
        with open(FLAGS.resume_checkpoint, "rb") as f:
            contents = f.read()
            flax.serialization.from_bytes(state, contents)

    for step in range(FLAGS.num_epochs):
        logging.info("Step %d", step)
        if step % FLAGS.checkpoint_every_epochs == 0:
            save_state(state, step)
        dump_knn(model, state.params, debug_tokens, token_dictionary)
        state, train_loss = train_epoch(state, FLAGS.steps_per_epoch, train_iterator)
        logging.info("Training loss %f", train_loss)
        wandb.log({"train_loss" : train_loss})

    art = wandb.Artifact(f"glove-wikipedia-{wandb.run.id}", type="model")
    with art.new_file("glove.flax", "wb") as f:
        serialized = flax.serialization.to_bytes(state)
        f.write(serialized)
    wandb.log_artifact(art)

if __name__ == "__main__":
    app.run(main)
