#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Trains the co-occurrence matrix.
  See the GloVe paper for the math.
  https://nlp.stanford.edu/pubs/glove.pdf
"""

from flax import linen as nn

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf

from token_dictionary import TokenDictionary
from cooccurrence_matrix import CooccurrenceGenerator


FLAGS = flags.FLAGS
flags.DEFINE_string("train_input_pattern", None, "Input cooccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string("validation_input_pattern", None, "Input cooccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_integer("max_terms", 20, "Max terms per row to dump")
flags.DEFINE_integer("embedding_dim", 64,
                     "Embedding dimension.")
flags.DEFINE_integer("batch_size", 1024,
                     "Batch size")
flags.DEFINE_integer("seed", 1701,
                     "Random number seed.")
flags.DEFINE_integer("shuffle_buffer_size", 5000000,
                     "Shuffle buffer size")
flags.DEFINE_string("terms", None, "CSV of terms to dump")
flags.DEFINE_string("tensorboard_dir", None, "Location to store training logs.")
flags.DEFINE_string("checkpoint_dir", None, "Location to save checkpoints.")
flags.DEFINE_integer("steps_per_epoch", 1000,
                     "Number of steps per epoch")
flags.DEFINE_integer("num_epochs", 100,
                     "Number of epochs")
flags.DEFINE_integer("validation_steps", 100,
                     "Number of validation steps")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_float("learning_rate_decay", 0.9, "Learning rate decay")

# Required flag.
flags.mark_flag_as_required("train_input_pattern")
flags.mark_flag_as_required("validation_input_pattern")

class Glove(nn.Module):
    """A simple embedding model based on gloVe.
       https://nlp.stanford.edu/projects/glove/
    """
    num_embeddings: int = 1024
    features: int = 64
    
    def setup(self):
        self._token_embedding = nn.Embed(self.num_embeddings,
                                         self.features)
        self._bias = nn.Embed(self.num_embeddings, 1)

    def __call__(self, inputs):
        """Calculates the approximate log count between tokens 1 and 2.

        Args:
          A batch of (token1, token2) integers representing co-occurence.

        Returns:
          Approximate log count between x and y.
        """
        token1, token2 = inputs
        embed1 = self._token_embedding(token1)
        bias1 = self._bias(token1)
        embed2 = self._token_embedding(token2)
        bias2 = self._bias(token2)
        dot = jnp.sum(embed1 * embed2, axis=1)
        output = dot + bias1 + bias2
        return output

# Define glove loss.
def glove_loss(y_true, y_pred):
    """The GloVe weighted loss."""
    weight = K.minimum(1.0, y_true / 100.0)
    weight = K.pow(weight, 0.75)
    # Approximate log base 10.
    log_true = K.log(1.0 + y_true) / 2.3
    return K.mean(K.square(log_true - y_pred) * weight, axis=-1)


def main(argv):
    """Main function."""
    del argv  # Unused.

    # Disable GPU use for tensorflow as we are just using the tf.data part
    tf.config.experimental.set_visible_devices([], "GPU")
    logging.info('JAX process: %d / %d',
                 jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())
  
    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    num_tokens = token_dictionary.get_embedding_dictionary_size()

    model = Glove(num_embeddings=num_tokens, features=FLAGS.embedding_dim)

    train_data = CooccurrenceGenerator(FLAGS.train_input_pattern)
    validation_data = CooccurrenceGenerator(FLAGS.validation_input_pattern)

    train_iterator = train_data.get_batch(FLAGS.batch_size, FLAGS.shuffle_buffer_size)
    validation_iterator = validation_data.get_batch(FLAGS.batch_size, FLAGS.shuffle_buffer_size)

    key = jax.random.PRNGKey(FLAGS.seed)
    x, _ = next(train_iterator)
    params = model.init(key, x)
    out = model.apply(params, x)
    print(out)


if __name__ == "__main__":
    app.run(main)
