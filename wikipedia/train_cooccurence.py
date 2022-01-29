#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Trains the co-occurrence matrix.
  See the GloVe paper for the math.
  https://nlp.stanford.edu/pubs/glove.pdf
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from absl import app
from absl import flags
from token_dictionary import TokenDictionary
from cooccurrence_matrix import CooccurrenceGenerator
import debug_callbacks

FLAGS = flags.FLAGS
flags.DEFINE_string("train_input_pattern", None, "Input cooccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string("validation_input_pattern", None, "Input cooccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_integer("max_terms", 20, "Max terms per row to dump")
flags.DEFINE_integer("embedding_dim", 64,
                     "Embedding dimension.")
flags.DEFINE_integer("batch_size", 1024,
                     "Batch size")
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


def make_callbacks(token_dictionary):
    """Makes the model hook callbacks."""
    callbacks = []
    if FLAGS.tensorboard_dir is not None:
        callback = TensorBoard(log_dir=FLAGS.tensorboard_dir,
                               histogram_freq=100,
                               batch_size=FLAGS.batch_size,
                               write_graph=True,
                               update_freq='epoch')
        callbacks.append(callback)
    if FLAGS.terms is not None:
        callback = debug_callbacks.WordNNCallback(FLAGS.terms, FLAGS.max_terms, token_dictionary)
        callbacks.append(callback)
    if FLAGS.checkpoint_dir is not None:
        checkpointer = ModelCheckpoint(filepath=FLAGS.checkpoint_dir,
                                       verbose=1, mode='min')
        callbacks.append(checkpointer)
    if FLAGS.learning_rate_decay < 1.0:
        def schedule(epoch, lr):
            return lr * FLAGS.learning_rate_decay
        decay = LearningRateScheduler(schedule, verbose=1)
        callbacks.append(decay)
    if len(callbacks) == 0:
        return None
    return callbacks


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
    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    num_tokens = token_dictionary.get_embedding_dictionary_size()

    # First token
    word_embedding = Embedding(output_dim=FLAGS.embedding_dim,
                               input_dim=num_tokens,
                               input_length=1,
                               embeddings_initializer='he_normal',
                               name='word_embedding')

    token1 = Input(shape=(1,), dtype='int32', name='token')
    token2 = Input(shape=(1,), dtype='int32', name='other_token')

    bias = Embedding(output_dim=1,
                     input_dim=num_tokens,
                     input_length=1,
                     embeddings_initializer='zeros')
    embed1 = word_embedding(token1)
    bias1 = bias(token1)
    embed2 = word_embedding(token2)
    bias2 = bias(token2)
    dot = Dot(axes=2)([embed1, embed2])
    output = Add()([dot, bias1, bias2])
    output = Flatten()(output)

    model = Model(inputs=[token1, token2], outputs=output)
    optimizer = Adam(lr=FLAGS.learning_rate, epsilon=1e-6)
    model.compile(optimizer=optimizer, loss=glove_loss)

    train_data = CooccurrenceGenerator(FLAGS.train_input_pattern)
    validation_data = CooccurrenceGenerator(FLAGS.validation_input_pattern)

    train_iterator = train_data.get_batch(FLAGS.batch_size, FLAGS.shuffle_buffer_size)
    validation_iterator = validation_data.get_batch(FLAGS.batch_size, FLAGS.shuffle_buffer_size)

    callbacks = make_callbacks(token_dictionary)

    model.fit_generator(train_iterator,
                        steps_per_epoch=FLAGS.steps_per_epoch,
                        epochs=FLAGS.num_epochs,
                        callbacks=callbacks,
                        validation_data=validation_iterator,
                        validation_steps=FLAGS.validation_steps)


if __name__ == "__main__":
    app.run(main)
