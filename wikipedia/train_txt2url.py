#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Trains the text to url and url 2 url.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Subtract
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.constraints import max_norm

from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

from absl import app
from absl import flags
from token_dictionary import TokenDictionary
from ioutil import proto_generator
from ioutil import shuffle_generator
import nlp_pb2 as nlp_pb

import debug_callbacks

FLAGS = flags.FLAGS
flags.DEFINE_string("txt2url_train_input_pattern", None, "Input sdoc.pb.b64.bz2 file pattern.")
flags.DEFINE_string("txt2url_validation_input_pattern", None, "Input sdoc.pb.b64.bz2 file pattern.")
flags.DEFINE_string("url2url_train_input_pattern", None, "Input coccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string("url2url_validation_input_pattern", None, "Input coccur.pb.b64.bz2 file pattern.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_string("title_dictionary", None, "The title dictionary file.")
flags.DEFINE_string("word_embedding", None, "HDF5 model of the word embedding.")
flags.DEFINE_integer("sentence_length", 64,
                     "Max number of words in a sentence.")
flags.DEFINE_integer("max_sentence_per_example", 8,
                     "Max number sentences per example.")
flags.DEFINE_integer("max_terms", 20, "Max terms per row to dump")
flags.DEFINE_integer("word_embedding_dim", 64,
                     "Embedding dimension for the words.")
flags.DEFINE_integer("url_embedding_dim", 8,
                     "Embedding dimension for the urls.")
flags.DEFINE_integer("rnn_size", 128,
                     "RNN cell size.")
flags.DEFINE_integer("batch_size", 32,
                     "Batch size")
flags.DEFINE_integer("shuffle_buffer_size", 1000,
                     "Shuffle buffer size")
flags.DEFINE_string("sentence_csv", None, "CSV of terms to dump")
flags.DEFINE_string("tensorboard_dir", None, "Location to store training logs.")
flags.DEFINE_string("checkpoint_dir", None, "Location to save checkpoints.")
flags.DEFINE_string("loss_type", "MSE", "Type of loss")
flags.DEFINE_integer("steps_per_epoch", 1000,
                     "Number of steps per epoch")
flags.DEFINE_integer("num_epochs", 100,
                     "Number of epochs")
flags.DEFINE_integer("validation_steps", 100,
                     "Number of validation steps")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_float("learning_rate_decay", 0.9, "Learning rate decay")
flags.DEFINE_float("url_max_norm", 10.0, "Max norm for url embedding")
flags.DEFINE_float("text_l2", 0.0, "Text l2 regularization.")
flags.DEFINE_float("margin", 0.1, "Margin for text similarity.")

def make_callbacks(token_dictionary, title_dictionary):
    """Makes the model hook callbacks."""
    callbacks = []
    if FLAGS.tensorboard_dir is not None:
        callback = TensorBoard(log_dir=FLAGS.tensorboard_dir,
                               histogram_freq=100,
                               batch_size=FLAGS.batch_size,
                               write_graph=True,
                               update_freq='epoch')
        callbacks.append(callback)
    if FLAGS.sentence_csv is not None:
        callback = debug_callbacks.SentenceNNCallback(FLAGS.sentence_csv, FLAGS.sentence_length,
                                                      FLAGS.max_terms, token_dictionary, title_dictionary)
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


def url_triplet_generator(input_pattern, title_dictionary):
    """Makes a triplet of (url1, url2, score)"""
    gen = proto_generator(input_pattern, nlp_pb.CooccurrenceRow)
    while True:
        row = next(gen)
        main_count = title_dictionary.get_doc_frequency(row.index)
        # Compute the dice correlation coefficient.
        for j in range(len(row.other_index)):
            idx = row.other_index[j]
            joint_count = row.count[j]
            doc_count = title_dictionary.get_doc_frequency(idx)
            dice = 2.0 * joint_count / (doc_count + main_count)
            yield (row.index, idx, dice)


def txt2url_generator(input_pattern, sentence_length, max_sentence_per_example):
    """Makes a tuple of (url_near, [text tokens])"""
    gen = proto_generator(input_pattern, nlp_pb.SparseDocument)
    while True:
        sdoc = next(gen)
        # Check for short sentences and pad.
        num_tokens_in_page = len(sdoc.token_index)
        length_diff = sentence_length - num_tokens_in_page
        tokens = sdoc.token_index
        if length_diff is 0:
            yield (sdoc.primary_index, tokens)
        elif length_diff > 0:
            # pad till end.
            for i in range(length_diff):
                tokens.append(0)
            yield (sdoc.primary_index, tokens)
        else:
            for i in range(max_sentence_per_example):
                neg_idx = sdoc.primary_index
                # Pick a random sentence fragment.
                idx = np.random.randint(0, num_tokens_in_page - sentence_length)
                yield (sdoc.primary_index, tokens[idx:idx + sentence_length])


def print_url_triplet(tup, title_dictionary):
    a, b, c = tup
    a = title_dictionary.get_token(a)
    b = title_dictionary.get_token(b)
    c = title_dictionary.get_token(c)
    print("%s %s %s" % (a, b, c))


def print_txt2url(tup, title_dictionary, token_dictionary):
    near, text = tup
    print('near %s' % title_dictionary.get_token(near))
    tokens = [token_dictionary.get_token_from_embedding_index(t) for t in text]
    print(' '.join(tokens))


def similarity(a, b, name):
    result = Lambda(lambda x: K.dot(x[0], K.transpose(x[1])), name=name)([a, b])
    return result


def triplet_generator(
        url2url_pattern, txt2url_pattern, batch_size, shuffle_size,
        sentence_length, max_sentence_per_example, max_title_embedding, title_dictionary):
    """Generator that makes (url_near, url_far, text) and (url_near1, url_near2, urlfar) sequences."""
    url_triplet = url_triplet_generator(url2url_pattern, title_dictionary)
    if shuffle_size > 0:
        url_triplet = shuffle_generator(url_triplet, shuffle_size)
    txt2url_triplet = txt2url_generator(txt2url_pattern, sentence_length,
                                        max_sentence_per_example)
    while True:
        sentence_input = []
        url_near_text = []
        url1 = []
        url2 = []
        score = []
        for i in range(batch_size):
            url_near, tokens = next(txt2url_triplet)
            url_near_text.append(url_near)
            sentence_input.append(tokens)

            a, b, c = next(url_triplet)
            url1.append(a)
            url2.append(b)
            score.append(c)

        x = [np.asarray(url_near_text, dtype=np.int32),
             np.asarray(sentence_input, dtype=np.int32),
             np.asarray(url1, dtype=np.int32),
             np.asarray(url2, dtype=np.int32)]
        y = [np.zeros(shape=[batch_size], dtype=np.float32),
             np.sqrt(np.asarray(score, dtype=np.float32))]
        yield (x, y)


def main(argv):
    """Main function."""
    del argv  # Unused.

    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    title_dictionary = TokenDictionary(FLAGS.title_dictionary)
    max_title_embedding = title_dictionary.get_dictionary_size()
    num_word_tokens = token_dictionary.get_embedding_dictionary_size()

    # Word embeddings.
    word_embedding = Embedding(output_dim=FLAGS.word_embedding_dim,
                               input_dim=num_word_tokens,
                               input_length=1,
                               embeddings_initializer='he_normal',
                               mask_zero=True,
                               embeddings_constraint=max_norm(3.0),
                               name='word_embedding')

    sentence_input = Input(shape=(FLAGS.sentence_length,), dtype='int32', name='sentence_input')
    sentence_embedding = word_embedding(sentence_input)

    # The RNN is just used to create weights for the weighted sum.
    sentence_rnn = LSTM(units=FLAGS.rnn_size, recurrent_activation='sigmoid', activation='tanh',
                        name='sentence_rnn')(sentence_embedding)
    # (batch, time, rnn) -> (batch, time, 1)
    regularizer = tf.keras.regularizers.l2(FLAGS.text_l2)
    sentence_to_url = Dense(FLAGS.url_embedding_dim,
                            activity_regularizer=regularizer,
                            name='sentence_to_url')(sentence_rnn)

    url_embedding = Embedding(output_dim=FLAGS.url_embedding_dim,
                              input_dim=max_title_embedding,
                              input_length=1,
                              embeddings_initializer='he_normal',
                              embeddings_constraint=max_norm(FLAGS.url_max_norm),
                              name='url_embedding')
    url_near_text_input = Input(shape=(1,), dtype='int32', name='url_near_text')
    url_near_text = Flatten()(url_embedding(url_near_text_input))

    # Handle the URL triplets.
    url_near1_input = Input(shape=(1,), dtype='int32', name='url_near1_input')
    url_near1 = Flatten()(url_embedding(url_near1_input))
    url_near2_input = Input(shape=(1,), dtype='int32', name='url_near2_input')
    url_near2 = Flatten()(url_embedding(url_near2_input))

    input_list = [url_near_text_input, sentence_input,
                  url_near1_input, url_near2_input]

    url_loss = similarity(url_near1, url_near2, 'url_dice')
    text_sim = similarity(sentence_to_url, url_near_text, 'text2url')
    text_loss = Lambda(lambda x: K.square(K.relu(FLAGS.margin - x)), name='text_sim')(text_sim)

    output_list = [text_loss, url_loss]
    loss_list = ['MAE', FLAGS.loss_type]
    model = Model(inputs=input_list, outputs=output_list)

    if FLAGS.word_embedding:
        model.load_weights(FLAGS.word_embedding, by_name=True)

    optimizer = RMSprop(lr=FLAGS.learning_rate)
    model.compile(optimizer=optimizer, loss=loss_list)
    model.summary()

    train_iterator = triplet_generator(
        FLAGS.url2url_train_input_pattern,
        FLAGS.txt2url_train_input_pattern,
        FLAGS.batch_size,
        FLAGS.shuffle_buffer_size,
        FLAGS.sentence_length,
        FLAGS.max_sentence_per_example,
        max_title_embedding, title_dictionary)

    validation_iterator = triplet_generator(
        FLAGS.url2url_validation_input_pattern,
        FLAGS.txt2url_validation_input_pattern,
        FLAGS.batch_size,
        0,
        FLAGS.sentence_length,
        FLAGS.max_sentence_per_example,
        max_title_embedding, title_dictionary)

    callbacks = make_callbacks(token_dictionary, title_dictionary)

    model.fit_generator(train_iterator,
                        steps_per_epoch=FLAGS.steps_per_epoch,
                        epochs=FLAGS.num_epochs,
                        callbacks=callbacks,
                        validation_data=validation_iterator,
                        validation_steps=FLAGS.validation_steps)


if __name__ == "__main__":
    app.run(main)
