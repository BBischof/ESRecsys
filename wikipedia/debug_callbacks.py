#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Debugging callbacks for Keras.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


class WordNNCallback(Callback):
    """Nearest neighbor callback."""
    def __init__(self, csv, max_terms, token_dictionary):
        super(Callback, self).__init__()
        tokens = csv.split(',')
        self.tokens = []
        self.indices = []
        self.num_embeddings = token_dictionary.get_embedding_dictionary_size()
        self.max_terms = max_terms
        for token in tokens:
            index = token_dictionary.get_embedding_index(token)
            self.tokens.append(token)
            self.indices.append(index)
        self.indices = np.asarray(self.indices, dtype=np.int32)
        self.indices_as_tensors = tf.convert_to_tensor(self.indices)
        self.token_dictionary = token_dictionary

    def on_epoch_end(self, epoch, logs=None):
        embedding_model = Model(inputs=self.model.get_layer("token").input,
                                outputs=self.model.get_layer("word_embedding").output)
        all_indices = np.array(np.arange(self.num_embeddings), dtype=np.int32)
        all_indices = tf.convert_to_tensor(all_indices)
        embeddings = embedding_model(all_indices)
        target_embeddings = embedding_model(self.indices_as_tensors)
        # Find distances from all target embeddings to all other embeddings.
        results = K.dot(target_embeddings, K.transpose(embeddings))
        results = K.get_session().run(results)
        count = min(self.max_terms, self.num_embeddings)
        for i in range(len(self.tokens)):
            far_to_near_indices = np.argsort(results[i])
            result_list = []
            for j in range(count):
                idx = far_to_near_indices[self.num_embeddings - 1 - j]
                sim = results[i][idx]
                other_token = self.token_dictionary.get_token_from_embedding_index(idx)
                display = '%s:%3f' % (other_token, sim)
                result_list.append(display)
            print('Nearest to %s: %s' % (self.tokens[i], ','.join(result_list)))


class SentenceNNCallback(Callback):
    """Sentence nearest neighbor callback."""
    def __init__(self, csv, max_sentence_length, max_terms, token_dictionary, title_dictionary):
        super(Callback, self).__init__()
        self.sentences = csv.split(',')
        self.tokens = []
        self.indices = []
        self.num_embeddings = title_dictionary.get_dictionary_size()
        self.max_terms = max_terms
        indices = []
        for sentence in self.sentences:
            tokens = token_dictionary.simple_tokenize(sentence)
            idx = token_dictionary.get_embedding_indices(tokens)
            if len(idx) > max_sentence_length:
                idx = idx[:max_sentence_length]
            else:
                while len(idx) < max_sentence_length:
                    idx.append(0)
            indices.append(idx)

        self.indices = np.asarray(indices, dtype=np.int32)
        self.indices_as_tensors = tf.convert_to_tensor(self.indices)
        self.title_dictionary = title_dictionary

    def on_epoch_end(self, epoch, logs=None):
        sentence_model = Model(inputs=self.model.get_layer("sentence_input").input,
                               outputs=self.model.get_layer("sentence_to_url").output)
        url_model = Model(inputs=self.model.get_layer("url_near_text").input,
                          outputs=self.model.get_layer("url_embedding").output)
        all_indices = np.array(np.arange(self.num_embeddings), dtype=np.int32)
        all_indices = tf.convert_to_tensor(all_indices)
        url_embeddings = url_model(all_indices)
        target_embeddings = sentence_model(self.indices_as_tensors)
        distances = -K.dot(target_embeddings, K.transpose(url_embeddings))
        distances = K.get_session().run(distances)
        count = min(self.max_terms, self.num_embeddings)
        for i in range(len(self.sentences)):
            print('Nearest to %s' % self.sentences[i])
            # Find distances from all target embeddings to all other embeddings.
            near_to_far_indices = np.argsort(distances[i])
            result_list = []
            for j in range(count):
                idx = near_to_far_indices[j]
                sim = -distances[i][idx]
                other_token = self.title_dictionary.get_token(idx)
                display = '%s:%3f' % (other_token, sim)
                result_list.append(display)
            print('Nearest to %s: %s' % (self.sentences[i], ','.join(result_list)))
