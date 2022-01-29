#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
    Token dictionary, a class that handles conversion of a list of tokens to indices.
"""

import base64
import bz2
import nlp_pb2 as nlp_pb
import re
import binascii


class TokenDictionary:
    def __init__(self, dictionary_file=None):
        self.__token2index = {}
        self.__max_doc_frequency = 0
        self.__token_stat = []
        self.__filter = re.compile('[ !@#$%^&*()_+\t\n",.:;\\/?><|{}\'\[\]]')
        if dictionary_file is not None:
            self.load(dictionary_file)

    @staticmethod
    def save(all_tokens, output_filename):
        with bz2.open(output_filename, 'wb') as ofile:
            for item in all_tokens:
                serialized = base64.b64encode(item.SerializeToString())
                ofile.write(serialized)
                ofile.write(b'\n')

    def simple_tokenize(self, x):
        tokens = self.__filter.split(x)
        tokens = [t.lower() for t in tokens if len(t) > 0]
        return tokens

    @staticmethod
    def minhash(token):
        """Breaks a string up into chunks of overlapping 4 bytes and returns the smallest.."""
        count = len(token)
        if type(token) is str:
            b = bytes(token, 'utf-8')
        else:
            b = token
        minhash = 0xFFFFFFFF
        if (count <= 4):
            minhash = binascii.crc32(b) & 0xFFFF
        else:
            # Don't use more than the first 10 characters.
            count = min(10, count)
            for i in range(count - 4):
                curr = binascii.crc32(b[i:i+4]) & 0xFFFF
                minhash = min(curr, minhash)
        return minhash

    def get_embedding_index(self, token):
        """Computes the embedding index. 0 is special and reserved. Items not in dictionary are minhashed."""
        token_index = self.get_token_index(token)
        if token_index is not None:
            # 0 is reserved for the mask.
            return 1 + token_index
        return 1 + self.get_dictionary_size() + self.minhash(token)

    def get_embedding_dictionary_size(self):
        """Size of embedding space including 0 and the minhash space."""
        return 1 + 65536 + self.get_dictionary_size()

    def get_embedding_indices(self, tokens):
        """Convert a list of strings to tokens."""
        token_idx = []
        for token in tokens:
            token_index = self.get_embedding_index(token)
            token_idx.append(token_index)
        return token_idx

    def load(self, dictionary_file):
        """Loads a dictionary."""
        count = 0
        with bz2.open(dictionary_file, 'rb') as file:
            for line in file:
                # Drop the trailing \n
                line = line[:-1]
                serialized = base64.b64decode(line)
                token_stat = nlp_pb.TokenStat()
                token_stat.ParseFromString(serialized)
                assert(token_stat.index == count)
                self.__token2index[token_stat.token] = token_stat.index
                self.__max_doc_frequency = max(self.__max_doc_frequency, token_stat.doc_frequency)
                self.__token_stat.append(token_stat)
                # Ensure that we can use a direct index scheme to look up words from indices.
                count += 1

    def get_dictionary_size(self):
        return len(self.__token2index)

    def get_max_doc_frequency(self):
        return self.__max_doc_frequency

    def get_doc_frequency(self, token_index):
        return self.__token_stat[token_index].doc_frequency

    def get_token_index(self, token):
        if token not in self.__token2index:
            return None
        return self.__token2index[token]

    def get_token(self, token_index):
        return self.__token_stat[token_index].token

    def get_token_from_embedding_index(self, embedding_index):
        if embedding_index is 0:
            return "NULL"
        elif embedding_index <= self.get_dictionary_size():
            return self.get_token(embedding_index - 1)
        else:
            return "MINHASH %d" % (embedding_index - 65536 - 1)
