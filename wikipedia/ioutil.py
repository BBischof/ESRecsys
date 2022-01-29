#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import nlp_pb2 as nlp_pb
import base64
import tensorflow as tf
import bz2
import numpy as np

"""
  I/O utilities.
"""


def load_stopwords(input_file):
    """Loads a stopwords file"""
    stopwords = []
    with open(input_file, 'r') as file:
        for line in file:
            stopwords.append(line[:-1])
    print('%d stopwords loaded' % len(stopwords))
    return set(stopwords)


def parse_proto(rdd, proto):
    def parser(x):
        result = proto()
        try:
            result.ParseFromString(x)
        except google.protobuf.message.DecodeError:
            result = None
        return result
    output = rdd.map(base64.b64decode)\
        .map(parser)\
        .filter(lambda x: x is not None)
    return output


def parse_document(rdd):
    return parse_proto(rdd, nlp_pb.TextDocument)


def proto_generator(input_pattern, proto_func):
    """Generates protocol buffers from pb.b64.bz2 files and their proto function."""
    input_files = tf.gfile.Glob(input_pattern)
    total_files = len(input_files)
    while True:
        np.random.shuffle(input_files)
        file_epoch = 0
        for input_file in input_files:
            file_epoch += 1
            print('Opening %s (%d of %d)' % (input_file, file_epoch, total_files))
            with bz2.open(input_file, 'rb') as file:
                for line in file:
                    # Drop the trailing \n
                    line = line[:-1]
                    serialized = base64.b64decode(line)
                    proto = proto_func()
                    proto.ParseFromString(serialized)
                    yield proto


def shuffle_generator(other_generator, shuffle_size):
    """Takes another generator and shuffles it."""
    while True:
        buffer = [next(other_generator) for _ in range(shuffle_size)]
        np.random.shuffle(buffer)
        for item in buffer:
            yield item
