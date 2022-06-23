#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Makes co-occurrence matrix.
"""
import base64
import bz2
import random

from absl import app
from absl import flags
from token_dictionary import TokenDictionary
from pyspark import SparkContext

import nlp_pb2 as nlp_pb
import ioutil


FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input doc.pb.b64.bz2 file.")
flags.DEFINE_string("output_file", None, "Input cooccur.pb.b64.bz2 file.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_integer("context_window", 10, "Size of the context window.")
flags.DEFINE_integer("max_row_size", 1000, "Max number of items per row.")

# Required flag.
flags.mark_flag_as_required("input_file")


def process_doc(doc, token_dictionary, context_window, cooccur):
    """Counts co-occurrences of tokens."""
    # Flatten the document out into embedding indices.
    token_idx = token_dictionary.get_embedding_indices(doc.tokens)
    num_words = len(token_idx)
    for i in range(num_words):
        start = max(0, i - context_window)
        end = min(num_words, i + context_window)
        my_idx = token_idx[i]
        if my_idx not in cooccur:
            cooccur[my_idx] = {}
        row = cooccur[my_idx]
        for j in range(start, end):
            other_idx = token_idx[j]
            # Don't count the same word and co-ooccurence is symmetric so only keep when j > i
            if my_idx <= other_idx:
                continue
            dist = abs(i - j)
            increment = 1.0 / float(dist)
            if other_idx in row:
                row[other_idx] += increment
            else:
                row[other_idx] = increment


def cooccur_add(x, y):
    """Adds two dictionaries."""
    for key in y:
        if key not in x:
            x[key] = y[key]
        else:
            x[key] += y[key]
    return x


def process_text_rdd(sc, text_doc, output_file, token_dictionary, context_window, max_row_size):
    # Send the dictionary to all workers.
    token_dictionary_bc = sc.broadcast(token_dictionary)

    def process_text_partition(text_doc_iterator):
        my_td = token_dictionary_bc.value
        cooccur = {}
        for doc in text_doc_iterator:
            process_doc(doc, my_td, context_window, cooccur)
        for item in cooccur.items():
            yield item

    def cooccur_to_proto(cooccur):
        """Converts a co-occurrence to a proto."""
        proto = nlp_pb.CooccurrenceRow()
        proto.index = cooccur[0]
        for kv in cooccur[1].items():
            proto.other_index.append(kv[0])
            proto.count.append(kv[1])
            if len(proto.count) > max_row_size:
                yield proto
                proto = nlp_pb.CooccurrenceRow()
                proto.index = cooccur[0]
        if len(proto.count) > 0:
            yield proto

    # This rdd goes from token i to token j : count
    cooccur_rdd = text_doc.mapPartitions(process_text_partition)\
        .reduceByKey(cooccur_add)\
        .flatMap(cooccur_to_proto)
    cooccur_rdd.map(lambda x : x.SerializeToString()).map(base64.b64encode)\
        .saveAsTextFile(output_file,
                        compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")


def main(argv):
    """Main function."""
    del argv  # Unused.
    sc = SparkContext()

    token_dictionary = TokenDictionary(FLAGS.token_dictionary)

    input_rdd = sc.textFile(FLAGS.input_file)
    text_doc = ioutil.parse_document(input_rdd)

    process_text_rdd(sc, text_doc,
                     FLAGS.output_file,
                     token_dictionary,
                     FLAGS.context_window,
                     FLAGS.max_row_size)


if __name__ == "__main__":
    app.run(main)
