#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Makes dice correlation matrixs.
"""
import base64
import bz2
import nlp_pb2 as nlp_pb
from absl import app
from absl import flags
from token_dictionary import TokenDictionary
from pyspark import SparkContext
import ioutil
import random

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input sdoc.pb.b64.bz2 file.")
flags.DEFINE_string("output_file", None, "Input cooccur.pb.b64.bz2 file.")
flags.DEFINE_integer("max_row_size", 1000, "Max number of items per row.")

# Required flag.
flags.mark_flag_as_required("input_file")


def increment(key, matrix):
    row_idx, col_idx = key
    if row_idx not in matrix:
        row = {}
        matrix[row_idx] = row
    else:
        row = matrix[row_idx]
    if col_idx not in row:
        row[col_idx] = 1
    else:
        row[col_idx] += 1


def process_sdoc(sdoc, cooccur):
    """Counts co-occurrences of tokens."""
    all_indices = [sdoc.primary_index]
    all_indices.extend(sdoc.secondary_index)
    all_indices_set = sorted(list(set(all_indices)))
    count = len(all_indices_set)
    for i in range(count):
        idx1 = all_indices_set[i]
        # Because the counts are symmetric, just count when idx1 < idx2
        for j in range(i + 1, count):
            idx2 = all_indices_set[j]
            assert(idx1 < idx2)
            key = (idx1, idx2)
            increment(key, cooccur)


def cooccur_add(x, y):
    """Adds two dictionaries."""
    for key in y:
        if key not in x:
            x[key] = y[key]
        else:
            x[key] += y[key]
    return x


def process_sdoc_rdd(sc, sdoc, output_file, max_row_size):

    def process_sdoc_partition(sdoc_iterator):
        cooccur = {}
        count = 0
        for doc in sdoc_iterator:
            process_sdoc(doc, cooccur)
            count += 1
            if count % 100 is 0:
                for item in cooccur.items():
                    yield item
                cooccur = {}
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
    cooccur_rdd = sdoc.mapPartitions(process_sdoc_partition)\
        .reduceByKey(cooccur_add)\
        .flatMap(cooccur_to_proto)
    cooccur_rdd.map(lambda x : x.SerializeToString()).map(base64.b64encode)\
        .saveAsTextFile(output_file,
                        compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")


def main(argv):
    """Main function."""
    del argv  # Unused.
    sc = SparkContext()

    input_rdd = sc.textFile(FLAGS.input_file)
    sdoc = ioutil.parse_proto(input_rdd, nlp_pb.SparseDocument)

    process_sdoc_rdd(sc, sdoc, FLAGS.output_file, FLAGS.max_row_size)


if __name__ == "__main__":
    app.run(main)
