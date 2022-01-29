#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Dumps the dice co-occurrence matrix.
"""
import base64
import bz2
import numpy as np
import nlp_pb2 as nlp_pb
import ioutil
from absl import app
from absl import flags
from token_dictionary import TokenDictionary

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input cooccur.pb.b64.bz2 file.")
flags.DEFINE_string("title_dictionary", None, "The titlte dictionary file.")
flags.DEFINE_integer("max_terms", 20, "Max terms per row to dump")
flags.DEFINE_integer("max_rows", 20, "Max rows to dump")

# Required flag.
flags.mark_flag_as_required("input_file")


def main(argv):
    """Main function."""
    del argv  # Unused.
    title_dictionary = TokenDictionary(FLAGS.title_dictionary)
    generator = ioutil.proto_generator(FLAGS.input_file, nlp_pb.CooccurrenceRow)
    for i in range(FLAGS.max_rows):
        row = next(generator)
        title = title_dictionary.get_token(row.index)
        main_count = title_dictionary.get_doc_frequency(row.index)
        print("Nearest to %s" % title)
        # Compute the dice correlation coefficient.
        candidates = []
        for j in range(len(row.other_index)):
            idx = row.other_index[j]
            title = title_dictionary.get_token(idx)
            joint_count = row.count[j]
            doc_count = title_dictionary.get_doc_frequency(idx)
            dice = joint_count / (doc_count + main_count)
            candidates.append((dice, joint_count, title))
        candidates = sorted(candidates)
        candidates = list(reversed(candidates))
        count = min(FLAGS.max_rows, len(candidates))
        for j in range(count):
            print("\tScore %f joint count %d url %s" % candidates[j])


if __name__ == "__main__":
    app.run(main)
