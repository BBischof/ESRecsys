#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Dumps the co-occurrence matrix.
"""
import base64
import bz2
import numpy as np
import nlp_pb2 as nlp_pb
from absl import app
from absl import flags
from token_dictionary import TokenDictionary
from cooccurrence_matrix import CooccurrenceMatrix

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input cooccur.pb.b64.bz2 file.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_integer("max_terms", 20, "Max terms per row to dump")
flags.DEFINE_integer("max_rows", 20, "Max rows to dump")

# Required flag.
flags.mark_flag_as_required("input_file")


def main(argv):
    """Main function."""
    del argv  # Unused.
    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    matrix = CooccurrenceMatrix(FLAGS.input_file)
    matrix.debug_print(FLAGS.max_rows, token_dictionary, FLAGS.max_terms)

if __name__ == "__main__":
    app.run(main)
