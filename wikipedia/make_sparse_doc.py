#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Makes sparse documents from tokenized documents.
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
flags.DEFINE_string("input_file", None, "Input doc.pb.b64.bz2 file.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_string("title_dictionary", None, "The title dictionary file.")

flags.DEFINE_string("output_txt2url", None, "The output file for the text to url")
flags.DEFINE_string("output_url2url", None, "The output file for the url to url")


# Required flag.
flags.mark_flag_as_required("input_file")


def text_to_txt2url(doc, token_dictionary_bc, title_dictionary_bc):
    """Creates a txt2url sparse document from a document."""
    token_dict = token_dictionary_bc.value
    title_dict = title_dictionary_bc.value

    primary_idx = title_dict.get_token_index(doc.primary)
    if primary_idx is None:
        return None
    if len(doc.tokens) == 0:
        return None
    proto = nlp_pb.SparseDocument()
    proto.primary_index = primary_idx

    token_idx = token_dict.get_embedding_indices(doc.tokens)
    proto.token_index.extend(token_idx)
    serialized = proto.SerializeToString()
    return base64.b64encode(serialized)


def text_to_url2url(doc, title_dictionary_bc):
    """Creates a url2url sparse document from a document."""
    title_dict = title_dictionary_bc.value

    primary_idx = title_dict.get_token_index(doc.primary)
    if primary_idx is None:
        return None
    proto = nlp_pb.SparseDocument()
    proto.primary_index = primary_idx
    for title in doc.secondary:
        idx = title_dict.get_token_index(title)
        if idx is not None:
            proto.secondary_index.append(idx)
    if len(proto.secondary_index) is 0:
        return None

    serialized = proto.SerializeToString()
    return base64.b64encode(serialized)


def main(argv):
    """Main function."""
    del argv  # Unused.
    sc = SparkContext()

    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    title_dictionary = TokenDictionary(FLAGS.title_dictionary)

    token_dictionary_bc = sc.broadcast(token_dictionary)
    title_dictionary_bc = sc.broadcast(title_dictionary)

    input_rdd = sc.textFile(FLAGS.input_file)
    text_doc = ioutil.parse_document(input_rdd)

    # Generate the url2url documents.
    text_doc \
        .map(lambda x: text_to_url2url(x, title_dictionary_bc)) \
        .filter(lambda x: x is not None) \
        .coalesce(10) \
        .saveAsTextFile(FLAGS.output_url2url,
                        compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")

    # Generate the txt2url documents.
    text_doc\
        .map(lambda x : text_to_txt2url(x, token_dictionary_bc, title_dictionary_bc))\
        .filter(lambda x: x is not None) \
        .coalesce(10) \
        .saveAsTextFile(FLAGS.output_txt2url,
                        compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")


if __name__ == "__main__":
    app.run(main)
