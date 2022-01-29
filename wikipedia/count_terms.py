#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Converts text documents to indices and also makes co-occurrence matrix.
"""
import base64
import bz2
import numpy as np
import nlp_pb2 as nlp_pb
from absl import app
from absl import flags
from token_dictionary import TokenDictionary
import ioutil

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input doc.pb.b64.bz2 file.")
flags.DEFINE_string("output_file", None, "Input sparse.pb.b64.bz2 file.")
flags.DEFINE_string("stopwords_file", None,
                    "If specified a text file containing stopwords")
flags.DEFINE_string("title_dictionary", None,
                     "The title dictionary file.")
flags.DEFINE_string("token_dictionary", None,
                     "The token dictionary file.")

# Required flag.
flags.mark_flag_as_required("input_file")


def make_sparse_doc(doc, log_max_num_docs, title_dictionary, token_dictionary, stopwords):
    """Makes a sparse document from the document."""
    sparse_doc = nlp_pb.SparseDocument()
    sparse_doc.url = doc.url
    # Convert all titles and tokens to indices in the dictionaries.
    primary_index = title_dictionary.get_token_index(doc.primary)
    if primary_index:
        sparse_doc.primary_index = primary_index
    for title in doc.secondary:
        title_index = title_dictionary.get_token_index(title)
        if title_index is not None:
            sparse_doc.secondary_index.append(title_index)
    # Get the term frequencies.
    tf = {}
    for token in doc.tokens:
        if token in stopwords:
            continue
        if token in tf:
            tf[token] += 1.0
        else:
            tf[token] = 1.0
    norm = 0.0
    for token in tf:
        token_index = token_dictionary.get_token_index(token)
        if token_index is not None:
            sparse_doc.token_index.append(token_index)
            # Using the scikit learn variant by adding +1.
            df = token_dictionary.get_doc_frequency(token_index)
            idf = log_max_num_docs - np.log1p(df) + 1.0
            if idf < 0.0:
                idf = 0.0
            tfidf = tf[token] * idf
            norm += tfidf * tfidf
            sparse_doc.token_tfidf.append(tfidf)
    # L2 normalize the token tfidf.
    inorm = norm
    if inorm > 0.0:
        inorm = 1.0 / np.sqrt(inorm)
    else:
        inorm = 0.0
    for i in range(len(sparse_doc.token_tfidf)):
        sparse_doc.token_tfidf[i] *= inorm
    return sparse_doc


def process_file(inputfile, title_dictionary, token_dictionary, stopwords):
    max_num_docs = token_dictionary.get_max_doc_frequency()
    log_max_num_docs = np.log1p(max_num_docs)
    print('Number of documents is %d' % max_num_docs)
    count = 0
    with bz2.open(inputfile, 'rb') as file:
        for line in file:
            # Drop the trailing \n
            line = line[:-1]
            serialized = base64.b64decode(line)
            doc = nlp_pb.TextDocument()
            doc.ParseFromString(serialized)
            sparse_doc = make_sparse_doc(doc, log_max_num_docs,
                                         title_dictionary, token_dictionary, stopwords)
            print(sparse_doc)
            count = count + 1
            if count > 10:
                break


def main(argv):
    """Main function."""
    del argv  # Unused.
    if FLAGS.stopwords_file:
        stopwords = ioutil.load_stopwords(FLAGS.stopwords_file)
    else:
        stopwords = set()
    title_dictionary = TokenDictionary(FLAGS.title_dictionary)
    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    process_file(FLAGS.input_file, title_dictionary,
                 token_dictionary, stopwords)


if __name__ == "__main__":
    app.run(main)
