#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  This reads a wikipedia.pb.b64.bz2 file and tokenizes it using pyspark.
"""
import base64
import nlp_pb2 as nlp_pb
import wikipedia_pb2 as wikipedia_pb
from absl import app
from absl import flags
from token_dictionary import TokenDictionary
from pyspark import SparkContext
from url_normalize import url_normalize
import re

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input wiki.pb.b64.bz2 file.")
flags.DEFINE_string("output_file", None,
                    "The tokenized doc.pb.b64.bz2 file.")

# Required flag.
flags.mark_flag_as_required("input_file")
flags.mark_flag_as_required("output_file")


def convert_wiki_to_doc(page_iterator):
    """Processes a partition of pages."""
    # For efficiency we compile this outside.
    # This matches anything that has double square brackets [[ foo ]].
    title_re = re.compile('\[\[[^\]]*\]\]')
    td = TokenDictionary()

    # Some wikipedia groups are not for user consumption.
    title_reject_re = re.compile("^Wikipedia:|^User:|^File:|^MediaWiki:|^Template:|^Help:|^Portal:|^Draft:")

    def should_tokenize(page):
        """Determines if we should tokenize the page or not."""
        if page.redirect_title:
            return False
        if not page.title:
            return False
        if not page.revision:
            return False
        if title_reject_re.match(page.title):
            return False
        return True

    def make_url_from_title(title):
        url = "https://en.wikipedia.org/wiki/%s" % title.replace(' ', '_')
        return url_normalize(url)

    def tokenize_page(page, td):
        """Converts the text into a list of tokens"""
        body_text = page.revision[0].text
        # El cheapo cleaning
        # Get rid of the italics and bold markups
        tokens = td.simple_tokenize(body_text)
        return tokens

    def extract_title_cooccurrence(page):
        """Retrieves title to title co-occurrences."""
        titles = []
        all_blocks = title_re.findall(page.revision[0].text)
        for block in all_blocks:
            block = block.replace('[', '').replace(']', '')
            # The subsequent stuff after the | is just display text.
            subtitles = block.split('|')
            title = subtitles[0]
            if not title_reject_re.match(title):
              titles.append(title)
        return set(titles)

    for page in page_iterator:
        # Extract out all the title to title similarities.
        if not should_tokenize(page):
            continue
        titles = extract_title_cooccurrence(page)
        doc = nlp_pb.TextDocument()
        doc.primary = make_url_from_title(page.title)
        titles = [make_url_from_title(x) for x in titles]
        doc.secondary.extend(titles)
        tokens = tokenize_page(page, td)
        doc.tokens.extend(tokens)
        yield doc


def parse_wikipedia(rdd):
    """Changes a text rdd into a wikipedia proto rdd."""
    def parser(x):
        result = wikipedia_pb.Page()
        try:
            result.ParseFromString(x)
        except google.protobuf.message.DecodeError:
            result = None
        return result
    output = rdd.map(base64.b64decode)\
        .map(parser)\
        .filter(lambda x: x is not None)
    return output


def process_wikipedia(wikiproto):
    doc = wikiproto.mapPartitions(convert_wiki_to_doc)
    doc\
        .map(lambda x : x.SerializeToString())\
        .map(lambda x : base64.b64encode(x))\
        .saveAsTextFile(FLAGS.output_file,
                        compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")


def main(argv):
    """Main function."""
    del argv  # Unused.
    sc = SparkContext()
    input_rdd = sc.textFile(FLAGS.input_file)
    wikiproto = parse_wikipedia(input_rdd)
    process_wikipedia(wikiproto)


if __name__ == "__main__":
    app.run(main)
