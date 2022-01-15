"""
  This utility tokenizes wikipedia .pb.b64.bz2 files.
"""
import base64
import bz2
import re

import nlp_pb2 as nlp_pb
import nltk
import wikipedia_pb2 as wiki_pb
import ioutil
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input wiki.pb.b64.bz2 file.")
flags.DEFINE_string("output_file", None,
                    "The tokenized doc.pb.b64.bz2 file.")
flags.DEFINE_string("stopwords_file", None,
                    "The stopwords file.")

# Required flag.
flags.mark_flag_as_required("input_file")
flags.mark_flag_as_required("output_file")
import ioutil

def should_tokenize(page):
    """Determines if we should tokenize the page or not."""
    if page.redirect_title:
        return False
    if not page.title:
        return False
    if not page.revision:
        return False
    return True


discard_re = re.compile("'{2,3,4}|\{{2}|\}{2}|\[{2}|\]{2}|\||/|=|\(|\)")


def tokenize_page(page):
    """Converts the text into a list of tokens"""
    body_text = page.revision[0].text
    body_text = body_text.lower()
    # El cheapo cleaning
    # Get rid of the italics and bold markups
    body_text = re.sub(discard_re, ' ', body_text)
    tokens = nltk.word_tokenize(body_text)
    return tokens


# For efficiency we compile this outside.
# This matches anything that has double square brackets [[ foo ]].
title_re = re.compile('\[\[[^\]]*\]\]')


def extract_title_cooccurrence(page):
    """Retrieves title to title co-occurrences."""
    titles = []
    all_blocks = title_re.findall(page.revision[0].text)
    for block in all_blocks:
        block = block.replace('[', '').replace(']', '')
        # The subsequent stuff after the | is just display text.
        subtitles = block.split('|')
        titles.append(subtitles[0])
    return titles


def process_page(page, stopwords):
    """Processes a page."""
    # Extract out all the title to title similarities.
    titles = extract_title_cooccurrence(page)
    doc = nlp_pb.TextDocument()
    doc.primary = page.title
    doc.secondary.extend(titles)
    doc.url = ('https://en.wikipedia.org/wiki/%s' %
               doc.primary)
    tokens = tokenize_page(page)
    for token in tokens:
        if token not in stopwords:
            doc.tokens.append(token)
    return doc


def process_one_file(inputfile, outputfile, stopwords):
    """Reads one file."""
    with bz2.open(inputfile, 'rb') as ifile:
        with bz2.open(outputfile, 'wb') as ofile:
            for line in ifile:
                # Drop the trailing \n
                ll = line[:-1]
                serialized = base64.b64decode(ll)
                page = wiki_pb.Page()
                page.ParseFromString(serialized)
                if not should_tokenize(page):
                    continue
                doc = process_page(page, stopwords)
                print(doc.primary)
                serialized = base64.b64encode(doc.SerializeToString())
                ofile.write(serialized)
                ofile.write(b'\n')


def main(argv):
    """Main function."""
    del argv  # Unused.
    nltk.download('punkt')
    if FLAGS.stopwords_file:
        stopwords = ioutil.load_stopwords(FLAGS.stopwords_file)
    else:
        stopwords = set()
    process_one_file(FLAGS.input_file, FLAGS.output_file, stopwords)


if __name__ == "__main__":
    app.run(main)
