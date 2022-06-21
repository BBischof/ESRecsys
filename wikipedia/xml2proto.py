#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  This module reads wikipedia XML files and saves them as uuencoded gzip..
"""
import base64
import bz2
import getopt
import os
import sys
import xml.etree.ElementTree as ET
import wikipedia_pb2 as wiki_pb
from absl import app
from absl import flags
from absl import logging

FLAGS=flags.FLAGS
_INPUT_FILE = flags.DEFINE_string("input_file", None, "Name of wikipedia xml file.")
_OUTPUT_FILE = flags.DEFINE_string("output_file", None,
                                   "Name of .pb.bz2 file to write out.")

def read_file(filename):
    """Reads one wikipedia xml file and returns the iterator"""
    if 'bz2' in filename:
        file = bz2.open(filename, 'r')
    else:
        file = open(filename, 'r')
    it = ET.iterparse(file, events=('start', 'end'))
    return it


def parse_contributor(contributor_tree, nslen, contributor):
    """Parses a wikipedia contributor"""
    for el in contributor_tree:
        tag = el.tag[nslen:]
        if tag == 'username':
            contributor.username = el.text
        elif tag == 'id':
            contributor.id = int(el.text)
        elif tag == 'ip':
            contributor.ip = el.text
    return contributor


def parse_revision(revision_tree, nslen, revision):
    """Parses a wikipedia revision"""
    for el in revision_tree:
        tag = el.tag[nslen:]
        if tag == 'contributor':
            parse_contributor(el, nslen, revision.contributor)
        elif tag == 'id':
            revision.id = int(el.text)
        elif tag == 'parentid':
            revision.parentid = int(el.text)
        elif tag == 'timestamp':
            revision.timestamp = el.text
        elif tag == 'model':
            revision.model = el.text
        elif tag == 'format':
            revision.format = el.text
        elif tag == 'text':
            revision.text = el.text
        elif tag == 'sha1':
            revision.sha1 = el.text
    return revision


def parse_page(page_tree, nslen, page):
    """Parses a wikipedia page"""
    for el in page_tree:
        tag = el.tag[nslen:]
        if tag == 'revision':
            revision = page.revision.add()
            parse_revision(el, nslen, revision)
        elif tag == 'title':
            page.title = el.text
        elif tag == 'ns':
            page.ns = int(el.text)
        elif tag == 'id':
            page.id = int(el.text)
        elif tag == 'redirect':
            page.redirect_title = el.attrib['title']
    return page


def process_one_file(inputfile, outputfile):
    """Processes one wikipedia xml file"""
    os.mkdir(outputfile)
    it = read_file(inputfile)
    # The first element contains the mediawiki tag with the
    # xml namespace prefix, we obtain the prefix in the following.
    first = next(it)[1]
    xmlns = first.tag[:-len('mediawiki')]
    nslen = len(xmlns)
    count = 0
    part = 0
    output_filename = os.path.join(outputfile, 'part-%05d.bz2' % part)
    outfile = bz2.open(output_filename, 'wb')
    for ev, el in it:
        # Get the part of the tag after the namespace
        tag = el.tag[nslen:]
        # Keep on parsing until we get a page end.
        if ev != 'end':
            continue
        if tag == 'page':
            count = count + 1
            if count % 1000 == 0:
                 logging.info("Processed %d pages" % count)
                 outfile.flush()
                 outfile.close()
                 part = part + 1
                 output_filename = os.path.join(outputfile,
                                                'part-%05d.bz2' % part)
                 outfile = bz2.open(output_filename, 'wb')
            page = wiki_pb.Page()
            parse_page(el, nslen, page)
            encoded = base64.b64encode(page.SerializeToString())
            outfile.write(encoded)
            outfile.write(b'\n')
    outfile.flush()
    outfile.close()


def main(argv):
    """Main function."""
    del argv # unused
    process_one_file(_INPUT_FILE.value, _OUTPUT_FILE.value)


if __name__ == "__main__":
    app.run(main)
