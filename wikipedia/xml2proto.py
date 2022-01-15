"""
  This module reads wikipedia XML files and saves them as uuencoded gzip..
"""
import base64
import bz2
from absl import app
from absl import flags
import sys
import xml.etree.ElementTree as ET
import wikipedia_pb2 as wiki_pb

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input XML.bz2 file.")
flags.DEFINE_string("output_pattern", None, "Output as blah-%5d.pb.bz2")
flags.DEFINE_integer("max_articles_per_file", 100000, "Maximum number of articles per file")

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


def process_one_file(inputfile, output_pattern):
    """Processes one wikipedia xml file"""
    it = read_file(inputfile)
    # The first element contains the mediawiki tag with the
    # xml namespace prefix, we obtain the prefix in the following.
    first = next(it)[1]
    xmlns = first.tag[:-len('mediawiki')]
    nslen = len(xmlns)
    num_articles = 0
    num_errors = 0
    file_count = 0
    outfile = None
    for ev, el in it:
        # Get the part of the tag after the namespace
        tag = el.tag[nslen:]
        # Keep on parsing until we get a page end.
        if ev != 'end':
            continue
        if tag == 'page':
            if num_articles % 1000 is 0:
                print('Num articles processed = %d Errors = %d' % (num_articles, num_errors))
            if num_articles % FLAGS.max_articles_per_file is 0:
                if outfile is not None:
                    outfile.close()
                outputfile = output_pattern % file_count
                print('Writing to %s' % outputfile)
                outfile = bz2.open(outputfile, 'w')
                file_count += 1
            page = wiki_pb.Page()
            try:
              parse_page(el, nslen, page)
              encoded = base64.b64encode(page.SerializeToString())
              outfile.write(encoded)
              outfile.write(b'\n')
              num_articles += 1
            except TypeError:
                num_errors += 1

    outfile.close()
    print('Total num articles = %d' % num_articles)
def main(argv):
    """Main function."""
    del argv  # Unused
    process_one_file(FLAGS.input_file, FLAGS.output_pattern)


if __name__ == "__main__":
    app.run(main)
