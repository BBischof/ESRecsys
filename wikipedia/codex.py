"""
  This utility dumps wikipedia .pb.b64.bz2 files to stdout.
"""
import base64
import bz2
import wikipedia_pb2 as wiki_pb
import nlp_pb2 as nlp_pb
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input pb.b64.bz2 file.")
flags.DEFINE_string("proto", "wiki",
                    "Kind of protobuf")

# Required flag.
flags.mark_flag_as_required("input_file")


def main(argv):
    """Main function."""
    del argv  # Unused.
    with bz2.open(FLAGS.input_file, 'rb') as file:
        for line in file:
            # Drop the trailing \n
            ll = line[:-1]
            serialized = base64.b64decode(ll)
            if FLAGS.proto == 'wiki':
                page = wiki_pb.Page()
                page.ParseFromString(serialized)
                print(page)
            elif FLAGS.proto == 'doc':
                doc = nlp_pb.TextDocument()
                doc.ParseFromString(serialized)
                print(doc)
            elif FLAGS.proto == 'tstat':
                stat = nlp_pb.TokenStat()
                stat.ParseFromString(serialized)
                print(stat)


if __name__ == "__main__":
    app.run(main)
