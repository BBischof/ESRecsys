"""
  Makes co-occurrence matrix.
"""
import base64
import bz2
import nlp_pb2 as nlp_pb
from absl import app
from absl import flags
from token_dictionary import TokenDictionary

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input doc.pb.b64.bz2 file.")
flags.DEFINE_string("output_file", None, "Input cooccur.pb.b64.bz2 file.")
flags.DEFINE_string("token_dictionary", None, "The token dictionary file.")
flags.DEFINE_integer("context_window", 10, "Size of the context window.")

# Required flag.
flags.mark_flag_as_required("input_file")


def get_token_indices(doc, token_dictionary):
    """Converts the document to tokens."""
    token_idx = []
    for token in doc.tokens:
        token_index = token_dictionary.get_token_index(token)
        if token_index is not None:
            token_idx.append(token_index)
    return token_idx

def process_doc(doc, token_dictionary, context_window, cooccur):
    """Counts co-occurrences of tokens."""
    # Flatten the document out into token indices.
    token_idx = get_token_indices(doc, token_dictionary)
    num_words = len(token_idx)
    for i in range(num_words):
        start = max(0, i - context_window)
        end = min(num_words, i + context_window)
        my_idx = token_idx[i]
        row = cooccur[my_idx]
        for j in range(start, end):
            dist = abs(i - j)
            other_idx = token_idx[j]
            # Don't count the same word.
            if my_idx == other_idx:
                continue
            increment = 1.0 / float(dist)
            if other_idx not in row:
                row[other_idx] = increment
            else:
                row[other_idx] += increment


def process_file(input_file, output_file, token_dictionary, context_window):
    count = 0
    cooccur = [{} for _ in range(token_dictionary.get_dictionary_size())]
    with bz2.open(input_file, 'rb') as file:
        for line in file:
            # Drop the trailing \n
            line = line[:-1]
            serialized = base64.b64decode(line)
            doc = nlp_pb.TextDocument()
            doc.ParseFromString(serialized)
            print(doc.primary)
            process_doc(doc, token_dictionary, context_window, cooccur)
    with bz2.open(output_file, 'wb') as file:
        for i in range(token_dictionary.get_dictionary_size()):
            proto = nlp_pb.CooccurrenceRow()
            row = cooccur[i]
            proto.index = i
            to_sort = []
            for key in row:
                to_sort.append([row[key], key])
            # Sort by counts in decreasing order.
            to_sort = sorted(to_sort, reverse=True)
            for kv in to_sort:
                proto.other_index.append(kv[1])
                proto.count.append(kv[0])
            serialized = proto.SerializeToString()
            file.write(base64.b64encode(serialized))
            file.write(b'\n')


def main(argv):
    """Main function."""
    del argv  # Unused.
    token_dictionary = TokenDictionary(FLAGS.token_dictionary)
    process_file(FLAGS.input_file, FLAGS.output_file, token_dictionary, FLAGS.context_window)


if __name__ == "__main__":
    app.run(main)
