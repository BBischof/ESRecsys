"""
  This reads a doc.pb.b64.bz2 file and generates a dictionary.
"""
import base64
import bz2
import nlp_pb2 as nlp_pb
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None, "Input doc.pb.b64.bz2 file.")
flags.DEFINE_string("title_output", None,
                    "The title dictionary output file.")
flags.DEFINE_string("token_output", None,
                    "The token dictionary output file.")
flags.DEFINE_integer("min_term_frequency", 10,
                     "Minimum term frequency")

# Required flag.
flags.mark_flag_as_required("input_file")


def update_dict_term(term, dictionary):
    """Updates a dictionary with a term."""
    if term in dictionary:
        x = dictionary[term]
    else:
        x = nlp_pb.TokenStat()
        x.token = term
        dictionary[term] = x
    x.frequency += 1


def update_dict_doc(term, dictionary):
    """Updates a dictionary with the doc frequency."""
    dictionary[term].doc_frequency += 1


def count_tokens(doc, title_dict, token_dict):
    """Counts the titles and tokens."""
    # Handle the titles.
    all_titles = [doc.primary]
    all_titles.extend(doc.secondary)
    for title in all_titles:
        update_dict_term(title, title_dict)
    # Only titles have urls.
    title_dict[doc.primary].url = doc.url
    title_set = set(all_titles)
    for title in title_set:
        update_dict_doc(title, title_dict)
    # Handle the tokens.
    for term in doc.tokens:
        update_dict_term(term, token_dict)
    term_set = set(doc.tokens)
    for term in term_set:
        update_dict_doc(term, token_dict)


def process_file(inputfile, title_output, token_output,
                 min_term_frequency):
    """Processes documents, creating the title and token dictionaries."""
    title_dict = {}
    token_dict = {}
    with bz2.open(inputfile, 'rb') as file:
        for line in file:
            # Drop the trailing \n
            ll = line[:-1]
            serialized = base64.b64decode(ll)
            doc = nlp_pb.TextDocument()
            doc.ParseFromString(serialized)
            print(doc.primary)
            count_tokens(doc, title_dict, token_dict)

    # Title dictionary, including non primary titles.
    sorted_title_dict = sorted(title_dict.values(), key=lambda x: x.frequency, reverse=True)
    del title_dict
    count = 0
    with bz2.open(title_output, 'wb') as ofile:
        for item in sorted_title_dict:
            item.index = count
            serialized = base64.b64encode(item.SerializeToString())
            ofile.write(serialized)
            ofile.write(b'\n')
            count = count + 1
    print('Wrote %d titles' % count)

    # The term dictionary is filtered by term frequency.
    count = 0
    sorted_token_dict = sorted(token_dict.values(), key=lambda x: x.frequency, reverse=True)
    del token_dict
    with bz2.open(token_output, 'wb') as ofile:
        for item in sorted_token_dict:
            if item.frequency >= min_term_frequency:
                item.index = count
                serialized = base64.b64encode(item.SerializeToString())
                ofile.write(serialized)
                ofile.write(b'\n')
                count = count + 1
    print('Wrote %d tokens above threshold' % count)


def main(argv):
    """Main function."""
    del argv  # Unused.
    process_file(FLAGS.input_file, FLAGS.title_output, FLAGS.token_output,
                 FLAGS.min_term_frequency)


if __name__ == "__main__":
    app.run(main)
