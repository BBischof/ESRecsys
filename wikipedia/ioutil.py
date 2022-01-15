"""
  I/O utilities.
"""


def load_stopwords(input_file):
    """Loads a stopwords file"""
    stopwords = []
    with open(input_file, 'r') as file:
        for line in file:
            stopwords.append(line[:-1])
    print('%d stopwords loaded' % len(stopwords))
    return set(stopwords)

