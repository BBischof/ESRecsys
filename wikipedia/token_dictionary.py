"""
    Token dictionary, a class that handles conversion of a list of tokens to indices.
"""

import base64
import bz2
import nlp_pb2 as nlp_pb


class TokenDictionary:
    def __init__(self):
        self.__token2index = {}
        self.__max_doc_frequency = 0
        self.__token_stat = []

    def load(self, dictionary_file):
        """Loads a dictionary."""
        count = 0
        with bz2.open(dictionary_file, 'rb') as file:
            for line in file:
                # Drop the trailing \n
                line = line[:-1]
                serialized = base64.b64decode(line)
                token_stat = nlp_pb.TokenStat()
                token_stat.ParseFromString(serialized)
                assert(token_stat.index == count)
                self.__token2index[token_stat.token] = token_stat.index
                self.__max_doc_frequency = max(self.__max_doc_frequency, token_stat.doc_frequency)
                self.__token_stat.append(token_stat)
                # Ensure that we can use a direct index scheme to look up words from indices.
                count += 1

    def __init__(self, dictionary_file):
        self.__token2index = {}
        self.__max_doc_frequency = 0
        self.__token_stat = []
        self.load(dictionary_file)

    def get_dictionary_size(self):
        return len(self.__token2index)

    def get_max_doc_frequency(self):
        return self.__max_doc_frequency

    def get_doc_frequency(self, token_index):
        return self.__token_stat[token_index].doc_frequency

    def get_token_index(self, token):
        if token not in self.__token2index:
            return None
        return self.__token2index[token]

    def get_token(self, token_index):
        return self.__token_stat[token_index].token
