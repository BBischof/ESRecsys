"""
    Cooccurrence matrix library.
"""

import base64
import bz2
import nlp_pb2 as nlp_pb
import numpy as np

class CooccurrenceMatrix:

    def __reset(self):
        self.__matrix = []

    def debug_print(self, index, token_dictionary, num_terms):
        proto = self.__matrix[index]
        num_others = len(proto.other_index)
        token = token_dictionary.get_token(proto.index)
        print('Token [%s]' % token)
        nt = min(num_terms, num_others)
        for i in range(nt):
            token = token_dictionary.get_token(proto.other_index[i])
            print(' %s : %f' % (token, proto.count[i]))


    def load(self, input_file):
        """Loads a dictionary."""
        self.__reset()
        count = 0
        # List of co-occurrence of token i with token j.
        with bz2.open(input_file, 'rb') as file:
            for line in file:
                # Drop the trailing \n
                line = line[:-1]
                serialized = base64.b64decode(line)
                proto = nlp_pb.CooccurrenceRow()
                proto.ParseFromString(serialized)
                assert (proto.index == count)
                self.__matrix.append(proto)
                count += 1

    def __init__(self, input_file):
        self.load(input_file)


class CooccurrenceGenerator:
    def __init__(self, input_file):
        self.input_file = input_file

    def get_item(self):
        """Gets a single item of i, j, count"""
        while True:
            print('Opening %s' % self.input_file)
            with bz2.open(self.input_file, 'rb') as file:
                for line in file:
                    # Drop the trailing \n
                    line = line[:-1]
                    serialized = base64.b64decode(line)
                    proto = nlp_pb.CooccurrenceRow()
                    proto.ParseFromString(serialized)
                    count = len(proto.other_index)
                    for i in range(count):
                        yield (proto.index, proto.other_index[i], proto.count[i])

    def get_shuffled_items(self, num_items):
        """Pre-fetches and shuffles num_items of stuff."""
        iterator = self.get_item()
        while True:
            items = [next(iterator) for _ in range(num_items)]
            np.random.shuffle(items)
            for item in items:
                yield item

    def get_batch(self, batch_size, shuffle_size=0):
        if shuffle_size:
            iterator = self.get_shuffled_items(shuffle_size)
        else:
            iterator = self.get_item()
        while True:
            token1 = []
            token2 = []
            token_count = []
            for _ in range(batch_size):
                item = next(iterator)
                token1.append(item[0])
                token2.append(item[1])
                token_count.append(item[2])
            x = [np.asarray(token1, dtype=np.int32),
                 np.asarray(token2, dtype=np.int32)]
            y = np.asarray(token_count, dtype=np.float32)
            yield (x, y)
