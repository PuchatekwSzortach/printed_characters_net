"""
Various utilities
"""

import numpy as np

class Encoder:
    """
    A simple class for one hot encoding and decoding labels.
    """

    def __init__(self, labels):
        """
        :param labels: A complete list of labels to be used to define one-hot encoding.
         Position in list corresponds to position in encoding.
        :return:
        """

        self.labels = labels

    def encode(self, label):
        """
        Given a label, return a one-hot encoding for that label
        :param label:
        :return: a one-hot encoded vector
        """

        encoding = np.zeros([1, len(self.labels)])

        index = self.labels.index(label)
        encoding[0, index] = 1

        return encoding

    def decode(self, encoding):
        """
        Given encoding, return a label it represents
        :param encoding: one-hot encoded vector
        :return: label encoding represents
        """

        index = np.argmax(encoding)
        return self.labels[index]