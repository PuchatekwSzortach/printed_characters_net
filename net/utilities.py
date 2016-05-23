"""
Various utilities
"""

import numpy as np
import cv2
import glob

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

        encoding = np.zeros([len(self.labels), 1])

        index = self.labels.index(label)
        encoding[index] = 1

        return encoding

    def decode(self, encoding):
        """
        Given encoding, return a label it represents
        :param encoding: one-hot encoded vector
        :return: label encoding represents
        """

        index = np.argmax(encoding)
        return self.labels[index]


def get_images(path):
    """
    Given a path, return a list of grayscale images found at that path
    :param path: Path to images
    :return: A list of grayscale images
    """
    images_paths = glob.glob(path + "/*.jpg")
    images = [cv2.imread(image_path) for image_path in images_paths]
    return [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]


def get_data_dictionary(base_path, labels):
    """
    Given a base path and a list of labels, return a data dictionary.
    Base path is assumed to contain subdirectories, one subdirectory for each label.
    Each subdirectory is named after the label and contains jpg images with data
    for that label.
    :param base_path: Parent directory for all labels data
    :param labels: A list of labels for which data should be read
    :return: A dictionary of form {label : images list}
    """

    return {label : get_images(base_path + label + "/") for label in labels}


def get_training_test_data_split(data_dictionary, split_ratio):
    """
    Given a data dictionary of structure {label : images list},
    return a training list and a test list. Each list element is made up of
    (image, label) tuple. Split ratio determines ratio of data samples that are
    put into training set. Remaining items are put into test set.
    :param data_dictionary:
    :param split_ratio:
    :return:
    """

    training_data = []
    test_data = []

    for label, images in data_dictionary.items():

        training_size = int(len(images) * split_ratio)
        data = [(image, label) for image in images]

        training_data.extend(data[:training_size])
        test_data.extend(data[training_size:])

    return training_data, test_data
