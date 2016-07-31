# -*- coding: utf-8 -*-
"""
Various utilities
"""

import numpy as np
import cv2
import glob
import tqdm
import multiprocessing
import random
import orderedset


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

    print("Loading data")
    with multiprocessing.Pool() as pool:

        # Get a list of futures queued on the pool
        results = [pool.apply_async(get_images, (base_path + label + "/",)) for label in labels]

        # Build a results dictionary using labels as key and results of futures, which are
        # evaluated to lists of data for that label, as values. Wrap iteration over labels in tqdm
        # to add a printed progress bar to terminal output
        data_dictionary = {label: result.get() for label, result in zip(tqdm.tqdm(labels), results)}

    return data_dictionary


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

        random.shuffle(images)

        training_size = int(len(images) * split_ratio)
        data = [(image, label) for image in images]

        training_data.extend(data[:training_size])
        test_data.extend(data[training_size:])

    return training_data, test_data


def get_data_batches(data, batch_size):
    """
    Given a data in a list, return a list of batches, each batch of size batch_size.
    If len(data) doesn't divide evenly by batch_size, leftover items are not returned.
    :param data: list of data items
    :param batch_size: size or return batches
    :return: list of batches, each batch a list of data elements
    """

    last_batch_start_index = int(len(data) / batch_size) * batch_size

    batched_data = [data[index: index + batch_size] for index in
                    range(0, last_batch_start_index, batch_size)]

    return batched_data


def data_tuples_to_matrices(data):
    """
    Given a list of tuples, where each tuple is made up of numpy column vectors,
    convert them to matrices, such that corresponding elements from each tuple are
    laid out in matrices columns-wise.
    E.g. if data is a 5-elements list of 10x1 and 20x1 tuples, the result is a single
    tuple of 10x5 and 20x5 matrices.
    :param data: a list of tuples. Each tuple contains numpy column vectors
    :return: a list of 2D numpy arrays.
    """

    data_length = len(data)
    matrices_count = len(data[0])

    matrices = []

    for matrix_index in range(matrices_count):

        # Our reference for matrix size and type
        vector_length = data[0][matrix_index].shape[0]
        vector_type = data[0][matrix_index].dtype

        # Allocate matrix
        matrix = np.zeros(shape=[vector_length, data_length]).astype(vector_type)

        for data_index in range(data_length):

            matrix[:, data_index] = data[data_index][matrix_index].reshape([vector_length])

        matrices.append(matrix)

    return matrices


def sigmoid(z):

    try:

        # Do a bit of acrobatics to ensure we don't compute values that lead to division
        # by infinity. For inputs that would lead to that, simply return 0
        output = np.zeros(z.shape)

        indices = np.where(-50 < z)
        output[indices] = 1 / (1 + np.exp(-z[indices]))

        return output

    except RuntimeWarning as problem:

        print("Runtime warning occurred in sigmoid for input")
        print(problem)
        print(z)
        print("that gives output")
        print(output)
        exit(0)


def sigmoid_prime(z):

    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):

    try:
        # Clip values to sensible range for numerical stability
        clipped = np.clip(z, -50, 50)
        return np.exp(clipped) / np.sum(np.exp(clipped), axis=0)

    except RuntimeWarning as problem:

        print("Runtime warning occurred in softmax")
        print(problem)
        print("For input")
        print(z)
        exit(0)


def relu(z):
    return z * (z > 0)


def relu_prime(z):
    return (z > 0).astype(np.float32)


def remove_visually_identical_characters(characters):
    """
    Some Japanese characters, especially some hiragana and katakana, look the same.
    So if our set contains both, remove one of them
    :param characters:
    :return: characters with visually identical doubles removed and ordered of characters preserver
    """

    characters = list(orderedset.OrderedSet(characters))

    # There characters have different unicodes, but visually they are identical
    identical_characters_list = [
        ['ぺ', 'ペ'],
        ['ベ', 'べ'],
        ['ヘ', 'へ'],
    ]

    for identical_characters in identical_characters_list:

        try:
            first_index = characters.index(identical_characters[0])
            second_index = characters.index(identical_characters[1])

            index_to_pop = first_index if first_index > second_index else second_index
            characters.pop(index_to_pop)

        except ValueError:
            # It's okay if element doesn't exist in the list.
            # In fact it means we have no duplicates to remove
            pass

    return characters
