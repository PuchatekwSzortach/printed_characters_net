"""
Module with functions for loading mnist dataset. We use mnist as a sanity check for our network.
"""

import gzip
import struct

import numpy as np
import cv2


def get_labels(path):

    binary_data = gzip.open(path).read()

    header_struct = struct.Struct("> 2i")
    _, labels_count = header_struct.unpack(binary_data[:8])

    labels_struct = struct.Struct("> {}B".format(labels_count))
    labels = labels_struct.unpack(binary_data[8:])
    return labels


def get_images(path):

    binary_data = gzip.open(path).read()

    header_struct = struct.Struct("> 4i")
    _, images_count, rows_count, columns_count = header_struct.unpack(binary_data[:16])

    pixels_struct = struct.Struct("> {}B".format(images_count * rows_count * columns_count))
    pixels = pixels_struct.unpack(binary_data[16:])

    images_stack = np.array(pixels).reshape([images_count, rows_count, columns_count]) / 255.0

    images = [image for image in images_stack]
    return images


def load_mnist_data():

    training_images = get_images("../../data/mnist/train-images-idx3-ubyte.gz")
    training_labels = get_labels("../../data/mnist/train-labels-idx1-ubyte.gz")

    test_images = get_images("../../data/mnist/t10k-images-idx3-ubyte.gz")
    test_labels = get_labels("../../data/mnist/t10k-labels-idx1-ubyte.gz")

    training_data = [(image, label) for image, label in zip(training_images, training_labels)]
    test_data = [(image, label) for image, label in zip(test_images, test_labels)]

    return training_data, test_data


