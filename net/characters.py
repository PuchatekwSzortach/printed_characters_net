"""
A bunch of utility functions related to characters dataset
"""

import os
import numpy as np


def get_characters_list(data_path):
    """
    Give a path which contains directories with data for each character,
    return a list with characters we have data for
    :param data_path: path to data directory. It is assumed to contain
    directories names after characters, with each directory containing images of that character
    :return: a list of characters for which we have images
    """

    # First result contains parent directory itself, so need to discard it.
    # Also full paths are given and we want only basenames
    return [os.path.basename(result[0]) for result in os.walk(data_path)][1:]


def transform_image(image):
    """
    Given a 2D numpy array representing image, transform it to format
    recognized by net
    :param image: 2D numpy array
    :return: 1D numpy column
    """

    scaled_image = image.astype(np.float64) / 256.0
    flipped_image = 1 - scaled_image
    return flipped_image.reshape([image.size, 1])


def transform_data(data, encoder):
    """
    Given a list of tuples (label, image), transform data into a format
    that can be accepted by neural network.
    This consists of:
    1. Encoding label to a hot-one encoding.
    2. Scaling image to a -1/+1 range
    3. Casting image to a 1D vector
    :param data:
    :param encoder:
    :return: a list of tuples (label, image), where both label and image
    are suitable transformed
    """

    transformed_data = []

    for image, label in data:

        tuple = transform_image(image), encoder.encode(label)
        transformed_data.append(tuple)

    return transformed_data
