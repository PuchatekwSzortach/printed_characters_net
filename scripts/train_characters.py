"""
Module for training net to recognize printed characters
"""

import os
import net.utilities
import cv2
import numpy as np
import net.network
import random


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

        scaled_image = image.astype(np.float64) / 256.0
        encoded_label = encoder.encode(label)

        flipped_image = 1 - scaled_image

        tuple = flipped_image.reshape([image.size, 1]), encoded_label
        transformed_data.append(tuple)

    return transformed_data


def main():

    base_data_path = "../../data/characters/data/"

    characters = get_characters_list(base_data_path)
    data_dictionary = net.utilities.get_data_dictionary(base_data_path, characters)

    training_data, test_data = net.utilities.get_training_test_data_split(data_dictionary, 0.8)

    encoder = net.utilities.Encoder(characters)
    training_data = transform_data(training_data, encoder)
    test_data = transform_data(test_data, encoder)

    random.shuffle(training_data)
    random.shuffle(test_data)

    image_size = training_data[0][0].size
    labels_size = len(characters)

    network = net.network.Net(
        layers=[image_size, 100, labels_size], epochs=100, learning_rate=0.01, batch_size=4)

    network.train(data=training_data, test_data=test_data)


if __name__ == "__main__":
    main()