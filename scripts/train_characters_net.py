"""
Module for training net to recognize printed characters
"""

import net.utilities
import random

import net.network
import net.characters


def main():

    base_data_path = "../../data/characters/data/"

    characters = net.characters.get_characters_list(base_data_path)
    data_dictionary = net.utilities.get_data_dictionary(base_data_path, characters)

    training_data, test_data = net.utilities.get_training_test_data_split(data_dictionary, 0.8)

    encoder = net.utilities.Encoder(characters)
    training_data = net.characters.transform_data(training_data, encoder)
    test_data = net.characters.transform_data(test_data, encoder)

    random.shuffle(training_data)
    random.shuffle(test_data)

    image_size = training_data[0][0].size
    labels_size = len(characters)

    hyperparameters = net.network.NetHyperparameters(
        epochs=1000, learning_rate=0.01, regularization_coefficient=0.01, batch_size=1)

    network = net.network.Net(
            layers=[image_size, 50, labels_size])

    trainer = net.network.NetworkTrainer(hyperparameters)
    trainer.train(network=network, data=training_data, test_data=test_data,
                  output_path="./results/characters_net.json")


if __name__ == "__main__":
    main()