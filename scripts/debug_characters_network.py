"""
Simple program that helps understand classification mistakes
characters network does.
"""

import net.network
import net.characters
import net.utilities


def main():

    base_data_path = "../../data/characters/data/"

    characters = net.characters.get_characters_list(base_data_path)
    data_dictionary = net.utilities.get_data_dictionary(base_data_path, characters)

    training_data, test_data = net.utilities.get_training_test_data_split(data_dictionary, 0.8)

    encoder = net.utilities.Encoder(characters)
    training_data = net.characters.transform_data(training_data, encoder)
    test_data = net.characters.transform_data(test_data, encoder)

    network = net.network.Net.from_file("./results/characters_net.json")
    debugger = net.network.NetworkDebugger(network, encoder)

    mistakes_dictionary = debugger.get_mistakes(training_data + test_data)

    if len(mistakes_dictionary.keys()) == 0:

        print("Net made no mistakes, great!")

    else:

        for key, value in mistakes_dictionary.items():
            print("{} was classified as : {}".format(key, value))


if __name__ == "__main__":
    main()