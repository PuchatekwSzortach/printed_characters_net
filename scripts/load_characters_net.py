"""
A simple file that essentially checks if previously trained characters network can be
successfully retrieved from a file and used for predictions.
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
    test_data = net.characters.transform_data(test_data, encoder)

    network = net.network.Net.from_file("./results/characters_net.json")

    test_accuracy = network.get_accuracy(test_data)
    print("Test accuracy is {}".format(test_accuracy))

if __name__ == "__main__":

    main()
