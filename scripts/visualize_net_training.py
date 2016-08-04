"""
A simple program for visualizing networks development as it is trained
"""

import shelve
import configobj
import matplotlib.pyplot as plt
import seaborn


def plot_training_data(training_data):

    epochs = sorted(training_data.keys())
    layers_count = len(training_data[0]['max_weights'])

    print('layers count')
    print(layers_count)

    plt.subplot(layers_count + 1, 1, 1)
    accuracy = [training_data[epoch]['accuracy'] for epoch in epochs]
    plt.plot(epochs, accuracy)
    plt.title('Accuracy')

    for layer in range(layers_count):

        plt.subplot(layers_count + 1, 1, layer + 2)

        max_weights = [training_data[epoch]['max_weights'][layer] for epoch in epochs]
        min_weights = [training_data[epoch]['min_weights'][layer] for epoch in epochs]
        plt.plot(epochs, max_weights)
        plt.plot(epochs, min_weights)
        plt.title("Weights into layer {}".format(layer))

    plt.show()


def main():

    database_name = configobj.ConfigObj('configuration.ini')['database_name']
    shelf = shelve.open(database_name, writeback=True)
    print(shelf['network_parameters'])
    print(shelf['hyperparameters'])

    plot_training_data(shelf['training'])

if __name__ == "__main__":

    main()