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

    plt.subplot(layers_count + 1, 1, 1)
    accuracy = [training_data[epoch]['accuracy'] for epoch in epochs]
    plt.plot(epochs, accuracy)
    plt.title('Accuracy')

    # Plot weights
    for layer in range(layers_count):

        plt.subplot(layers_count + 1, 1, layer + 2)

        # This gives me a list of epochs size, each element of which contains values of different
        # percentiles at that epoch
        weights_percentiles = [training_data[epoch]['weights_percentiles'][layer] for epoch in epochs]

        # Now transpose above, so we have n percentiles lists, each of epochs numbers length
        individual_percentiles = [percentile for percentile in zip(*weights_percentiles)]
        labels = ['0%', '25%', '50%', '75%', '100%']

        for percentile, label in zip(individual_percentiles, labels):
            plt.plot(epochs, percentile, label=label)

        plt.title("Layer {} weights percentiles".format(layer))
        plt.legend()

    plt.show()


def main():

    database_name = configobj.ConfigObj('configuration.ini')['database_name']
    shelf = shelve.open(database_name, writeback=True)
    print(shelf['network_parameters'])
    print(shelf['hyperparameters'])

    plot_training_data(shelf['training'])

if __name__ == "__main__":

    main()