"""
A simple program for visualizing networks development as it is trained
"""

import shelve
import configobj
import matplotlib.pyplot as plt
import seaborn


def plot_training_data(training_data):

    epochs = sorted(training_data.keys())
    layers_count = len(training_data[0]['weights_percentiles'])

    figure, axes = plt.subplots(layers_count + 1, 1, sharex=True)
    accuracy = [training_data[epoch]['accuracy'] for epoch in epochs]

    axes[0].plot(epochs, accuracy)
    axes[0].set_ylim([0, 1])
    axes[0].set_title('Accuracy')

    # Plot weights
    for layer, axis in zip(range(layers_count), axes[1:]):

        # This gives me a list of epochs size, each element of which contains values of different
        # percentiles at that epoch
        weights_percentiles = [training_data[epoch]['weights_percentiles'][layer] for epoch in epochs]

        # Now transpose above, so we have n percentiles lists, each of epochs numbers length
        individual_percentiles = [percentile for percentile in zip(*weights_percentiles)]
        labels = ['0%', '25%', '50%', '75%', '100%']

        for percentile, label in zip(individual_percentiles, labels):
            axis.plot(epochs, percentile, label=label)
            axis.fill_between(epochs, 0, percentile, color=(0, 0.7, 1, 0.2))

        axis.set_title("Layer {} weights percentiles".format(layer))
        axis.legend()

    plt.show()


def main():

    database_name = configobj.ConfigObj('configuration.ini')['database_name']
    shelf = shelve.open(database_name, writeback=True)
    print(shelf['network_parameters'])
    print(shelf['hyperparameters'])

    plot_training_data(shelf['training'])

if __name__ == "__main__":

    main()