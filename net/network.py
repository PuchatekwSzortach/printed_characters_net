"""
Module with network code
"""
import warnings
import numpy as np
import random
import time
import json
import os

import net.utilities

warnings.filterwarnings('error')


class NetHyperparameters:
    """
    A very simple structure bundling together net hyperparameters
    """

    def __init__(self, epochs, learning_rate, regularization_coefficient, batch_size):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization_coefficient = regularization_coefficient
        self.batch_size = batch_size


class Net:
    """
    A simple neural network
    """

    def __init__(self, layers, output_path):

        self.layers = layers
        self.output_path = output_path

        self.biases = [np.random.rand(nodes_out, 1) for nodes_out in layers[1:]]

        self.weights = [np.random.rand(nodes_out, nodes_in) / np.sqrt(nodes_in)
                        for nodes_in, nodes_out in zip(layers[:-1], layers[1:])]

    def feedforward(self, x):

        _, activations = self.verbose_feedforward(x)
        return activations[-1]

    def verbose_feedforward(self, x):
        """
        Feedforward that eturns a list of preactivations and activations
        :param x: input
        :return: a tuple (preactivations list, activations list)
        """

        zs = []
        activations = [x]

        a = x

        for w, b in zip(self.weights, self.biases):

            z = np.dot(w, a) + b
            zs.append(z)

            a = net.utilities.relu(z)
            activations.append(a)

        # Use softmax output
        activations[-1] = net.utilities.softmax(zs[-1])

        return zs, activations

    def get_output_layer_error(self, y, prediction):
        """
        Get output layer error for cross entropy cost
        :param y:
        :param prediction:
        :return:
        """
        return prediction - y

    def get_accuracy(self, data):

        is_correct = []

        for x, y in data:

            prediction = self.feedforward(x)

            prediction_saturated = np.zeros(prediction.shape)
            prediction_saturated[np.argmax(prediction)] = 1

            is_correct.append(int(np.all(prediction_saturated == y)))

        return np.sum(is_correct) / len(is_correct)

    def save_net(self):

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        data = {
            "layers": self.layers,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }

        with open(self.output_path, "w") as file:
            json.dump(data, file)


class NetworkTrainer:
    """
    Class training a network
    """

    def __init__(self, hyperparameters):

        self.hyperparameters = hyperparameters

    def train(self, network, data, test_data):

        best_accuracy = 0

        for epoch in range(self.hyperparameters.epochs):

            random.shuffle(data)

            if epoch % 1 == 0:

                print("Epoch {}".format(epoch))
                accuracy = network.get_accuracy(test_data)
                print(accuracy)

                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    network.save_net()

            if epoch % 10 == 0:
                self.hyperparameters.learning_rate *= 0.25

            batched_data = net.utilities.get_data_batches(data, self.hyperparameters.batch_size)

            for index, batch in enumerate(batched_data):
                x_batch, y_batch = net.utilities.data_tuples_to_matrices(batch)
                self._update(network, x_batch, y_batch, self.hyperparameters.learning_rate)

    def _update(self, network, x, y, learning_rate):

        zs, activations = network.verbose_feedforward(x)

        bias_gradients = [None] * len(network.biases)
        weights_gradients = [None] * len(network.weights)

        error = network.get_output_layer_error(y, activations[-1])

        bias_gradients[-1] = np.mean(error, axis=1).reshape(error.shape[0], 1)
        weights_gradients[-1] = np.dot(error, activations[-2].T) / self.hyperparameters.batch_size

        indices = range(len(network.weights) - 2, -1, -1)

        for index in indices:

            error = np.dot(network.weights[index + 1].T, error) * net.utilities.relu_prime(zs[index])

            bias_gradients[index] = np.mean(error, axis=1).reshape(error.shape[0], 1)

            regularization_derivative = \
                self.hyperparameters.regularization_coefficient * network.weights[index] /\
                self.hyperparameters.batch_size

            weights_gradients[index] = \
                (np.dot(error, activations[index].T) / self.hyperparameters.batch_size) + \
                regularization_derivative

        network.weights = [w - (learning_rate * w_grad)
                       for w, w_grad in zip(network.weights, weights_gradients)]

        network.biases = [b - (learning_rate * b_grad)
                      for b, b_grad in zip(network.biases, bias_gradients)]
