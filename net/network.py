"""
Module with network code
"""
import warnings
import numpy as np
import random

import net.utilities

# warnings.filterwarnings('error')

def sigmoid(z):

    try:

        # Do a bit of acrobatics to ensure we don't compute values that lead to division
        # by infinity. For inputs that would lead to that, simply return 0
        output = np.zeros(z.shape)

        indices = np.where(-50 < z)
        output[indices] = 1 / (1 + np.exp(-z[indices]))

        return output

    except RuntimeWarning as problem:

        print("Runtime warning occurred in sigmoid for input")
        print(problem)
        print(z)
        print("that gives output")
        print(output)
        exit(0)

def sigmoid_prime(z):

    return sigmoid(z) * (1 - sigmoid(z))


class Net:
    """
    A simple neural network
    """

    def __init__(self, layers, epochs, learning_rate, batch_size):

        self.layers = layers

        self.biases = [np.random.rand(nodes_out, 1) for nodes_out in layers[1:]]

        self.weights = [np.random.rand(nodes_out, nodes_in) / np.sqrt(nodes_in)
                        for nodes_in, nodes_out in zip(layers[:-1], layers[1:])]

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def feedforward(self, x):

        a = x.copy()

        for w, b in zip(self.weights, self.biases):

            z = np.dot(w, a) + b
            a = sigmoid(z)

        return a

    def train(self, data, test_data):

        for epoch in range(self.epochs):

            random.shuffle(data)

            if epoch % 1 == 0:

                print("Epoch {}".format(epoch))
                print(self.get_accuracy(test_data))

            batched_data = net.utilities.get_data_batches(data, self.batch_size)

            for x, y in data:
                self._update(x, y, self.learning_rate)

    def _update(self, x, y, learning_rate):


        zs = []
        activations = [x]

        a = x

        for w, b in zip(self.weights, self.biases):

            z = np.dot(w, a) + b
            zs.append(z)

            a = sigmoid(z)
            activations.append(a)

        bias_gradients = [None] * len(self.biases)
        weights_gradients = [None] * len(self.weights)

        error = self._get_output_layer_error(y, activations[-1])

        bias_gradients[-1] = error
        weights_gradients[-1] = np.dot(error, activations[-2].T)

        indices = range(len(self.weights) - 2, -1, -1)

        for index in indices:

            error = np.dot(self.weights[index + 1].T, error) * sigmoid_prime(zs[index])

            bias_gradients[index] = error
            weights_gradients[index] = np.dot(error, activations[index].T)

        self.weights = [w - (learning_rate * w_grad)
                        for w, w_grad in zip(self.weights, weights_gradients)]

        self.biases = [b - (learning_rate * b_grad)
                        for b, b_grad in zip(self.biases, bias_gradients)]

    def _get_cost(self, y, prediction):
        """
        Compute cross entropy cost
        :param y:
        :param prediction:
        :return:
        """

        a = y * np.nan_to_num(np.log(prediction))
        b = (1 - y) * np.nan_to_num(np.log(1 - prediction))

        cost = -(a + b)
        return cost

    def _get_output_layer_error(self, y, prediction):
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


