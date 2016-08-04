"""
File for training network on mnist data.
This acts as a sanity check that our network code really can learn and predict.
"""

import numpy as np
import cv2
import warnings
import net.mnist
import net.network


def main():

    training_data, test_data = net.mnist.load_mnist_data()

    vectorized_training_data = net.mnist.vectorize_mnist(training_data)
    vectorized_test_data = net.mnist.vectorize_mnist(test_data)

    image_size = training_data[0][0].size

    hyperparameters = net.network.NetHyperparameters(
        epochs=20, learning_rate=0.01, regularization_coefficient=0.1, batch_size=8)

    network = net.network.Net(layers=[image_size, 30, 10])
    trainer = net.network.Trainer(hyperparameters)

    trainer.train(
            network=network, data=vectorized_training_data,
            test_data=vectorized_test_data, output_path='./results/mnist_net.json')

    test_accuracy = network.get_accuracy(vectorized_test_data)
    print("Test accuracy is {}".format(test_accuracy))


if __name__ == "__main__":

    main()
