"""
File for training network on mnist data.
This acts as a sanity check that our network code really can learn and predict.
"""

import numpy as np
import cv2
import warnings
import net.mnist
import net.network


def vectorize_mnist(data):

    images, labels = zip(*data)

    vectorized_images = [image.reshape(image.size, 1) for image in images]

    vectorized_labels = []

    for label in labels:

        vector = np.zeros([10, 1])
        vector[label] = 1
        vectorized_labels.append(vector)

    return list(zip(vectorized_images, vectorized_labels))


def main():

    training_data, test_data = net.mnist.load_mnist_data()

    vectorized_training_data = vectorize_mnist(training_data)
    vectorized_test_data = vectorize_mnist(test_data)

    image_size = training_data[0][0].size

    network = net.network.Net([image_size, 30, 10])
    network.train(
        data=vectorized_training_data, test_data=vectorized_test_data,
        epochs=20, learning_rate=0.01)

    test_accuracy = network.get_accuracy(vectorized_training_data)
    print("Test accuracy is {}".format(test_accuracy))


if __name__ == "__main__":

    main()
