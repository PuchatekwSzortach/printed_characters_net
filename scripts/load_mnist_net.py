"""
A simple file that essentially checks if previously trained mnist network can be
successfully retrieved from a file and used for predictions.
"""
import net.mnist
import net.network


def main():

    _, test_data = net.mnist.load_mnist_data()
    vectorized_test_data = net.mnist.vectorize_mnist(test_data)

    network = net.network.Net.from_file("./results/mnist_net.json")

    test_accuracy = network.get_accuracy(vectorized_test_data)
    print("Test accuracy is {}".format(test_accuracy))

if __name__ == "__main__":

    main()