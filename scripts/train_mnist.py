"""
File for training network on mnist data.
This acts as a sanity check that our network code really can learn and predict.
"""
import net.mnist

def main():

    training_data, test_data = net.mnist.load_mnist_data()


if __name__ == "__main__":
    main()
