from UTTnet.Activation.Activation import Activation
import numpy as np


class Sigmoid(Activation):
    """ Sigmoid activation function

    See https://en.wikipedia.org/wiki/Sigmoid_function for more details.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """ Sigmoid computation

        Compute the sigmoid of a given input

        :param x: The input data to pass threw the sigmoid function
        :type x: numpy.ndarray
        :return: The output data passed threw the sigmoid
        :rtype: numpy.ndarray
        """
        return 1 / (1 + np.exp(-x))

    def prime(self, x):
        """ Sigmoid prime

        Compute the sigmoid prime of a given input

        :param x: The input data to pass threw the sigmoid prime function
        :type x: numpy.ndarray
        :return: The output data passed threw the sigmoid prime
        :rtype: numpy.ndarray
        """
        sig = self(x)
        return sig * (1 - sig)

    def __str__(self):
        return "Sigmoid"


if __name__ == '__main__':
    sig = Sigmoid()
    print(sig(np.array([10, 8, 9, 7])))
    print(sig.prime(np.array([10, 8, 9, 7])))
