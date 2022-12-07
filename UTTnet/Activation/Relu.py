from UTTnet.Activation.Activation import Activation
import numpy as np


class Relu(Activation):
    """ Rectified linear unit

    Implementation of the Relu activation function.
    See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) for more details.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """ Relu computation

        Compute the relu of a given input

        :param x: The input data to pass threw the relu
        :type x: numpy.ndarray
        :return: The output data passed threw the relu
        :rtype: numpy.ndarray
        """
        return np.maximum(0, x)

    def prime(self, x):
        """ Compute the relu prime

        Compute the relu prime of a given input

        :param x: The input data to pass threw the activation function
        :type x: numpy.ndarray
        :return: The output data passed threw the relu prime
        :rtype: numpy.ndarray
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def __str__(self):
        return "Relu"


if __name__ == '__main__':
    relu = Relu()
    print(relu(np.array([10, -10, 9, -7])))
    print(relu.prime(np.array([10, -8, -9, 7])))
