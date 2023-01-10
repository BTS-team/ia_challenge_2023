from UTTnet.Activation.Activation import Activation
import numpy as np


class Tanh(Activation):
    """ Hyperbolic tangent activation function

    See https://en.wikipedia.org/wiki/Hyperbolic_functions for more details.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """ Tanh function

        Calculate the hyperbolic tangent of the input

        :param x: The input to calculate the hyperbolic tangent
        :type x: numpy.ndarray
        :return: The hyperbolic tangent of the input
        :rtype: numpy.ndarray
        """
        return np.tanh(x)

    def prime(self, x):
        """ Tanh prime

        Compute the tanh prime of a given input

        :param x: The input data to pass threw the tanh prime function
        :type x: numpy.ndarray
        :return: The output data passed threw the tanh prime
        :rtype: numpy.ndarray
        """
        return 1 - np.tanh(x) ** 2

    def __str__(self):
        return "Tanh"


if __name__ == '__main__':
    tanh = Tanh()
    print(tanh(np.array([10, 8, 9, 7])))
    print(tanh.prime(np.array([10, 8, 9, 7])))
