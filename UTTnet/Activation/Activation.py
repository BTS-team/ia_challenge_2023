class Activation:
    """ Activation abstract Class

    An abstract class used to implement all activations functions such as relu, sigmoid, tanch, etc..

    :param input: The input data to pass threw the activation function
    :type input: numpy.ndarray
    :param output: The activated data
    :type output: numpy.ndarray
    """
    def __init__(self):
        """Constructor method
        """
        self.input = None
        self.output = None

    def __call__(self, input_data):
        """ Compute the activation

        :param input: The input data to pass threw the activation function
        :type input: numpy.ndarray
        """
        raise NotImplementedError

    def prime(self, x):
        """ Compute the activation prime

        :param input: The input data to pass threw the activation function
        :type input: numpy.ndarray
        """
        raise NotImplementedError

    def forward(self, x):
        """ Forward pass

        Compute the activation of input data and stored the input for the gradient descent

        :param x: The input data to pass threw the activation function
        :type x: numpy.ndarray
        :return: The activated data
        :rtype: numpy.ndarray
        """
        self.input = x
        self.output = self(x)
        return self.output

    def backward(self, dx):
        """ Compute the loss of the input

        Compute the gradient of the input in function of the output error
        :param dx: Output error
        :type dx: numpy.ndarray
        :return: The input error
        :rtype: numpy.ndarray
        """
        return self.prime(self.input) * dx
