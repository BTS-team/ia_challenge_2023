import numpy as np
from UTTnet.Optimizer import SGD
from UTTnet.Metric import Mse
from tqdm import tqdm


class Network:
    """ Neural Network Class

    This class is used to implement a multi layers neural network.

    :param loss: Loss function of the neural network
    :type loss: UTTnet.Metric.Metric.Metric
    :param optimizer: The optimizer of the neural network
    :type optimizer: UTTnet.Optimizer.Optimizer.Optimizer
    :param layers: A list containing all the layers of the network
    :type layers: List[UTTnet.Layer.Layer.Layer]
    """
    def __init__(self, loss=Mse(), optimizer=SGD()):
        """ Constructor method
        """
        self.layers = []
        self.loss = loss
        self.optimizer = optimizer

    def add(self, layer):
        """ Add a new layer to the network

        :param layer: The layer to add to the network
        :type layer: UTTnet.Layer.Layer.Layer
        """
        self.layers.append(layer)

    def __call__(self, X):
        """ Predict method

        This method is the redefinition of the __call__ method.

        :param X: The input on which to make a prediction
        :type X: numpy.ndarray
        :return: The prediction of the model for the given input
        :rtype: numpy.ndarray
        """
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)
            samples = X.shape[0]
            result = []

            for i in range(samples):
                output = X[i]
                for layer in self.layers:
                    output = layer.forward_pass(output)
                result.append(output)
            return result
        else:
            output = np.expand_dims(X, axis=0)
            for layer in self.layers:
                output = layer.forward_pass(output)
            return output

    def backward(self, error):
        """ Backward pass

        Implementation of the backward pass from the backward propagation algorithm.

        :param error: The prediction error
        :type error: numpy.ndarray
        """
        for layer in reversed(self.layers):
            error = layer.backward_pass(error)

    def fit(self, x_train, y_train, epochs):
        """ Train the network

        Implementation of the backward propagation algorithm to train the network

        :param x_train: The input values on which to train the network
        :type x_train: numpy.ndarray
        :param y_train: The train labels of the inputs values
        :type y_train: numpy.ndarray
        :param epochs: The number of iteration to train the network
        :type epochs: int
        :return: A list containing the loss occording to the epochs
        :rtype: List
        """
        samples = len(x_train)

        loss = []
        for i in tqdm(range(epochs)):
            err = 0
            for j in range(samples):
                output = self(x_train[j])
                err += self.loss(y_train[j], output)
                error = self.loss.prime(y_train[j], output)
                self.backward(error)
                self.optimizer.step(self.layers)

            loss.append([i, err / samples])
        return loss

    def __str__(self):
        res = "Neural Network(\n"
        res += "  layers = [\n"
        for i in self.layers:
            res += f"    {i}\n"
        res += "  ],\n"
        res += f"  Optimizer = {self.optimizer},\n"
        res += f"  Loss function = {self.loss}\n"
        res += ")\n"
        return res
