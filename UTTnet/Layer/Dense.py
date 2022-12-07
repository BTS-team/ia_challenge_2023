from UTTnet.Layer.Layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, input_size, output_size, activation=None, bias=True):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = bias
        self.dInputs = None
        self.dWeights = None
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        self.output = None
        if self.bias:
            self.biases = np.random.rand(1, output_size) - 0.5
            self.dBiases = None

        super().__init__(activation)

    def forward_pass(self, X):
        self.input = X
        if self.bias:
            self.output = np.dot(self.input, self.weights) + self.biases
        else:
            self.output = np.expand_dims(np.dot(self.input, self.weights), axis=0)
        return super().forward_pass(self.output)

    def backward_pass(self, output_error):
        output_error = super().backward_pass(output_error)
        self.dInputs = np.dot(output_error, self.weights.T)
        self.dWeights = np.dot(self.input.T, output_error)

        if self.bias:
            self.dBiases = np.sum(output_error, axis=0, keepdims=True)
        return self.dInputs

    def __str__(self):
        return f"Dense(input={self.input_size},output={self.output_size},bias={self.bias},activation={self.activation})"
