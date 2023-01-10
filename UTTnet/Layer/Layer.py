class Layer:
    def __init__(self, activation=None):
        self.activation = activation

    def forward_pass(self, X):
        if self.activation is not None:
            return self.activation.forward(X)
        return X

    def backward_pass(self, output_error):
        if self.activation is not None:
            return self.activation.backward(output_error)
        return output_error

