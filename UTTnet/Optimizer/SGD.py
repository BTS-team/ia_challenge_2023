from UTTnet.Optimizer.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        self.learning_rate = learning_rate

    def step(self, layer):
        for i in layer:
            print(i.dWeights)
            i.weights -= self.learning_rate * i.dWeights
            i.biases -= self.learning_rate * i.dBiases

    def __str__(self):
        return f"SGB(lr={self.learning_rate})"
