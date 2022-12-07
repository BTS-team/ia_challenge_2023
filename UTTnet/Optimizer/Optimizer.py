

class Optimizer:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def step(self, layer):
        raise NotImplementedError
