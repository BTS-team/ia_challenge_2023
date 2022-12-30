from datascience.model import DeepLearningModel
from torch.nn import Linear, Module, Dropout
from torch.optim import RMSprop
from torch.nn.functional import tanh
import warnings

warnings.filterwarnings("ignore")


class NNModel(Module):
    def __init__(self):
        super().__init__()
        self.input = Linear(109, 109)
        self.hid = Linear(109, 109)
        self.output = Linear(109, 1)
        self.dropout = Dropout(0.2)

    def forward(self, x):
        x = tanh(self.input(x))
        x = self.dropout(x)
        x = tanh(self.hid(x))
        x = self.dropout(x)
        return self.output(x)


if __name__ == '__main__':
    model = NNModel()
    nn = DeepLearningModel(model)
    nn.train(optimizer=RMSprop, epochs=15, learning_rate=0.001, batch_size=64, show=True)
    nn.save(path="./model", name="relative_4layer_tanh_dropout_rms")
