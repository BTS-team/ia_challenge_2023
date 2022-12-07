from datascience.data_loading import prepare_dataloader, torch_test_set
from datascience.model import DeepLearningModel
import torch
from torch.nn import Linear, MSELoss, Module, Dropout, BatchNorm1d
from torch import flatten
from torch.nn.functional import relu, sigmoid, tanh
import warnings

warnings.filterwarnings("ignore")


class NNModel(Module):
    def __init__(self):
        super().__init__()
        self.input = Linear(11, 22)
        self.hid = Linear(22, 22)
        self.hid_2 = Linear(22, 22)
        self.output = Linear(22, 1)
        self.dropout = Dropout(0.2)

    def forward(self, x):
        x = tanh(self.input(x))
        x = tanh(self.hid(x))
        x = tanh(self.hid_2(x))
        return self.output(x)


if __name__ == '__main__':
    model = NNModel()
    nn = DeepLearningModel(model, dtype="relative")
    nn.train(epochs=16, learning_rate=0.01, batch_size=16)
    nn.save(path="./model", name="relative_3layer_tanh")
