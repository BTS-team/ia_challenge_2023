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
        self.input = Linear(109, 109)
        self.output = Linear(109, 1)
        self.dropout = Dropout(0.2)

    def forward(self, x):
        x = tanh(self.input(x))
        return self.output(x)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()


if __name__ == '__main__':
    model = NNModel()
    nn = DeepLearningModel(model)
    nn.train(epochs=10, learning_rate=0.001, batch_size=16)
    nn.save(path="./model", name="3_layer_tanh")
