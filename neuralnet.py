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
        self.m = BatchNorm1d(109)
        self.output = Linear(109, 1)
        self.dropout = Dropout(0.2)

    def forward(self, x):
        x = tanh(self.input(x))
        x = self.m(x)
        return self.output(x)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()


if __name__ == '__main__':
    model = NNModel()
    nn = DeepLearningModel(model)
    nn.train("test.pth", epochs=10, learning_rate=0.001, batch_size=16)
    # print("training done")
    nn.submission().to_csv("nnet_submission_order_requests.csv", index=False)
