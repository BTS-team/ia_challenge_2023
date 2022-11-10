from datascience.data_loading import prepare_dataloader, torch_test_set
from datascience.model import DeepLearningModel
import torch
from torch.nn import Linear, MSELoss, Module, Dropout, Conv1d
from torch import flatten
from torch.nn.functional import relu, sigmoid
import warnings

warnings.filterwarnings("ignore")


class NNModel(Module):
    def __init__(self):
        super().__init__()
        self.input = Linear(108, 108)
        self.hid_1 = Linear(108, 54)
        self.hid_2 = Linear(54, 54)
        self.hid_3 = Linear(54, 54)
        self.output = Linear(54, 1)
        self.dropout = Dropout(0.2)

    def forward(self, x):
        x = sigmoid(self.input(x))
        x = self.dropout(x)
        x = sigmoid(self.hid_1(x))
        x = self.dropout(x)
        x = sigmoid(self.hid_2(x))
        x = sigmoid(self.hid_3(x))
        x = self.dropout(x)
        return self.output(x)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()


if __name__ == '__main__':
    model = NNModel()
    nn = DeepLearningModel(model)
    nn.train("test.pth", epochs=50)
    # print("training done")
    nn.submission().to_csv("nnet_submission_epoch_more_data.csv", index=False)
