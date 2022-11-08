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
        self.input = Conv1d(in_channels=12, out_channels=10, kernel_size=3, bias=True)
        #self.hidden_conv = Conv1d(in_channels=10, out_channels=9, kernel_size=2, bias=True)
        #self.hidden_conv_2 = Conv1d(in_channels=9, out_channels=7, kernel_size=3, bias=True)
        self.linear_1 = Linear(70, 140)
        self.output = Linear(140, 1)
        self.dropout = Dropout(0.25)

    def forward(self, x):
        #print(x.shape)
        x = relu(self.input(x))
        # print(x.shape)
        x = self.dropout(x)
        # x = sigmoid(self.hidden_conv(x))
        #print(x.shape)
        # x = self.dropout(x)
        # x = sigmoid(self.hidden_conv_2(x))
        # x = self.dropout(x)
        #print(x.shape)
        x = flatten(x, start_dim=1)
        x = sigmoid(self.linear_1(x))
        return self.output(x)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()


class ConvModel(DeepLearningModel):
    def __init__(self, model, dataset='dataset/', features_hotels='meta_data/features_hotels.csv'):
        self.dataset = prepare_dataloader(dataset, features_hotels, matrix=True)
        self.model = model
        self.features_hotels = features_hotels

    def load_test_set(self, path):
        index, x = torch_test_set(path, self.features_hotels, matrix=True)
        return index, x


if __name__ == '__main__':
    model = NNModel()
    print(model.input.weight)
    nn = ConvModel(model)
    nn.train("test.pth", epochs=5, show=True)
    print(model.input.weight)
    # print("training done")
    #nn.submission().to_csv("nnet_submission_epoch_conv.csv", index=False)
