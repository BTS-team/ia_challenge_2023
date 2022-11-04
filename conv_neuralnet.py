import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
from datascience.model import MLModel
from datascience.data_loading import prepare_dataloader
from datascience.data_loading import torch_test_set
import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, Module, Conv1d
from torch.nn.functional import relu, sigmoid
from torch.optim import Adam
import warnings
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class NNModel(Module):
    def __init__(self):
        super().__init__()
        self.input = Conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=True)
        print(self.input)
        #self.hidden = Conv1d(in_channels=2, out_channels=2, kernel_size=5, bias=True)
        #self.hidden_2 = Conv1d(in_channels=2, out_channels=1, kernel_size=5, bias=True)
        self.hidden_3 = Linear(60, 52)
        self.output = Linear(52, 1)

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(0)
        print(x.shape)
        x = sigmoid(self.input(x))
        #x = sigmoid(self.hidden(x))
        #x = sigmoid(self.hidden_2(x))
        x = sigmoid(self.hidden_3(x))
        return self.output(x)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()


class DeepLearningModel:
    def __init__(self, dataset='dataset/', features_hotels='meta_data/features_hotels.csv'):
        self.dataset = prepare_dataloader(dataset, features_hotels)
        self.model = NNModel()
        print(self.model)
        self.features_hotels = features_hotels

    def train(self, optimizer=Adam, loss_fn=MSELoss(), epochs=150, learning_rate=0.01, show=False):
        optimizer = optimizer(self.model.parameters(), learning_rate)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        loss_values = []
        val_loss_values = []
        for epoch in range(epochs):
            epoch_train_loss = 0.0

            # Training Loop
            for X_train, y_train in self.dataset[0]:
                self.model.zero_grad()
                prediction = self.model(X_train)
                loss = loss_fn(prediction, y_train)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # Calculate training loss value
            train_loss_value = epoch_train_loss / len(self.dataset[0])
            val_loss_value, rmse = self.validate(loss_fn)
            loss_values.append(train_loss_value)
            val_loss_values.append(val_loss_value)

            if epochs % 10 == 9:
                learning_rate /= 10
            print(
                f"Epoch {epoch} - Training Loss : {train_loss_value} - Validation loss : {val_loss_value} - RMSE : {rmse.round(3)}")

        torch.save(self.model, "backup/19_56713/nn_model.pth")

        if show:
            x = list(range(1, epochs + 1))
            plt.plot(x, loss_values, color='b', label='train')
            plt.plot(x, val_loss_values, color='r', label='validation')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()

    def predict(self, x):
        row = Tensor([x])
        prediction = self.model(row)
        prediction = prediction.detach().numpy()
        return prediction[0][0]

    def validate(self, loss_fn):
        y_predicted = []
        y_actual = []
        vall_loss = 0
        with torch.no_grad():
            self.model.eval()
            for X_val, y_val in self.dataset[1]:
                prediction = self.model(X_val)[0][0]
                val_loss = loss_fn(prediction, y_val)
                vall_loss += val_loss.item()
                y_predicted.extend(prediction.tolist())
                y_actual.extend(y_val.tolist())
        val_loss_value = vall_loss / len(self.dataset[1])
        rmse = mean_squared_error(y_actual, y_predicted, squared=False)

        return val_loss_value, rmse

    def submission(self, test_set='meta_data/test_set.csv'):
        index, x = torch_test_set(test_set, self.features_hotels)
        submission_df = []
        index = index.to_numpy()
        x = x.to_numpy()
        for i in range(len(x)):
            prediction = self.predict(x[i])
            submission_df.append([index[i], prediction])

        submission_df = pd.DataFrame(submission_df, columns=['index', 'price'])
        return submission_df


if __name__ == '__main__':
    nn = DeepLearningModel()
    nn.train(epochs=15)
    #print("training done")
    #nn.submission().to_csv("nnet_submission_epoch_30_new.csv", index=False)

