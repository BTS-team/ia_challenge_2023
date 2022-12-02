import os
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib
from datascience.model import MLModel
from datascience.data_loading import prepare_dataloader
from datascience.data_loading import torch_test_set
from datascience.utils import get_folder, ModelAlreadyExist
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Adam
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class DeepLearningModel(MLModel):
    def __init__(self, model, dataset='dataset/', features_hotels='meta_data/features_hotels.csv'):
        self.dataset = prepare_dataloader(dataset, features_hotels)
        self.model = model
        self.features_hotels = features_hotels

    def train(self, optimizer=Adam, loss_fn=MSELoss(), epochs=150, learning_rate=0.01, show=False, batch_size=64):
        optimizer = optimizer(self.model.parameters(), learning_rate, weight_decay=0.01)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
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
                prediction = self.model(X_val)
                val_loss = loss_fn(prediction, y_val)
                vall_loss += val_loss.item()
                y_predicted.extend(prediction.tolist())
                y_actual.extend(y_val.tolist())

        val_loss_value = vall_loss / len(self.dataset[1])
        rmse = mean_squared_error(y_actual, y_predicted, squared=False)

        return val_loss_value, rmse

    def load_test_set(self, path):
        index, x = torch_test_set(path, self.features_hotels)
        index = index.to_numpy()
        x = x.to_numpy()
        return index, x

    def submission(self, test_set='meta_data/test_set.csv'):
        index, x = self.load_test_set(test_set)
        submission_df = []
        for i in range(len(x)):
            prediction = self.predict(x[i])
            submission_df.append([index[i], prediction])

        submission_df = pd.DataFrame(submission_df, columns=['index', 'price'])
        return submission_df

    def save(self, path, name):
        model = get_folder(path)
        if name not in model:
            os.mkdir(f"{path}/{name}")

        attempt = get_folder(f"{path}/{name}")
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H_%M")
        if dt_string not in attempt:
            os.mkdir(f"{path}/{name}/{dt_string}")
            torch.save(self.model, f"{path}/{name}/{dt_string}/model.pth")
            f = open(f"{path}/{name}/{dt_string}/architecture.txt", "w")
            f.write(self.model.__str__())
            f.close()
            self.submission().to_csv(f"{path}/{name}/{dt_string}/submission.csv", index=False)
        else:
            raise ModelAlreadyExist(name, dt_string)


if __name__ == '__main__':
    pass
