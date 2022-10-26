import torch
from torch import Tensor
from torch.nn import Linear, ModuleList, MSELoss, Module
from torch.nn.functional import relu, sigmoid, elu, silu
from torch.optim import SGD, Adam
import warnings
from data import apply
from dataset import prepare_dataloader
from sklearn.metrics import mean_squared_error
import pandas as pd

warnings.filterwarnings("ignore")


class Model(Module):
    def __init__(self, layer=[10, 5, 1]):
        super().__init__()
        # self.layers = ModuleList()
        # for i in range(1, len(layer)):
        #    self.layers.append(Linear(layer[i - 1], layer[i]))

        self.layer_1 = Linear(10, 10)
        self.layer_2 = Linear(10, 1)

    def forward(self, x):
        # for i in range(len(self.layers) - 1):
        #    x = relu(self.layers[i](x))
        # x = self.layers[-1](x)
        x = sigmoid(self.layer_1(x))
        x = self.layer_2(x)
        return x


def saveModel(model, path):
    torch.save(model.state_dict(), path)


def train(model, train_data, validation_data, learning_rate=0.001, loss_function=MSELoss(), optimizer=Adam, epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optimizer(model.parameters(), learning_rate)
    loss_values = []
    accuracy_values = []

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_accuracy = 0.0
        epoch_vall_loss = 0.0
        total = 0

        # Training Loop
        for X_train, y_train in train_data:
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict output from the model
            prediction = model(X_train)
            # calculate loss for the predicted output
            loss = loss_function(prediction, y_train)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
            # update the loss value
            epoch_train_loss += loss.item()

        # Calculate training loss value
        train_loss_value = epoch_train_loss / len(train_data)

        loss_values.append(train_loss_value)

        y_predicted = []
        y_actual = []
        # Validation Loop
        with torch.no_grad():
            model.eval()
            for X_val, y_val in validation_data:
                prediction = model(X_val)
                val_loss = loss_function(prediction, y_val)
                epoch_vall_loss += val_loss.item()
                total += y_val.size(0)
                print(y_val)
                print(prediction)
                y_predicted.extend(prediction.tolist())
                y_actual.extend(y_val.tolist())

        # Calculate validation loss value
        val_loss_value = epoch_vall_loss / len(validation_data)

        # Calculate the accuracy
        # print(len(y_actual))
        # print(len(y_predicted))
        accuracy = mean_squared_error(y_actual, y_predicted, squared=False)
        accuracy_values.append(accuracy)

        # Saving the model
        if accuracy > max(accuracy_values):
            saveModel('')
        print(
            f"Epoch {epoch} - Training Loss : {train_loss_value} - Validation loss : {val_loss_value} - RMSE : {(accuracy).round(3)}")

    return model, loss_values, accuracy_values


def test_model(model, test_data):
    y_predicted = []
    y_actual = []
    total = 0
    with torch.no_grad():
        for X_test, y_test in test_data:
            y_test = y_test.to(torch.float32)
            prediction = model(X_test)
            total += y_test.size(0)
            y_predicted.extend(prediction.tolist())
            y_actual.extend(y_test.tolist())

        print(
            f"RMSE of the model based on the test set of {len(X_test)} inputs is {(mean_squared_error(y_actual, y_predicted, squared=False)).round(3)}")


def predict(row, model):
    row = list(map(lambda x: apply(x), row))
    print(row)
    row = Tensor([row])
    prediction = model(row)
    prediction = prediction.detach().numpy()
    return prediction


def submission():
    dataloader = prepare_dataloader('../dataset', '../meta_data/features_hotels.csv', batch_size=254)
    model = Model()
    metrics = train(model, dataloader[0], dataloader[2])

    model = metrics[0]

    to_predict = pd.read_csv('../data/test_set.csv')
    hotels = pd.read_csv('../meta_data/features_hotels.csv', index_col=['hotel_id', 'city'])
    to_predict = to_predict.join(hotels, on=['hotel_id', 'city'])

    submission_df = []

    for i in to_predict.to_dict('index').values():
        index = i['index']
        X = [
            i['city'],
            i['date'],
            i['language'],
            i['mobile'],
            i['stock'],
            i['group'],
            i['brand'],
            i['parking'],
            i['pool'],
            i['children_policy']
        ]
        X = list(map(lambda x: apply(x), X))
        X = Tensor([X])
        prediction = model(X)
        prediction = prediction.detach().numpy()

        submission_df.append([index, prediction[0][0]])

    submission_df = pd.DataFrame(submission_df, columns=['index', 'price'])
    submission_df.to_csv('../sample_submission.csv',index=False)


if __name__ == '__main__':
    submission()
