import torch
from torch import Tensor
from torch.nn import Linear, ModuleList, MSELoss, Module
from torch.nn.functional import relu
from torch.optim import SGD
import warnings
from data import apply

warnings.filterwarnings("ignore")


class Model(Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = ModuleList()
        for i in range(1, len(layer)):
            self.layers.append(Linear(layer[i - 1], layer[i]))

    def forward(self, x):
        for i in self.layers:
            x = relu(i(x))
        return x


def saveModel(model, path):
    torch.save(model.state_dict(), path)


def train(model, train_data, validation_data, learning_rate=0.1, loss_function=MSELoss(), optimizer=SGD, epochs=100):
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

        # Validation Loop
        with torch.no_grad():
            model.eval()
            for X_val, y_val in validation_data:
                prediction = model(X_val)
                val_loss = loss_function(prediction, y_val)
                epoch_vall_loss += val_loss.item()
                total += y_val.size(0)
                epoch_accuracy += (prediction == y_val).sum().item()

        # Calculate validation loss value
        val_loss_value = epoch_vall_loss / len(validation_data)

        # Calculate the accuracy
        accuracy = (100 * epoch_accuracy / total)
        accuracy_values.append(accuracy)

        # Saving the model
        if accuracy > max(accuracy_values):
            saveModel()
        print(
            f"Epoch {epoch} - Training Loss : {train_loss_value} - Validation loss : {val_loss_value} - accuracy : {accuracy}%")

    return loss_values, accuracy_values


def test_model(model, test_data):
    test_accuracy = 0
    total = 0
    with torch.no_grad():
        for X_test, y_test in test_data:
            y_test = y_test.to(torch.float32)
            prediction = model(X_test)
            total += y_test.size(0)
            test_accuracy += (prediction == y_test).sum().item()

        print(
            f"Accuracy of the model based on the test set of {len(X_test)} inputs is {(100 * test_accuracy / total)}%")


def display_metrics(loss, accuracy):
    pass


def predict(row, model):
    row = list(map(lambda x: apply(x), row))
    row = Tensor([row])
    prediction = model(row)
    prediction = prediction.detach().numpy()
    return prediction


if __name__ == '__main__':
    model = Model([10, 5, 3, 1])
    print(model)
