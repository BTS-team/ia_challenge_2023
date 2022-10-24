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
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_vall_loss = 0.0
        total = 0

        # Training Loop
        for X_train, y_train in train_data:
            optimizer.zero_grad()  # zero the parameter gradients
            prediction = model(X_train)  # predict output from the model
            loss = loss_function(prediction, y_train)  # calculate loss for the predicted output
            loss.backward()  # backpropagate the loss
            optimizer.step()  # adjust parameters based on the calculated gradients
            running_train_loss += loss.item()  # track the loss value

        # Calculate training loss value
        train_loss_value = running_train_loss / len(train_data)
        loss_values.append(train_loss_value)
        # Validation Loop
        with torch.no_grad():
            model.eval()
            for X_val, y_val in validation_data:
                prediction = model(X_val)
                val_loss = loss_function(prediction, y_val)
                running_vall_loss += val_loss.item()
                total += y_val.size(0)
                running_accuracy += (prediction == y_val).sum().item()

                # Calculate validation loss value
        val_loss_value = running_vall_loss / len(validation_data)
        accuracy = (100 * running_accuracy / total)
        accuracy_values.append(accuracy)

        # Saving the model
        if accuracy > max(accuracy_values):
            saveModel()
        print(f"Epoch {epoch} - Training Loss : {train_loss_value} - Validation loss : {val_loss_value} - accuracy : {accuracy}%")


def test_model(model, test_data):
    running_accuracy = 0
    total = 0
    with torch.no_grad():
        for X_test, y_test in test_data:
            y_test = y_test.to(torch.float32)
            prediction = model(X_test)
            total += y_test.size(0)
            running_accuracy += (prediction == y_test).sum().item()

        print(
            f"Accuracy of the model based on the test set of {len(X_test)} inputs is {(100 * running_accuracy / total)}%")


def predict(row, model):
    row = map(lambda x: apply(x), row)
    row = Tensor([row])
    prediction = model(row)
    prediction = prediction.detach().numpy()
    return prediction


if __name__ == '__main__':
    model = Model([10, 5, 3, 1])
    print(model)
