import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datascience.model import MLModel


def initialisation(dimensions):
    parametres = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres


def forward_propagation(X, parametres):
    activations = {'A0': X}

    # C est le nombre de couches si on a 4 paramètres cela signifie qu'on a deux couches
    C = len(parametres) // 2

    for c in range(1, C):
        Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
        Z = Z.astype(np.float64)
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    # print(activations)
    Z = parametres['W' + str(C)].dot(activations['A' + str(C - 1)]) + parametres['b' + str(C)]
    Z = Z.astype(np.float64)
    activations['A' + str(C)] = Z
    return activations


def back_propagation(y, parametres, activations):
    m = y.shape[1]
    C = len(parametres) // 2

    dZ = activations['A' + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (
                    1 - activations['A' + str(c - 1)])

    return gradients


def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2

    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    return parametres


def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2
    Af = activations['A' + str(C)]
    return Af


def prep_data(X, y):
    centrereduit = StandardScaler()
    X_cr = centrereduit.fit_transform(X)
    X_train_transpose = X_cr.T
    Y_train_transpose = y.T
    return X_train_transpose, Y_train_transpose


def deep_neural_network(X, y, hidden_layers=(16, 16, 16), learning_rate=0.001, n_iter=3000):
    X, y = prep_data(X, y)
    # initialisation parametres
    dimensions = list(hidden_layers)
    # On insert les nombres de coordonnées pour un individu
    dimensions.insert(0, X.shape[0])
    # On donne une sortie
    dimensions.append(y.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_iter), 2))

    C = len(parametres) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parametres)

        gradients = back_propagation(y, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]
        print(Af)
        # calcul du log_loss et de l'accuracy
        # print(y.flatten().astype(np.int32).shape)
        # print(Af.flatten().shape)
        training_history[i, 0] = (mean_squared_error(y.flatten().astype(np.int32), Af.flatten(), squared=False))
        y_pred = predict(X, parametres)
        print(y_pred)
        # print(y_pred.flatten().shape)
        # print(y.flatten().astype(np.int32).shape)
        # training_history[i, 1] = (accuracy_score(y.flatten().astype(np.int32), y_pred.flatten()))

    y_pred = predict(X, parametres)
    print(y_pred)
    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    # plt.plot(training_history[:, 1], label='train acc')
    # plt.legend()
    plt.show()

    return parametres


"""
X, y = make_blobs(n_samples=2835, n_features=8, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.show()

centrereduit = StandardScaler()
X_cr = centrereduit.fit_transform(X)
Col_True = np.random.choice(["TRUE", "FALSE"], size=X_cr.shape[0], replace=True, p=[0.7, 0.3])
indice_true = np.where(Col_True == "TRUE")
indice_false = np.where(Col_True == "FALSE")
X_train = X_cr[indice_true]

Y_train = y[indice_true]

X_test = X_cr[indice_false]

Y_test = y[indice_false]

X_train_transpose = X_train.T
X_test_transpose = X_test.T
Y_train_transpose = Y_train.T
Y_test_transpose = Y_test.T

parametres = deep_neural_network(X_train_transpose, Y_train_transpose, hidden_layers=(16, 16, 16), learning_rate=0.01,
                                 n_iter=5000)
y_pred_test = predict(X_test_transpose, parametres)
accuracy_score(Y_test_transpose.flatten(), y_pred_test.flatten())
"""


class AmbreNet(MLModel):
    def __init__(self, dataset='./dataset', features_hotels='./meta_data/features_hotels.csv'):
        super().__init__(dataset=dataset, features_hotels=features_hotels)
        self.parametres = None

    def train(self, hidden_layers=(16, 16), learning_rate=0.00001):
        self.parametres = deep_neural_network(self.dataset.x, self.dataset.y, hidden_layers=hidden_layers,
                                              learning_rate=learning_rate,
                                              n_iter=10)

    def predict(self, x):
        return predict(x, self.parametres)


if __name__ == '__main__':

    #model = AmbreNet()
    #model.train()
