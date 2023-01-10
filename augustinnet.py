from UTTnet import Network
from UTTnet.Activation import Tanh, Relu
from UTTnet.Metric import Mse
from UTTnet.Layer import Dense
from UTTnet.Optimizer import SGD
from datascience.data_loading import one_hot_encoding
from datascience.data_loading import load_dataset

dataset = load_dataset('./dataset', './meta_data/features_hotels.csv', dtype="pandas")
dataset.x = one_hot_encoding(dataset.x)
dataset.to_numpy()

# %%

model_2 = Network(
    loss=Mse(),
    optimizer=SGD()
)

model_2.add(Dense(109, 109, activation=Tanh()))
model_2.add(Dense(109, 1))

loss = model_2.fit(dataset.x[:1500], dataset.y[:1500], epochs=5)
# test the model on
out = model_2(dataset.x[2000])
print(f"x = {dataset.x[2000]}")
print(f"prediction = {out}")
print(f"true = {dataset.y[2000]}")
