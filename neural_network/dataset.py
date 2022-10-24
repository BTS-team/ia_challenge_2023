import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from data import apply


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


def prepare_dataloader(stored_requests, features_hotels, dist=[0.6, 0.3, 0.1], batch_size=64):
    stored_r = pd.read_csv(stored_requests)
    hotels = pd.read_csv(features_hotels, index_col=['hotel_id', 'city'])
    pricing_requests = stored_r.join(hotels, on=['hotel_id', 'city'])
    y_data_set = pricing_requests['price'].to_numpy()
    x_data_set = pricing_requests[[
        'city',
        'date',
        'language',
        'mobile',
        'stock',
        'group',
        'brand',
        'parking',
        'pool',
        'children_policy'
    ]]
    x_data_set = x_data_set.applymap(apply).to_numpy()

    dist = list(map(lambda x: int(x * len(x_data_set)), dist))
    for i in range(1,len(dist)):
        dist[i] += dist[i-1]

    X_train, y_train = x_data_set[:dist[0], :], y_data_set[:dist[0]]
    X_test, y_test = x_data_set[dist[0]:dist[1], :], y_data_set[dist[0]:dist[1]]
    X_validation, y_validation = x_data_set[dist[1]:, :], y_data_set[dist[1]:]

    train_data, test_data, validation_data = Data(X_train, y_train), Data(X_test, y_test), Data(X_validation,
                                                                                                y_validation)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader, validation_dataloader


if __name__ == '__main__':
    loader = prepare_dataloader("../data/test.csv", "../data/features_hotels.csv",batch_size=16)
    for batch, (X, y) in enumerate(loader[0]):
        print(f"Batch: {batch + 1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
