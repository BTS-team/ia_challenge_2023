import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datascience.data_loading import load_dataset,load_test_set
from torch.nn.functional import one_hot
from datascience.utils import city, language, brand, group
import pandas as pd
from sklearn import preprocessing

pd.option_context('display.max_columns', None)


def add_one_hot_columns(result, column, label):
    onehot = one_hot(column, num_classes=len(label.keys())).numpy()
    # result += pd.DataFrame(onehot.numpy(), columns=label.keys())
    for i, key in enumerate(label.keys()):
        result[key] = onehot[:, i]


def one_hot_encoding(x):
    pd.set_option('display.max_columns', len(list(x)))
    result = x[[
        'date',
        'mobile',
        'parking',
        'children_policy',
        'pool',
        'stock'
    ]]
    city_df = torch.from_numpy(x['city'].to_numpy())
    language_df = torch.from_numpy(x['language'].to_numpy())
    brand_df = torch.from_numpy(x['brand'].to_numpy())
    group_df = torch.from_numpy(x['group'].to_numpy())
    add_one_hot_columns(result, city_df, city)
    add_one_hot_columns(result, language_df, language)
    add_one_hot_columns(result, brand_df, brand)
    add_one_hot_columns(result, group_df, group)
    return result


def torch_test_set(test_set='meta_data/test_set.csv', features_hotels='meta_data/features_hotels.csv'):
    index, test_set = load_test_set(test_set, features_hotels)
    return index, one_hot_encoding(test_set)


class Data(Dataset):
    def __init__(self, dataset_path, features_hotels):
        dataset = load_dataset(dataset_path, features_hotels, dtype="pandas")
        x = one_hot_encoding(dataset.x)
        self.X = torch.from_numpy(x.to_numpy().astype(np.float32))
        self.y = torch.from_numpy(dataset.y.to_numpy().astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


def prepare_dataloader(dataset_path, features_hotels, dist=[0.9, 0.08, 0.02], batch_size=64):
    dataset = Data(dataset_path, features_hotels)
    rep = list(map(lambda x: int(x * dataset.__len__()), dist))
    rep[-1] += dataset.__len__() - sum(rep)
    train, valid, test = torch.utils.data.random_split(dataset, rep)
    train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader, validation_dataloader


if __name__ == '__main__':
    dataset = Data('../../dataset', "../../meta_data/features_hotels.csv")
    # loader = prepare_dataloader('../../dataset', "../meta_data/features_hotels.csv", batch_size=1)
    """ for batch, (X, y) in enumerate(loader[0]):
        print(f"Batch: {batch + 1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    """
