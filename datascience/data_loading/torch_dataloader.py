import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datascience.data_loading import load_dataset, load_test_set
from torch.nn.functional import one_hot
from datascience.utils import city, language, brand, group
import pandas as pd
from sklearn import preprocessing

pd.option_context('display.max_columns', None)


def add_one_hot_columns(result, column, label):
    if isinstance(label, dict):
        length = len(label.keys())
    else:
        length = label[1]
        name = label[0]
    onehot = one_hot(column, num_classes=length).numpy()

    if isinstance(label, dict):
        for i, key in enumerate(label.keys()):
            result[key] = onehot[:, i]
    else:
        for i in range(length):
            key = f"{name}_{i}"
            result[key] = onehot[:, i]


def one_hot_encoding(x):
    pd.set_option('display.max_columns', len(list(x)))
    result = x[[
        'mobile',
        'parking',
        'pool',
        'stock',
        'order_requests'
    ]]
    city_test = x['city'].to_numpy()
    city_df = torch.from_numpy(x['city'].to_numpy())
    language_df = torch.from_numpy(x['language'].to_numpy())
    brand_df = torch.from_numpy(x['brand'].to_numpy())
    group_df = torch.from_numpy(x['group'].to_numpy())
    children_df = torch.from_numpy(x['children_policy'].to_numpy())
    date_df = torch.from_numpy(x['date'].to_numpy())
    add_one_hot_columns(result, city_df, city)
    add_one_hot_columns(result, language_df, language)
    add_one_hot_columns(result, brand_df, brand)
    add_one_hot_columns(result, group_df, group)
    add_one_hot_columns(result, children_df, ('children_policy', 3))
    add_one_hot_columns(result, date_df, ('date', 45))
    return result


def relative_column(x):
    result = x[[
        'stock',
        'order_requests'
    ]]

    temp = x[[
        'mobile',
        'parking',
        'pool',
        'stock',
        'city',
        'language',
        'group',
        'children_policy',
        'date'
    ]]

    temp = temp / temp.max()

    return pd.concat([result, temp], axis=1)


def to_matrix(x):
    x = x.to_numpy()
    dataset = []
    for row in x:
        temp = np.reshape(row, (12, 9))
        dataset.append(temp)
    dataset = np.array(dataset)
    return dataset


def torch_test_set(test_set='meta_data/test_set.csv', features_hotels='meta_data/features_hotels.csv', dtype='onehot'):
    index, test_set = load_test_set(test_set, features_hotels)

    if dtype == 'onehot':
        test_set = one_hot_encoding(test_set)
        return index, test_set
    elif dtype == 'matrix':
        test_set = one_hot_encoding(test_set)
        return index, to_matrix(test_set)
    elif dtype == 'relative':
        return index, relative_column(test_set)
    else:
        raise f"{dtype} => Invalid type for loading test set"


class Data(Dataset):
    def __init__(self, dataset_path, features_hotels, dtype="onehot"):
        dataset = load_dataset(dataset_path, features_hotels, dtype="pandas")

        if dtype == "matrix":
            x = one_hot_encoding(dataset.x)
            x = to_matrix(x)
            self.X = torch.from_numpy(x.astype(np.float32))
        elif dtype == "onehot":
            x = one_hot_encoding(dataset.x)
            self.X = torch.from_numpy(x.to_numpy().astype(np.float32))
        elif dtype == "relative":
            x = relative_column(dataset.x)
            self.X = torch.from_numpy(x.to_numpy().astype(np.float32))
        else:
            raise f"{dtype} => Invalid type for loading test set"

        self.y = torch.from_numpy(dataset.y.to_numpy().astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


def prepare_dataloader(dataset_path, features_hotels, dist=[0.8, 0.2], batch_size=64, dtype="onehot"):
    dataset = Data(dataset_path, features_hotels, dtype)
    rep = list(map(lambda x: int(x * dataset.__len__()), dist))
    rep[-1] += dataset.__len__() - sum(rep)

    split = torch.utils.data.random_split(dataset, rep)
    data = []

    for i in split:
        data.append(DataLoader(dataset=i, batch_size=batch_size, shuffle=True, drop_last=True))

    return tuple(data)


if __name__ == '__main__':
    dataset = Data('../../dataset', "../../meta_data/features_hotels.csv", divideByMax=True)

    # loader = prepare_dataloader('../../dataset', "../meta_data/features_hotels.csv", batch_size=1)
    """ for batch, (X, y) in enumerate(loader[0]):
        print(f"Batch: {batch + 1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    """
