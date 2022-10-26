import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from data import apply
from utils import get_folder
from sklearn import preprocessing


class Data(Dataset):
    def __init__(self, dataset_path, features_hotels):
        city_folder = get_folder(dataset_path)
        rows = None
        for i in city_folder:
            language_file = get_folder(f"{dataset_path}/{i}")
            for j in language_file:
                temp = pd.read_csv(f"{dataset_path}/{i}/{j}")
                if rows is None:
                    rows = temp.to_numpy()
                else:
                    rows = np.concatenate((rows, temp.to_numpy()))

        np.random.shuffle(rows)
        rows = pd.DataFrame(rows,
                            columns=['hotel_id', 'price', 'stock', 'city', 'date', 'language', 'mobile', 'avatar_id'])
        hotels = pd.read_csv(features_hotels, index_col=['hotel_id', 'city'])
        pricing_requests = rows.join(hotels, on=['hotel_id', 'city'])
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
        #print(len(y_data_set))

        x_data_set = x_data_set.applymap(apply).to_numpy()

        #standard = preprocessing.scale(x_data_set)
        self.X = torch.from_numpy(x_data_set.astype(np.float32))
        self.y = torch.from_numpy(y_data_set.astype(np.float32))
        #print(self.y)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


def prepare_dataloader(dataset_path, features_hotels, dist=[0.8, 0.2, 0], batch_size=64):
    dataset = Data(dataset_path, features_hotels)
    rep = list(map(lambda x: int(x * dataset.__len__()), dist))
    rep[-1] += dataset.__len__() - sum(rep)
    train, valid,test = torch.utils.data.random_split(dataset, rep)
    train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader, validation_dataloader


if __name__ == '__main__':

    loader = prepare_dataloader('../dataset', "../meta_data/features_hotels.csv", batch_size=1)
    """ for batch, (X, y) in enumerate(loader[0]):
        print(f"Batch: {batch + 1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    """