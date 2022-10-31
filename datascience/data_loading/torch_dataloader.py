import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datascience.data_loading import load_dataset


class Data(Dataset):
    def __init__(self, dataset_path, features_hotels):
        x_data_set, y_data_set = load_dataset(dataset_path, features_hotels)
        self.X = torch.from_numpy(x_data_set.astype(np.float32))
        self.y = torch.from_numpy(y_data_set.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


def prepare_dataloader(dataset_path, features_hotels, dist=[0.8, 0.2, 0], batch_size=64):
    dataset = Data(dataset_path, features_hotels)
    rep = list(map(lambda x: int(x * dataset.__len__()), dist))
    rep[-1] += dataset.__len__() - sum(rep)
    train, valid, test = torch.utils.data.random_split(dataset, rep)
    train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader, validation_dataloader


if __name__ == '__main__':
    loader = prepare_dataloader('../../dataset', "../meta_data/features_hotels.csv", batch_size=1)
    """ for batch, (X, y) in enumerate(loader[0]):
        print(f"Batch: {batch + 1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    """
