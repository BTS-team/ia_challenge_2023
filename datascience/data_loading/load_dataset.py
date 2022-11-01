from datascience.utils import get_folder, apply
from datascience.utils import NotSupportedDataTypeError, NotEqualDataTypeError
import pandas as pd
import numpy as np


def assert_equal(x, y):
    """ A function to raise an exception in both argument are not of same type

    :param x: First argument
    :param y: Seconde argument
    :return: True if both argument are of the same type
    :raise: NotEqualDataTypeError
    """
    if type(x) == type(y):
        return True
    else:
        raise NotEqualDataTypeError(x, y)


def assert_argument(x):
    """ A function to raise an exception if the given argument is not of an authorised data type

    :param x: The argument to check
    :return: True if the argument is of the right data type
    :raise: NotSupportedDataTypeError
    """
    auth = [np.ndarray, pd.core.frame.DataFrame]
    if isinstance(x, pd.core.frame.DataFrame):
        return True
    elif isinstance(x, np.ndarray):
        return True
    else:
        raise NotSupportedDataTypeError(x, auth)


class CustomDataset:
    """ A class to handle dataset using numpy array or pandas Dataframe

    :param x: The features of the dataset
    :param y: The output
    """
    def __init__(self, x, y):
        assert_equal(x, y)
        assert_argument(x)
        assert_argument(y)
        self.x = x
        self.y = y

    def getsize(self):
        """ Calculate the length of the data set

        :return: The length
        :rtype: int
        """
        return self.x.shape[0]

    def split(self, dist=[0.8, 0.15]):
        """ Split the data set in multiple subset

        If n is the number of subset, the list dist should contain n-1 values

        :param dist: A list containg the size of output subset
        :return: A tuple containing all subset
        """
        dist = list(map(lambda x: int(self.getsize() * x), dist))
        for i in range(1, len(dist)):
            dist[i] += dist[i - 1]
        result = []
        if isinstance(self.x, pd.core.frame.DataFrame):
            dist = [0] + dist + [self.getsize()]

            for i in range(1, len(dist)):
                x = self.x.iloc[dist[i - 1]:dist[i], :]
                y = self.y.iloc[dist[i - 1]:dist[i], :]
                x.reset_index(drop=True,inplace=True)
                y.reset_index(drop=True,inplace=True)
                result.append(CustomDataset(x, y))
        elif isinstance(self.x, np.ndarray):
            x_split = np.array_split(self.x, dist)
            y_split = np.array_split(self.y, dist)

            for i in range(len(x_split)):
                result.append(CustomDataset(x_split[i], y_split[i]))

        return tuple(result)

    def to_numpy(self):
        if isinstance(self.x, pd.core.frame.DataFrame):
            self.x = self.x.to_numpy()
            self.y = self.y.to_numpy()


def load_dataset(dataset_path, features_hotels, dtype="numpy"):
    """ A function to load the dataset directory

    :param dataset_path: The path of the dataset directory
    :param features_hotels: The path of the file containing features of each hotel
    :param dtype: The output data type of the function
    :return: An object containing the dataset in the data type specified in the dtype parameter
    :rtype: CustomDataset
    """
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
    y_data_set = pricing_requests[['price']]
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
    x_data_set = x_data_set.applymap(apply)
    if dtype == "numpy":
        return CustomDataset(x_data_set.to_numpy(), y_data_set.to_numpy())
    elif dtype == "pandas":
        return CustomDataset(x_data_set, y_data_set)
    else:
        raise "Wrong data type"


if __name__ == '__main__':
    # a = np.array([4, 2])
    b = np.array([4, 2, 4, 7, 8, 9, 5, 7, 5, 6, 8, 8, 5, 3])
    # print(isinstance(test, np.ndarray))
    test = CustomDataset(b, b)
    sp = test.split()

    for i in sp:
        print(i.x)
        print()