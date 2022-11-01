import os
from datascience.utils.data import language, city
import pandas as pd


def get_folder(path):
    """A function to get the list of elements in a given directory

    :param path: The path of the directory
    :return: A list of string containing all files names and directories names
    :rtype: list of string
    """
    arr = os.listdir(path)
    return arr


def generate_api_requests(path):
    """ A function to generate all possible api requests

    :param path: The path of the file where to save the possible requests
    :return: None
    """
    api_requests = []
    for c in city:
        for l in language:
            for i in range(45):
                api_requests.append([c, l, i, 0, 0])
                api_requests.append([c, l, i, 1, 0])

    api_requests_df = pd.DataFrame(api_requests, columns=['city', 'language', 'date', 'mobile', 'used'])
    api_requests_df.to_csv(path, index=False)


def generate_histo(gen_request):
    """ A function to generate the distribution of city among the generated requests

    :param gen_request: The path of the file containing all already generated requests
    :return: A tuple containing the distribution of cities and the total number of rows in the dataset
    """
    generated_r = pd.read_csv(gen_request)['city']
    histo = {}

    for i in generated_r.to_list():
        if i in histo.keys():
            histo[i] += 1
        else:
            histo[i] = 1
    total = sum(histo.values())
    data = {'city': histo.keys(), 'nb_requests': histo.values()}
    distribution = pd.DataFrame.from_dict(data)
    distribution['dataset'] = distribution['nb_requests'] / total
    return distribution, total


def get_nb_row_dataset(dataset="dataset/"):
    """A function to get the number of rows in the dataset

    :param dataset: The path of the dataset
    :return: The number of rows
    :rtype: int
    """
    city_folder = get_folder(dataset)
    total = 0
    for i in city_folder:
        language_file = get_folder(f"{dataset}/{i}")
        for j in language_file:
            temp = pd.read_csv(f"{dataset}/{i}/{j}")
            total += temp.shape[0]

    return total
