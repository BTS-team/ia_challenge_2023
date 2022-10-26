import sys
import pandas as pd
from data import apply, language, response_test_1, response_test_2, city
import requests
import urllib.parse
import urllib.error
import random
import numpy as np
import os


def get_folder(path):
    """A function to get the list of elements in a given directory

    :param path: The path of the directory
    :return: A list of string containing all files names and directories names
    :rtype: list of string
    """
    arr = os.listdir(path)
    return arr


class Connector:
    """A class to connect to the api

    :param user_id: The private key to connect to the api
    :param path: The web path of the api
    """
    def __init__(self, key):
        """Constructor method
        :param key: The private key of the api
        """
        domain = "51.91.251.0"
        port = 3000
        host = f"http://{domain}:{port}"
        self.path = lambda x: urllib.parse.urljoin(host, x)
        self.user_id = key

    def create_avatar(self, name):
        """A method to create an avatar

        :param name: The name of the avatar to create
        :return: None
        """
        try:
            r = requests.post(self.path(f'avatars/{self.user_id}/{name}'))
            print(r)
        except urllib.error as e:
            print(e)

    def get_avatar(self):
        """A method to get all avatar for the private key given in the constructor method

        :return: A list of tuple containing each avatars and theirs ids
        """
        try:
            r = requests.get(self.path(f"avatars/{self.user_id}"))
            result = []
            for avatar in r.json():
                result.append([avatar['id'], avatar['name']])
            return result

        except urllib.error as e:
            print(e)

    def query(self, params):
        """A method to make a request to the api

        :param params: A dictionary containing all parameters of the request. This dictionary must be defined as in the tutorial.
        :return: Either the request result in dictionary format nor the error code
        """
        try:
            r = requests.get(self.path(f"pricing/{self.user_id}"), params=params)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 422:
                return 422
            else:
                print(r.status_code, r.json()['detail'])
                sys.exit(1)

        except urllib.error as e:
            print(e)


def update_dataset(city, language, queries, dataset):
    """ A function to save a request in the dataset directory taking into account the sub directories and sub files.

    :param city: The city of the request to save in the dataset
    :param language: the language of the request to save in the dataset
    :param queries: A pandas.DataFrame containing the api response for the request
    :param dataset: The path of the dataset directory
    :return: None
    """
    city_folder = get_folder(dataset)

    if city not in city_folder:
        os.mkdir(f"{dataset}/{city}")
        queries.to_csv(f"{dataset}/{city}/{city}_{language}.csv", index=False)

    else:
        language_file = get_folder(f"{dataset}/{city}")
        if f"{city}_{language}.csv" not in language_file:
            queries.to_csv(f"{dataset}/{city}/{city}_{language}.csv", index=False)
        else:
            temp = pd.read_csv(f"{dataset}/{city}/{city}_{language}.csv")
            temp = pd.concat([temp, queries])
            temp.to_csv(f"{dataset}/{city}/{city}_{language}.csv", index=False)


def request(connector, params, dataset, path_generated_request, poss_request):
    """ A function to perform and save the result of a request

    :param connector: A Connector object to communicate with the API
    :param params: The parameters of the request to perform (see the kaggle tutorial)
    :param dataset: The path of the directory where to save the result of the request
    :param path_generated_request: The path of the csv file containing all already used request
    :param poss_request: The path of the file containing all possible requests that can be made
    :return: None
    """
    r = connector.query(params=params)
    if r != 422:
        queries = pd.DataFrame(r['prices']).assign(**r['request'])
        update_dataset(params['city'], params['language'], queries, dataset)

        gen = pd.read_csv(path_generated_request)
        gen = pd.concat([
            gen,
            pd.DataFrame.from_records([params])[[
                'city',
                'language',
                'date',
                'mobile'
            ]]
        ])
        gen.to_csv(path_generated_request, index=False)

        poss_api_requests = pd.read_csv(poss_request)
        poss_api_requests.loc[
            (poss_api_requests['city'] == params['city']) &
            (poss_api_requests['date'] == params['date']) &
            (poss_api_requests['language'] == params['language']) &
            (poss_api_requests['mobile'] == params['mobile']), 'used'
        ] = 2
        poss_api_requests.to_csv(poss_request, index=False)
        print(f"Request({params['city']},{params['language']},{params['date']},{params['mobile']}) ==> Done")


def fake_request(stored_requests, query):
    """ A function to simulate a request

    Not used anymore

    :param stored_requests:
    :param query:
    :return:
    """
    queries = []
    for r in query:
        queries.append(pd.DataFrame(r['prices']).assign(**r['request']))

    queries = pd.concat(queries)
    queries.to_csv(stored_requests)


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


def take_n_requests(path_requests, path_city, nb_requests, generated_r):
    """ A function to generate n requests randomly

    :param path_requests: The path of the CSV file containing all possible api requests
    :param path_city: The path of the file containing the distribution of hotels among the different cities
    :param nb_requests: The number of request to generate
    :param generated_r: The path of the file containing all already generated requests
    :return: A pandas.DataFrame containing the n generated requests
    """
    real_dist, _ = generate_histo(generated_r)
    theo = pd.read_csv(path_city)
    theo = theo.sort_values(by=['city'], ignore_index=True)
    real_dist['hotels'] = theo['distribution']
    real_dist['difference'] = theo['distribution'] - real_dist['dataset']
    real_dist['corrige'] = real_dist['hotels'] + real_dist['difference']
    print(real_dist)
    poss_api_requests = pd.read_csv(path_requests)

    cities_weights = real_dist['corrige'].tolist()
    cities = real_dist['city'].tolist()
    poss_api_requests.loc[(poss_api_requests['used'] == 1)] = 0
    queries = []

    while len(queries) < nb_requests:
        r_city = random.choices(cities, weights=cities_weights)[0]
        r_date = random.randint(0, 44)
        r_language = random.choice(list(language.items()))[0]
        r_mobile = random.randint(0, 1)

        choosen_request = poss_api_requests.loc[
            (poss_api_requests['city'] == r_city) &
            (poss_api_requests['date'] == r_date) &
            (poss_api_requests['language'] == r_language) &
            (poss_api_requests['mobile'] == r_mobile)
            ]
        if choosen_request['used'].all() == 0:
            queries.append([r_city, r_language, r_date, r_mobile])
            poss_api_requests.loc[
                (poss_api_requests['city'] == r_city) &
                (poss_api_requests['date'] == r_date) &
                (poss_api_requests['language'] == r_language) &
                (poss_api_requests['mobile'] == r_mobile), 'used'
            ] = 1
    poss_api_requests.to_csv(path_requests, index=False)
    generated_requests = pd.DataFrame(queries, columns=['city', 'language', 'date', 'mobile'])
    return generated_requests


def assigning_avatar(queries, connector):
    """ A function to create and assign avatar to requests.
    This function is used to create the right amount of avatar for a group a request and assign an avatar to each request depending of the date of this request.

    :param queries: A pandas.DataFrame containing all requests
    :param connector: A Connector object to communicate with the API
    :return: None
    """
    avatars = np.array(connector.get_avatar())
    start_avatar = max(list(map(lambda x: int(x), avatars[:, 1]))) + 1
    avatar_dict = dict()
    date = sorted(list(set(queries['date'].tolist())))

    for i in range(len(date)):
        temp = start_avatar + i
        avatar_dict[date[i]] = temp
        connector.create_avatar(f"{temp}")

    queries['avatar_name'] = queries['date'].apply(lambda x: avatar_dict[x])


def making_n_requests(path_requests, path_city, nb_requests, private_key, dataset, path_gen_request):
    """ A function used to perform n different requests to the api

    :param path_requests: The path of the file containing all possible requests that can be made
    :param path_city: The path of the csv file containing the distribution of hotels among the different cities
    :param nb_requests: The number of request to perform
    :param private_key: The private key needed to connect with the api
    :param dataset: The path of the directory used to store the results of each api requests
    :param path_gen_request: The path of the csv file containing all already used request
    :return: None
    """
    queries = take_n_requests(path_requests, path_city, nb_requests, path_gen_request)
    connector = Connector(private_key)
    assigning_avatar(queries, connector)
    queries_dict = list(queries.to_dict('index').values())

    for i in queries_dict:
        request(connector, i, dataset, path_gen_request, path_requests)


def get_nb_row_dataset(dataset="../dataset"):
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


if __name__ == '__main__':
    params = {
        'path_requests': '../meta_data/possible_api_requests.csv',
        'path_city': '../meta_data/city_distribution.csv',
        'nb_requests': 80,
        'private_key': 'c760f776-e640-4d8c-a26e-dba910cc7218',
        'path_gen_request': '../meta_data/generated_requests.csv',
        'dataset': '../dataset'
    }
    #making_n_requests(**params)
    histo,total = generate_histo('../meta_data/generated_requests.csv')
    print()
    print(histo)
    print(f"Nb de requetes {total}")
    print(f"Nb de ligne {get_nb_row_dataset()}")

