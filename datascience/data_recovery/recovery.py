import pandas as pd
from datascience.utils import language, get_folder, generate_histo, Connector, get_nb_row_dataset
import random
import numpy as np
import os


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


def request(connector, params, dataset, path_generated_request, poss_request, avatar_path, retry=False):
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
        try:
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
            avatar_use = pd.read_csv(avatar_path)
            avatar = connector.get_avatar()
            avatar_id = list(avatar.keys())[list(avatar.values()).index(str(params['avatar_name']))]

            temp = avatar_use.loc[
                (avatar_use['avatar_id'] == avatar_id)
            ]
            if temp.shape[0] < 1:
                avatar_use.append({
                    'avatar_id': avatar_id,
                    'date': params['date'],
                    'order_requests': 1
                }, ignore_index=True).to_csv(avatar_path, index=False)
                # print(f"created {avatar_id}")
            else:
                avatar_use.loc[
                    (avatar_use['avatar_id'] == avatar_id), 'order_requests'
                ] += 1
                avatar_use.to_csv(avatar_path, index=False)
                # print(f"modified {avatar_id}")
            print(f"Request({params['city']},{params['language']},{params['date']},{params['mobile']}) ==> Done")
        except:
            if retry:
                print(params)
            else:
                avatar_id, name = create_avatar(connector)
                params['avatar_id'] = avatar_id
                request(connector, params, dataset, path_generated_request, poss_request, avatar_path, retry=True)


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


def create_avatar(connector):
    avatar = connector.get_avatar()
    name = max(list(map(lambda x: int(x), list(avatar.values())))) + 1
    connector.create_avatar(str(name))
    avatar = connector.get_avatar()
    avatar_id = list(avatar.keys())[list(avatar.values()).index(str(name))]
    return avatar_id, name


def random_avatar(date, connector, avatar_use):
    temp = np.array(avatar_use)
    poss_avatar = temp[(temp[:, 1] > date)]

    if poss_avatar.shape[0] == 0:
        avatar_id, name = create_avatar(connector)
        avatar_use.append([avatar_id, date, 1])
        return name
    else:
        avatar_id = random.choice(poss_avatar[:, 0])
        return connector.get_avatar()[avatar_id]


def assigning_avatar(queries, connector, avatar_path):
    """ A function to create and assign avatar to requests.
    This function is used to create the right amount of avatar for a group a request and assign an avatar to each request depending of the date of this request.

    :param avatar_path: The path of the csv file listing all the avatars used for the project, theirs start date and number of requests
    :param queries: A pandas.DataFrame containing all requests
    :param connector: A Connector object to communicate with the API
    :return: None
    """
    avatar_use = pd.read_csv(avatar_path)
    avatar_use = avatar_use.loc[
        (avatar_use['order_requests'] <= 8)
    ]
    avatar_use = avatar_use.to_numpy().tolist()
    queries['avatar_name'] = queries['date'].apply(lambda x: random_avatar(x, connector, avatar_use))


def making_n_requests(path_requests, path_city, nb_requests, private_key, dataset, path_gen_request, avatar_path):
    """ A function used to perform n different requests to the api

    :param avatar_path: The path of the csv file listing all the avatars used for the project, theirs start date and number of requests
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
    assigning_avatar(queries, connector, avatar_path)
    queries_dict = list(queries.to_dict('index').values())

    for i in queries_dict:
        request(connector, i, dataset, path_gen_request, path_requests, avatar_path)


if __name__ == '__main__':
    params = {
        'path_requests': '../meta_data/possible_api_requests.csv',
        'path_city': '../meta_data/city_distribution.csv',
        'nb_requests': 80,
        'private_key': 'c760f776-e640-4d8c-a26e-dba910cc7218',
        'path_gen_request': '../meta_data/generated_requests.csv',
        'dataset': '../dataset'
    }
    # making_n_requests(**params)
    histo, total = generate_histo('../meta_data/generated_requests.csv')
    print()
    print(histo)
    print(f"Nb de requetes {total}")
    print(f"Nb de ligne {get_nb_row_dataset()}")
