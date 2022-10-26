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
    arr = os.listdir(path)
    return arr


class Connector:
    def __init__(self, key):
        domain = "51.91.251.0"
        port = 3000
        host = f"http://{domain}:{port}"
        self.path = lambda x: urllib.parse.urljoin(host, x)
        self.user_id = key

    def create_avatar(self, name):
        try:
            r = requests.post(self.path(f'avatars/{self.user_id}/{name}'))
            print(r)
        except urllib.error as e:
            print(e)

    def get_avatar(self):
        try:
            r = requests.get(self.path(f"avatars/{self.user_id}"))
            result = []
            for avatar in r.json():
                result.append([avatar['id'], avatar['name']])
            return result

        except urllib.error as e:
            print(e)

    def query(self, params):
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
    queries = []
    for r in query:
        queries.append(pd.DataFrame(r['prices']).assign(**r['request']))

    queries = pd.concat(queries)
    queries.to_csv(stored_requests)


def generate_api_requests(path):
    api_requests = []
    for c in city:
        for l in language:
            for i in range(45):
                api_requests.append([c, l, i, 0, 0])
                api_requests.append([c, l, i, 1, 0])

    api_requests_df = pd.DataFrame(api_requests, columns=['city', 'language', 'date', 'mobile', 'used'])
    api_requests_df.to_csv(path, index=False)


def take_n_requests(path_requests, path_city, nb_requests):
    city_distribution = pd.read_csv(path_city)
    poss_api_requests = pd.read_csv(path_requests)

    cities_weights = city_distribution['nb_hotel'].tolist()
    cities = city_distribution['city'].tolist()
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
    queries = take_n_requests(path_requests, path_city, nb_requests)
    connector = Connector(private_key)
    assigning_avatar(queries, connector)
    queries_dict = list(queries.to_dict('index').values())

    for i in queries_dict:
        request(connector, i, dataset, path_gen_request, path_requests)


def generate_histo(dataset="../dataset"):
    city_folder = get_folder(dataset)
    histo = []

    for i in city_folder:
        language_file = get_folder(f"{dataset}/{i}")

        total_city = 0
        for j in language_file:
            temp = pd.read_csv(f"{dataset}/{i}/{j}")
            total_city += temp.shape[0]

        histo.append([i, total_city])

    histo = pd.DataFrame(histo, columns=['city', 'row_nb'])
    total = sum(histo['row_nb'])
    histo['distribution'] = (histo['row_nb']/total).round(3)
    print(histo)


def get_nb_row_dataset(dataset="../dataset"):
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
        'nb_requests': 1,
        'private_key': 'c760f776-e640-4d8c-a26e-dba910cc7218',
        'path_gen_request': '../meta_data/generated_requests.csv',
        'dataset': '../dataset'
    }
    # print(response.history)
    making_n_requests(**params)
    generate_histo()
    # generate_api_requests('../meta_data/possible_api_requests.csv')
