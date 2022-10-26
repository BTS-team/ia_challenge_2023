import sys

import pandas as pd
from data import apply, language, response_test_1, response_test_2, city
import requests
import urllib.parse
import urllib.error
import random
import numpy as np


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


def request(connector, params, stored_requests):
    r = connector.query(params=params)
    if r != 200:
        try:
            queries = pd.DataFrame(r['prices']).assign(**r['request'])
            stored_r = pd.read_csv(stored_requests)
            pricing_requests = pd.concat([stored_r, queries])
            pricing_requests.to_csv(stored_requests, index=False)
        except:
            queries.to_csv(stored_requests, index=False)


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


def take_n_requests(path_requests, path_city, nb_requests, path_gen_request):
    city_distribution = pd.read_csv(path_city)
    poss_api_requests = pd.read_csv(path_requests)

    cities_weights = city_distribution['nb_hotel'].tolist()
    cities = city_distribution['city'].tolist()

    queries = []

    while len(queries) < nb_requests:
        r_city = random.choices(cities, weights=cities_weights)[0]
        r_date = random.randint(0, 44)
        r_language = random.choice(list(language.items()))[0]
        r_mobile = random.randint(0, 1)
        # print(r_city, r_date, r_language, r_mobile)

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
    already_gen = pd.read_csv(path_gen_request)
    already_gen = pd.concat([already_gen, generated_requests])
    already_gen.to_csv(path_gen_request, index=False)
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


def making_n_requests(path_requests, path_city, nb_requests, private_key, stored_request, path_gen_request):
    queries = take_n_requests(path_requests, path_city, nb_requests, path_gen_request)
    connector = Connector(private_key)
    assigning_avatar(queries, connector)
    queries_dict = list(queries.to_dict('index').values())

    for i in queries_dict:
        request(connector, i, stored_request)


def generate_histo():
    generated_r = pd.read_csv('../data/generated_requests.csv')

    attribute_row = generated_r['city'].to_numpy()
    keys = set(attribute_row.tolist())
    total_hotel = len(attribute_row)
    histo = []

    for i in keys:
        nb_hotel = len(list(filter(lambda x: x == i, attribute_row)))
        pourcentage = nb_hotel / total_hotel
        histo.append([i, nb_hotel, pourcentage])

    print(pd.DataFrame(histo))



if __name__ == '__main__':
    params = {
        'path_requests': '../data/possible_api_requests.csv',
        'path_city': '../data/city_distribution.csv',
        'nb_requests': 89,
        'private_key': 'c760f776-e640-4d8c-a26e-dba910cc7218',
        'path_gen_request': '../data/generated_requests.csv',
        'stored_request': '../data/stored_requests.csv'
    }
    #print(response.history)
    #making_n_requests(**params)

    print("fin de la generation")
    generate_histo()
    # generate_api_requests('../data/possible_api_requests.csv')

    # store = pd.read_csv('../data/stored_requests.csv')
    # stored = store[['hotel_id', 'price', 'stock', 'city', 'date', 'language', 'mobile', 'avatar_id']]
    # print(stored)
    # stored.to_csv('../data/stored_requests.csv',index=False)
