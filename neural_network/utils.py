import pandas as pd
from data import apply, language, response_test_1, response_test_2
import requests
import urllib.parse
import urllib.error
import random

class Connector:
    def __init__(self):
        domain = "51.91.251.0"
        port = 3000
        host = f"http://{domain}:{port}"
        self.path = lambda x: urllib.parse.urljoin(host, x)
        self.user_id = 'private key'

    def creating_avatar(self, name):
        try:
            r = requests.post(self.path(f'avatars/{self.user_id}/{name}'))
            print(r)
        except urllib.error as e:
            print(e)

    def getting_avatar(self):
        try:
            r = requests.get(self.path(f"avatars/{self.user_id}"))
            result = []
            for avatar in r.json():
                result.append((avatar['id'], avatar['name']))
            return result

        except urllib.error as e:
            print(e)

    def requests(self, params):
        try:
            r = requests.get(self.path(f"pricing/{self.user_id}"), params=params)
            return r.json()
        except urllib.error as e:
            print(e)


def request(connector, params, stored_requests):
    r = connector.requests(params=params)
    request = pd.DataFrame(r.json()['prices']).assign(**r.json()['request'])
    try:
        stored_r = pd.read_csv(stored_requests)
        pricing_requests = pd.concat([stored_r, request])
        pricing_requests.to_csv(stored_requests)
    except:
        request.to_csv(stored_requests)


def fake_request(stored_requests, request):
    requests = []
    for r in request:
        requests.append(pd.DataFrame(r['prices']).assign(**r['request']))

    requests = pd.concat(requests)
    requests.to_csv(stored_requests)


if __name__ == '__main__':
    city_distribution = pd.read_csv('../data/city_distribution.csv')

    cities = city_distribution['city'].tolist()
    cities_weights =city_distribution['nb_hotel'].tolist()

    random_cities=random.choices(cities, weights=cities_weights, k=1)
    random_city=random_cities[0]

    random_date=random.randint(0,44)

    random_language= random.choice(list(language.items()))[0]

    random_mobile=random.randint(0,1)

    possible_api_requests = pd.read_csv('../data/possible_api_requests.csv')
    possible_api_requests = possible_api_requests.to_numpy().tolist()

    new_request_raw =[random_city,random_language,random_date,random_mobile]
    params_condition= new_request_raw in possible_api_requests

    if params_condition:
        raw_index=possible_api_requests.index(new_request_raw)
        possible_api_requests.pop(raw_index)

        api_requests_df = pd.DataFrame(possible_api_requests, columns=['city', 'language','date','mobile'])
        api_requests_df.to_csv(f'../data/possible_api_requests.csv', index=False)

        generated_requests = pd.read_csv('../data/generated_requests.csv')
        generated_requests = generated_requests.to_numpy().tolist()
        generated_requests.append(new_request_raw)
        generated_requests_df = pd.DataFrame(generated_requests, columns=['city', 'language','date','mobile'])
        generated_requests_df.to_csv(f'../data/generated_requests.csv', index=False)

        params = {
        "avatar_name": str(random_date),
        "language": random_language,
        "city": random_city,
        "date": str(random_date),
        "mobile": str(random_mobile),
        }

        print(params)
    else:
        print("Try again")
