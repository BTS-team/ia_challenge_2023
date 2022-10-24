import pandas as pd
from data import apply, response_test
import requests
import urllib.parse
import urllib.error


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


if __name__ == '__main__':
    print(prepare_train("../data/test.csv", "../data/features_hotels.csv"))
