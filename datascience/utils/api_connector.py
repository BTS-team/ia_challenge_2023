import sys
import urllib
import requests


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
