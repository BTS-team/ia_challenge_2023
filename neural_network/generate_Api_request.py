import pandas as pd
from data import language
import random
import numpy as np

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

    params = {
    "avatar_name": str(random_date),
    "language": random_language,
    "city": random_city,
    "date": str(random_date),
    "mobile": str(random_mobile),
    }
else:
    print("Try again")

print(params)