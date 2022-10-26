import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def get_distribution(hotels, attribute):
    attribute_row = hotels[attribute].to_numpy()
    print(attribute_row)
    keys = set(attribute_row.tolist())
    total_hotel = len(attribute_row)
    histo = []

    for i in keys:
        nb_hotel = len(list(filter(lambda x: x == i, attribute_row)))
        pourcentage = nb_hotel / total_hotel
        histo.append([i, nb_hotel, pourcentage])

    attribute_distribution = pd.DataFrame(histo, columns=[attribute, 'nb_hotel', 'distribution'])
    attribute_distribution['distribution'] = attribute_distribution['distribution'].round(3)
    attribute_distribution.to_csv(f'../data/{attribute}_distribution.csv', index=False)


if __name__ == '__main__':
    index_col = ['hotel_id', 'group', 'brand', 'city', 'parking', 'pool', 'children_policy']
    hotels = pd.read_csv('../meta_data/features_hotels.csv')

    attribute = ['city', 'brand', 'group', 'pool', 'parking', 'children_policy']
    for i in attribute:
        get_distribution(hotels, i)
