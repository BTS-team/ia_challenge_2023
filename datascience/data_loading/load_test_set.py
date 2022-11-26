import pandas as pd
from datascience.utils import apply


def load_test_set(test_set='meta_data/test_set.csv', features_hotels='meta_data/features_hotels.csv'):
    to_predict = pd.read_csv(test_set)
    hotels = pd.read_csv(features_hotels, index_col=['hotel_id', 'city'])
    to_predict = to_predict.join(hotels, on=['hotel_id', 'city'])
    to_predict = to_predict.applymap(apply)

    return to_predict['index'], to_predict[[
        'city',
        'date',
        'language',
        'mobile',
        'stock',
        'group',
        'brand',
        'parking',
        'pool',
        'children_policy',
        'order_request'
    ]]
