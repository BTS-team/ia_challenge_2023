import torch
import pandas as pd
import numpy as np
from sklearn import linear_model
from utils import get_folder
from data import apply

def regression():
    dataset_path = '../dataset'
    features_hotels = "../meta_data/features_hotels.csv"

    city_folder = get_folder(dataset_path)
    rows = None
    for i in city_folder:
        language_file = get_folder(f"{dataset_path}/{i}")
        for j in language_file:
            temp = pd.read_csv(f"{dataset_path}/{i}/{j}")
            if rows is None:
                rows = temp.to_numpy()
            else:
                rows = np.concatenate((rows, temp.to_numpy()))

    np.random.shuffle(rows)
    rows = pd.DataFrame(rows,
                        columns=['hotel_id', 'price', 'stock', 'city', 'date', 'language', 'mobile', 'avatar_id'])
    hotels = pd.read_csv(features_hotels, index_col=['hotel_id', 'city'])
    pricing_requests = rows.join(hotels, on=['hotel_id', 'city'])

    x_data_set = pricing_requests[[
        'city',
        'date',
        'language',
        'mobile',
        'stock',
        'group',
        'brand',
        'parking',
        'pool',
        'children_policy'
    ]]

    x_data_set = x_data_set.applymap(apply)
    y_data_set = pricing_requests['price']

    reg = linear_model.LinearRegression()
    reg.fit(x_data_set,y_data_set)

    #Prediction Part
    to_predict = pd.read_csv('../data/test_set.csv')
    hotels = pd.read_csv('../meta_data/features_hotels.csv', index_col=['hotel_id', 'city'])
    to_predict = to_predict.join(hotels, on=['hotel_id', 'city'])

    submission_df = []

    for i in to_predict.to_dict('index').values():
        index = i['index']
        X = [
            i['city'],
            i['date'],
            i['language'],
            i['mobile'],
            i['stock'],
            i['group'],
            i['brand'],
            i['parking'],
            i['pool'],
            i['children_policy']
        ]
        X = list(map(lambda x: apply(x), X))
        
        price_prediction=reg.predict([X])
        submission_df.append([index, price_prediction[0]])
    
    submission_df = pd.DataFrame(submission_df, columns=['index', 'price'])
    submission_df.to_csv('../sample_submission.csv',index=False)

if __name__ == '__main__':
    regression()