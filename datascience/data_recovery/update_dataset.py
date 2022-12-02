import pandas as pd
import numpy as np
import os
from datascience.utils import utils


def update_csvfile(path, dataset="./dataset"):
    requests = pd.read_csv(path).to_dict('index').values()

    df_dict = {}
    for i in requests:
        city, language = i['city'], i['language']

        city_folder = utils.get_folder(dataset)
        # print(i)
        if city not in city_folder:
            os.mkdir(f"{dataset}/{city}")
            df_dict[f"{dataset}/{city}/{city}_{language}.csv"] = pd.DataFrame.from_records([i])
        else:
            if f"{dataset}/{city}/{city}_{language}.csv" not in df_dict.keys():
                language_file = utils.get_folder(f"{dataset}/{city}")
                if f"{city}_{language}.csv" not in language_file:
                    df_dict[f"{dataset}/{city}/{city}_{language}.csv"] = pd.DataFrame.from_records([i])
                else:
                    temp = pd.read_csv(f"{dataset}/{city}/{city}_{language}.csv")
                    df_dict[f"{dataset}/{city}/{city}_{language}.csv"] = pd.concat(
                        [temp, pd.DataFrame.from_records([i])])
            else:
                df_dict[f"{dataset}/{city}/{city}_{language}.csv"] = pd.concat(
                    [df_dict[f"{dataset}/{city}/{city}_{language}.csv"], pd.DataFrame.from_records([i])])

    for i in df_dict.keys():
        df_dict[i].drop_duplicates(keep='first').to_csv(i, index=False)


def generate_generated_requests(dataset="./dataset"):
    city_folder = utils.get_folder(dataset)
    generated_request = pd.DataFrame(columns=['city', 'language', 'date', 'mobile'])

    for i in city_folder:
        language_file = utils.get_folder(f"{dataset}/{i}")
        for j in language_file:
            temp = pd.read_csv(f"{dataset}/{i}/{j}")[['city', 'language', 'date', 'mobile']]
            generated_request = pd.concat([generated_request, temp])

    generated_request.drop_duplicates(keep='first').to_csv("meta_data/generated_requests.csv", index=False)


def update_possible_requests(gen_request="meta_data/generated_requests.csv",
                             path_poss_request="meta_data/possible_api_requests.csv"):
    gen = pd.read_csv(gen_request).to_dict('index').values()
    poss_request = pd.read_csv(path_poss_request)

    for i in gen:
        poss_request.loc[
            (poss_request['city'] == i['city']) &
            (poss_request['date'] == i['date']) &
            (poss_request['language'] == i['language']) &
            (poss_request['mobile'] == i['mobile']), 'used'
        ] = 2

    poss_request.to_csv(path_poss_request, index=False)


def add_column_order_request(dataset="../../dataset"):
    city_folder = utils.get_folder(dataset)
    for i in city_folder:
        language_file = utils.get_folder(f"{dataset}/{i}")
        for j in language_file:
            temp = pd.read_csv(f"{dataset}/{i}/{j}")
            temp['order_requests'] = 0
            temp.to_csv(f"{dataset}/{i}/{j}", index=False)


def change_name_order_request(dataset="../../dataset"):
    city_folder = utils.get_folder(dataset)
    for i in city_folder:
        language_file = utils.get_folder(f"{dataset}/{i}")
        for j in language_file:
            temp = pd.read_csv(f"{dataset}/{i}/{j}")
            temp = temp.rename(columns={'order_request': 'order_requests'})
            temp.to_csv(f"{dataset}/{i}/{j}", index=False)


def add_order_request(dataset="../../dataset", gen_request="../../meta_data/generated_requests.csv"):
    gen_request = pd.read_csv(gen_request).to_numpy()
    order_request = {}

    for i in gen_request:
        temp = pd.read_csv(f"{dataset}/{i[0]}/{i[0]}_{i[1]}.csv")
        avatar_id = temp.loc[(temp['date'] == i[2]) & (temp['mobile'] == i[3])]['avatar_id'].to_numpy()[0]
        if avatar_id in order_request.keys():
            order_request[avatar_id] += 1
        else:
            order_request[avatar_id] = 1

        temp.loc[
            (temp['date'] == i[2]) & (temp['mobile'] == i[3]), 'order_requests'
        ] = order_request[avatar_id]
        temp.to_csv(f"{dataset}/{i[0]}/{i[0]}_{i[1]}.csv", index=False)


def update_test_set(test_set_path="../../meta_data/test_set.csv"):
    test_set = pd.read_csv(test_set_path)
    columns = list(test_set.columns)
    test_set = test_set.to_numpy()

    count = test_set[0][1]

    for i in range(1, len(test_set)):
        if test_set[i][6] != test_set[i - 1][6]:
            count = test_set[i][1]

        test_set[i][1] += -count + 1
    test_set = pd.DataFrame(test_set, columns=columns)
    test_set.to_csv(test_set_path, index=False)


if __name__ == '__main__':
    # update_csvfile('data/stored_requests.csv')
    # update_csvfile('backup/stored_requests_3.csv')
    # print(get_nb_row_dataset())
    update_test_set()
