import pandas as pd
import numpy as np
import os


def get_folder(path):
    arr = os.listdir(path)
    return arr


def update_csvfile(path, dataset="./dataset"):
    requests = pd.read_csv(path).to_dict('index').values()

    df_dict = {}
    for i in requests:
        city, language = i['city'], i['language']

        city_folder = get_folder(dataset)
        # print(i)
        if city not in city_folder:
            os.mkdir(f"{dataset}/{city}")
            df_dict[f"{dataset}/{city}/{city}_{language}.csv"] = pd.DataFrame.from_records([i])
        else:
            if f"{dataset}/{city}/{city}_{language}.csv" not in df_dict.keys():
                language_file = get_folder(f"{dataset}/{city}")
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


def get_nb_row_dataset(dataset="./dataset"):
    city_folder = get_folder(dataset)
    total = 0
    for i in city_folder:
        language_file = get_folder(f"{dataset}/{i}")
        for j in language_file:
            temp = pd.read_csv(f"{dataset}/{i}/{j}")
            total += temp.shape[0]

    return total


if __name__ == '__main__':
    #update_csvfile('data/stored_requests.csv')
    update_csvfile('backup/stored_requests_3.csv')
    print(get_nb_row_dataset())
