from datascience.data_recovery import making_n_requests
from datascience.utils import generate_histo, get_nb_row_dataset

params = {
    'path_requests': 'meta_data/possible_api_requests.csv',
    'path_city': 'meta_data/city_distribution.csv',
    'nb_requests': 200,
    'private_key': 'c760f776-e640-4d8c-a26e-dba910cc7218',
    'path_gen_request': 'meta_data/generated_requests.csv',
    'dataset': 'dataset/',
    'avatar_path': './meta_data/avatar_utilization.csv'
}
#making_n_requests(**params)

histo, total = generate_histo('meta_data/generated_requests.csv')
print()
print(histo)
print(f"Nb de requetes {total}")
print(f"Nb de ligne {get_nb_row_dataset()}")
