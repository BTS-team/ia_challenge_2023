import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings('ignore')


index_col = ['hotel_id', 'group', 'brand', 'city', 'parking', 'pool', 'children_policy']

hotels = pd.read_csv('../data/features_hotels.csv')

city = hotels['city'].to_numpy()

histo = {}

for i in city:
    if i in histo:
        histo[i] += 1
    else:
        histo[i] = 1

print(histo)
