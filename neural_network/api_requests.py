import pandas as pd
from data import language, city

api_requests=[]
for c in city:
    for l in language:
        for i in range(45):
            api_requests.append([c,l,str(i),str(0)])
            api_requests.append([c,l,str(i),str(1)])

api_requests_df = pd.DataFrame(api_requests, columns=['city', 'language','date','mobile'])
api_requests_df.to_csv(f'../data/possible_api_requests.csv', index=False)
