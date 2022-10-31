from data_loading import load_test_set, load_dataset
import pandas as pd
from sklearn.metrics import mean_squared_error


class MLModel:
    def __init__(self, dataset='../dataset', features_hotels='../meta_data/features_hotels.csv'):
        self.dataset = load_dataset(dataset, features_hotels)
        self.dataset.to_numpy()
        self.features_hotels = features_hotels

    def train(self, **kwargs):
        raise NotImplementedError

    def validate(self, x, y):
        y_predicted = []
        for i in x:
            prediction = self.predict(i)
            y_predicted.append(prediction)
        rmse = mean_squared_error(y, y_predicted, squared=False)
        return rmse

    def predict(self, x):
        raise NotImplementedError

    def submission(self, test_set='../meta_data/test_set.csv'):
        index, x = load_test_set(test_set, self.features_hotels)
        submission_df = []
        index = index.to_numpy()
        x = x.to_numpy()
        for i in range(len(x)):
            prediction = self.predict(x[i])
            submission_df.append([index[i], prediction])

        submission_df = pd.DataFrame(submission_df, columns=['index', 'price'])
        return submission_df
