from datascience.model import MLModel
from datascience.data_loading import load_dataset


class DeepLearningModel(MLModel):
    def __init__(self, dataset='dataset/', features_hotels='meta_data/features_hotels.csv'):
        self.dataset = load_dataset(dataset, features_hotels, dtype="pandas")
        print(self.dataset.x)


if __name__ == '__main__':
    DeepLearningModel()
