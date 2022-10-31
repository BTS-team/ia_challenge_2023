import warnings
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from datascience.model import MLModel
from datascience.utils import apply
warnings.filterwarnings("ignore")


class Regression(MLModel):
    def __init__(self, dataset='dataset/', features_hotels='meta_data/features_hotels.csv'):
        super().__init__(dataset, features_hotels)
        self.train_set, self.valid_set = self.dataset.split(dist=[0.9])
        #self.model = linear_model.LinearRegression()
        self.model = PolynomialFeatures(degree=1)

    def train(self):
        self.model.fit(self.train_set.x, self.train_set.y)

    def predict(self, x):
        x = list(map(lambda x: apply(x), x))
        return self.model.predict([x])[0]

    def validate(self):
        return super().validate(self.valid_set.x, self.valid_set.y)


if __name__ == '__main__':
    model = Regression()
    model.train()
    print(model.validate())
    #print(model.submission())
