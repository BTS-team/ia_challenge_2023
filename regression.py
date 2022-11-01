import warnings
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from datascience.model import MLModel
from datascience.utils import apply
warnings.filterwarnings("ignore")


class Regression(MLModel):
    def __init__(self, dataset='dataset/', features_hotels='meta_data/features_hotels.csv'):
        super().__init__(dataset, features_hotels)
        self.train_set, self.valid_set = self.dataset.split(dist=[0.98])
        self.model = linear_model.LinearRegression()
        #degree 4 is the best
        self.poly_model = PolynomialFeatures(degree=1)

    def train(self):
        poly_x_train = self.poly_model.fit_transform(self.train_set.x)
        self.model.fit(poly_x_train, self.train_set.y)

    def predict(self, x):
        x = list(map(lambda x: apply(x), x))

        poly_x_predict= self.poly_model.fit_transform([x])

        return self.model.predict(poly_x_predict)[0]

    def validate(self):
        return super().validate(self.valid_set.x, self.valid_set.y)


if __name__ == '__main__':
    model = Regression()
    model.train()
    #print(model.validate())
    model.submission().to_csv("./polynomial_submission_degree4_1.csv", index=False)
