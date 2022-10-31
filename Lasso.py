from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import pandas as pd
from datascience.data_loading import load_dataset
from utils import apply


def lasso(fold, dataset_path='../dataset', features_hotels="../meta_data/features_hotels.csv"):
    dataset = load_dataset(dataset_path=dataset_path, features_hotels=features_hotels)
    model = LassoCV(cv=fold, random_state=0, max_iter=10000)
    model.fit(dataset.x, dataset.y)
    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(dataset.x, dataset.y)

    to_predict = pd.read_csv('meta_data/test_set.csv')
    hotels = pd.read_csv('meta_data/features_hotels.csv', index_col=['hotel_id', 'city'])
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

        price_prediction = model.predict([X])
        submission_df.append([index, price_prediction[0]])

    submission_df = pd.DataFrame(submission_df, columns=['index', 'price'])
    submission_df.to_csv('../sample_submission_lassoregression_2.csv', index=False)


def lass(x_set, y_set, percentage, fold):
    X_train, X_test, y_train, y_test = train_test_split(x_set, y_set, test_size=percentage, random_state=10)
    # We are using LassoCV with 5 folds but we could also try with 10 folds
    model = LassoCV(cv=fold, random_state=0, max_iter=10000)
    # Fit model
    model.fit(X_train, y_train)
    # We are using the best value of alpha 
    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(X_train, y_train)
    # Here are the coeff of the lasso regression per metrics
    print(list(zip(lasso_best.coef_, x_data_set)))
    # Compute the RMSE
    rmse = mean_squared_error(y_test, lasso_best.predict(X_test), squared=False)
    print(rmse)


# lass(x_data_set, y_data_set, 0.3, 5)

# lass(x_data_set, y_data_set, 0.2, 10)

if __name__ == '__main__':
    lasso(10)
