from sklearn.metrics import mean_squared_error


def calculate_validation(model, valid):
    valid.to_numpy()
    y_predicted = []
    for i in valid.x:
        prediction = model.predict(i)
        y_predicted.append(prediction)
    rmse = mean_squared_error(valid.y, y_predicted, squared=False)
    return rmse
