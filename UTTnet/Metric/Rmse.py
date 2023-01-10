from UTTnet.Metric.Mse import Mse
import numpy as np


class Rmse(Mse):
    """ Root Mean Squared Error

    Implementation of the root mean squared error.
    See https://en.wikipedia.org/wiki/Root-mean-square_deviation for more details.
    """
    def __call__(self, y_true, y_pred):
        """ RMSE calculation

        Calculate the RMSE between true values and predicted values

        :param y_true: The true values
        :type y_true: numpy.ndarray
        :param y_pred: The predicted values
        :type y_pred: numpy.ndarray
        :return: A float or a numpy array containing the RMSE values corresponding to the input values
        :rtype: Float or numpy.ndarray
        """
        return np.sqrt(super().__call__(y_true, y_pred))

    def prime(self, y_true, y_pred):
        """ RMSE prime

        Calculate the RMSE prime between true values and predicted values

        :param y_true: The true values
        :type y_true: numpy.ndarray
        :param y_pred: The predicted values
        :type y_pred: numpy.ndarray
        :return: A float or a numpy array containing the RMSE prime values corresponding to the input values
        :rtype: Float or numpy.ndarray
        """
        mse_prime = super().prime(y_true, y_pred)
        return mse_prime * (2/np.sqrt(self(y_true, y_pred)))

    def __str__(self):
        return "RMSE()"


if __name__ == '__main__':
    rmse = Rmse()
    print(rmse(np.array([10, 8, 8, 9]), np.array([12, 5, 10, 4])))
    print(rmse.prime(np.array([10, 8, 8, 9]), np.array([12, 5, 10, 4])))
