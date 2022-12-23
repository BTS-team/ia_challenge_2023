from UTTnet.Metric.Metric import Metric
import numpy as np


class Mse(Metric):
    """ Mean Squared Error

    Implementation of the mean squared error.
    See https://en.wikipedia.org/wiki/Mean_squared_error for more details.
    """

    def __call__(self, y_true, y_pred):
        """ MSE calculation

        Calculate the MSE between true values and predicted values

        :param y_true: The true values
        :type y_true: numpy.ndarray
        :param y_pred: The predicted values
        :type y_pred: numpy.ndarray
        :return: A float or a numpy array containing the MSE values corresponding to the input values
        :rtype: Float or numpy.ndarray
        """
        try:
            return np.mean(np.power(y_true-y_pred, 2))
        except Exception as err:
            print(y_true, y_pred)
            raise "error"

    def prime(self, y_true, y_pred):
        """ MSE prime

        Calculate the MSE prime between true values and predicted values

        :param y_true: The true values
        :type y_true: numpy.ndarray
        :param y_pred: The predicted values
        :type y_pred: numpy.ndarray
        :return: A float or a numpy array containing the MSE prime values corresponding to the input values
        :rtype: Float or numpy.ndarray
        """
        return 2*(y_pred-y_true)/y_true.size

    def __str__(self):
        return "MSE()"


if __name__ == '__main__':
    mse = Mse()
    print(mse(np.array([10, 8, 8, 9]), np.array([12, 5, 10, 4])))
    print(mse.prime(np.array([10, 8, 8, 9]), np.array([12, 5, 10, 4])))
