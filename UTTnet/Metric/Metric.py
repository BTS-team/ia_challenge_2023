class Metric:
    """ Abstract class for implementing metrics and loss function
    """
    def __call__(self, y_true, y_pred):
        """ Redefinition of the __call_ method
        :param y_true: The true labels
        :param y_pred: The predicted values
        """
        raise NotImplementedError

    def prime(self, y_true, y_pred):
        """ Derivative of the function implemnted in __call__ method
        :param y_true: The true labels
        :param y_pred: The predicted values
        """
        raise NotImplementedError
