class NotSupportedDataTypeError(Exception):
    """ Exception raised when the given data type of an argument is not supported by the function or method

    :param argument: The argument that is in wrong datatype
    :param authorised_dtype: The list of autorised datatypes
    """

    def __init__(self, argument, authorised_dtype):
        self.dtype = type(argument)
        self.auth_dtype = authorised_dtype
        self.message = f"DataType not supported, please use one of those {self.auth_dtype}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.dtype} --> {self.message}"


class NotEqualDataTypeError(Exception):
    """ Exception raised when data types of two arguments are not identical whereas they should be

    :param x: First argument
    :param y: Seconde argument
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.message = "Both datatypes must be identical"
        super().__init__(self.message)

    def __str__(self):
        return f"{type(self.x)} and {type(self.y)} --> {self.message}"


class ModelAlreadyExist(Exception):
    """ Exception raised when a model with the same name and date already exist in the target folder

    :param name: Name of the model
    :param date: Date of the model
    """
    def __init__(self,name,date):
        self.name = name
        self.date = date
        self.message = "This model already exist in the target folder"
        super().__init__(self,self.message)

    def __str__(self):
        return f"{self.name} & {self.date} --> {self.message}"

