class ApiException(Exception):
    """Exception raised for errors in the snog rest api.
    Args:
        message (str): Message to display to the user.
        statusCode (int): Status code to return from the API.
    """

    def __init__(self, message, statusCode):
        self.message = message
        self.statusCode = statusCode
        super().__init__(self.message)


class ModelNotFound(ApiException):
    def __init__(self, msg="Model not found", statusCode=404):
        super().__init__(msg, statusCode)

class InvalidGrid(ApiException):
    def __init__(self, msg="Grid is invalid", statusCode=400):
        super().__init__(msg, statusCode)