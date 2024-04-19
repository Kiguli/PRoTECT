

class BarrierNotFoundError(Exception):
    def __init__(self):
        super(BarrierNotFoundError, self).__init__()


class ExpressionFromStringError(Exception):
    def __init__(self, message):
        super().__init__(message)


class RequiredParameterMissingError(Exception):
    def __init__(self, message):
        super().__init__(message)
