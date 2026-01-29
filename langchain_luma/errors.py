# errors.py
class LumaError(Exception):
    pass


class ValidationError(LumaError):
    pass


class PayloadTooLarge(LumaError):
    pass


class DimMismatch(LumaError):
    pass


class NotFound(LumaError):
    pass


class AlreadyExists(LumaError):
    pass
