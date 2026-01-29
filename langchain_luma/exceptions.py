class LumaError(Exception):
    """Base exception for Luma SDK."""

    pass


class LumaAuthError(LumaError):
    """Raised when authentication fails (401/403)."""

    pass


class LumaNotFound(LumaError):
    """Raised when a resource is not found (404)."""

    pass


class LumaConflict(LumaError):
    """Raised when a conflict occurs (409)."""

    pass


class LumaConnectionError(LumaError):
    """Raised when connection to server fails."""

    pass
