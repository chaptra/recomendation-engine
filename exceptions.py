"""Custom exceptions for the recommendation engine."""


class RecommendationEngineError(Exception):
    """Base exception for recommendation engine."""
    pass


class DatabaseConnectionError(RecommendationEngineError):
    """Raised when database connection fails."""
    pass


class CacheError(RecommendationEngineError):
    """Raised when cache operations fail."""
    pass


class ModelNotReadyError(RecommendationEngineError):
    """Raised when model is not properly initialized."""
    pass


class BookNotFoundError(RecommendationEngineError):
    """Raised when requested book is not found."""
    pass


class InvalidInputError(RecommendationEngineError):
    """Raised when input validation fails."""
    pass


class RecommendationTimeoutError(RecommendationEngineError):
    """Raised when recommendation computation times out."""
    pass