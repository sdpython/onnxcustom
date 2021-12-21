"""
@file
@brief Exceptions.
"""


class ConvergenceError(RuntimeError):
    """
    Raised when a learning algorithm failed
    to converge.
    """
    pass


class EvaluationError(RuntimeError):
    """
    Raised when an evaluation failed.
    """
    pass


class ProviderError(RuntimeError):
    """
    Raised when an input is not on the expected device (CPU, GPU).
    """
    pass
