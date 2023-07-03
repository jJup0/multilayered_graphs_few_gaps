from typing import Any, Callable, NoReturn, TypeVar

T = TypeVar("T")


def lgraph_sorting_algorithm(func: T) -> T:
    """Decorator for function which sorts a layered graph.

    Does not modify the function in any way
    """
    return func


def deprecated_lgraph_sorting(func: Callable[..., Any]) -> Callable[..., NoReturn]:
    """Decorator for function which is deprecated.
    Any deprecation note is expected in the docstring.
    """

    def wrapper(*args: Any, **kwargs: Any) -> NoReturn:
        raise DeprecationWarning(f"Do not use: {func.__doc__}")

    return wrapper
