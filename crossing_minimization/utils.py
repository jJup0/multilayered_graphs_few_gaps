from typing import Any, Callable, Literal, NoReturn, TypeVar

from multilayered_graph.multilayered_graph import MultiLayeredGraph

T = TypeVar("T")

DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION = 3


def lgraph_sorting_algorithm(func: T) -> T:
    """Decorator for function which sorts a layered graph.

    Function head should look like this:
    (ml_graph: MultiLayeredGraph, *, max_iterations: int = 3, one_sided: bool = False) -> None

    Does not modify the function in any way.
    """
    return func


def deprecated_lgraph_sorting(func: Callable[..., Any]) -> Callable[..., NoReturn]:
    """Decorator for function which is deprecated.
    Any deprecation note is expected in the docstring.
    """

    def wrapper(*args: Any, **kwargs: Any) -> NoReturn:
        raise DeprecationWarning(f"Do not use: {func.__doc__}")

    return wrapper


def sorting_parameter_check(
    ml_graph: MultiLayeredGraph, *, max_iterations: int, one_sided: bool
):
    # input parameter validation
    if not (one_sided is True or one_sided is False):
        raise ValueError(f'one_side must be true or false, received "{one_sided}"')
    if one_sided:
        if ml_graph.layer_count != 2:
            raise ValueError(
                f"One-sided crossing minimization can only be performed on graphs with exactly 2 layers."
                f"Input graph has {ml_graph.layer_count} layers."
            )
    elif max_iterations <= 0:
        raise ValueError(f"iterations must be > 0, received {max_iterations}")


def get_layer_idx_above_or_below(
    layer_idx: int, above_or_below: Literal["above"] | Literal["below"]
):
    if above_or_below == "above":
        return layer_idx + 1
    if above_or_below == "below":
        return layer_idx - 1
    raise ValueError(f'{above_or_below} needs to be either "above" or "below".')


def generate_layers_to_above_or_below(
    ml_graph: MultiLayeredGraph, max_iterations: int, one_sided: bool
) -> list[tuple[int, Literal["above"] | Literal["below"]]]:
    layers_to_above_below: list[tuple[int, Literal["above"] | Literal["below"]]] = [
        (layer_idx, "below") for layer_idx in range(1, ml_graph.layer_count)
    ]
    if not one_sided:
        layers_to_above_below.extend(
            (layer_idx, "above")
            for layer_idx in range(ml_graph.layer_count - 2, -1, -1)
        )

        layers_to_above_below *= max_iterations

    return layers_to_above_below
