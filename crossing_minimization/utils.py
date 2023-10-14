from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, NoReturn, TypeAlias, TypeVar

from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

T = TypeVar("T")
Above_or_below_T: TypeAlias = Literal["above"] | Literal["below"]

DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION = 3


def lgraph_sorting_algorithm(func: T) -> T:
    """Decorator for function which sorts a layered graph.

    Function head should look like this:
    (ml_graph: MultiLayeredGraph, *, max_iterations: int = 3, only_one_up_iteration: bool = False) -> None

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


def get_layer_idx_above_or_below(layer_idx: int, above_or_below: Above_or_below_T):
    if above_or_below == "above":
        return layer_idx + 1
    if above_or_below == "below":
        return layer_idx - 1
    raise ValueError(f'Parameter must be "above" or "below", got: {above_or_below}')


def get_graph_neighbors_from_above_or_below(
    ml_graph: MultiLayeredGraph, above_or_below: Above_or_below_T
):
    if above_or_below == "below":
        return ml_graph.nodes_to_in_edges
    if above_or_below == "above":
        return ml_graph.nodes_to_out_edges
    raise ValueError(f'Parameter must be "above" or "below", got: {above_or_below}')


def generate_layers_to_above_or_below(
    ml_graph: MultiLayeredGraph, max_iterations: int, only_one_up_iteration: bool
) -> list[tuple[int, Above_or_below_T]]:
    layers_to_above_below: list[tuple[int, Above_or_below_T]] = [
        (layer_idx, "below") for layer_idx in range(1, ml_graph.layer_count)
    ]
    if not only_one_up_iteration:
        up_iter = layers_to_above_below.copy()
        layers_to_above_below.extend(
            (layer_idx, "above")
            for layer_idx in range(ml_graph.layer_count - 2, -1, -1)
        )

        layers_to_above_below *= max_iterations
        layers_to_above_below.extend(up_iter)

    return layers_to_above_below


class GraphSorter(ABC):
    algorithm_name = "<set algorithm name>"

    def __str__(self) -> str:
        return self.algorithm_name

    @classmethod
    def sort_graph(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int,
        only_one_up_iteration: bool,
        side_gaps_only: bool,
        max_gaps: int,
    ) -> None:
        """Sorts the nodes of a multilayered graph.

        Args:
            ml_graph (MultiLayeredGraph): The graph to sort.
            max_iterations (int): Maximum up & down iterations for sorting layers.
            only_one_up_iteration (bool): Whether to only do one up iteration or not.
            side_gaps_only (bool): Whether gaps are only allowed to be on the side or not.
            max_gaps (int): Maximum amount of gaps allowed.
        """
        cls._sorting_parameter_check(
            ml_graph,
            max_iterations=max_iterations,
            only_one_up_iteration=only_one_up_iteration,
            side_gaps_only=side_gaps_only,
            max_gaps=max_gaps,
        )
        return cls._sort_graph(
            ml_graph,
            max_iterations=max_iterations,
            only_one_up_iteration=only_one_up_iteration,
            side_gaps_only=side_gaps_only,
            max_gaps=max_gaps,
        )

    @classmethod
    @abstractmethod
    def _sort_graph(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int,
        only_one_up_iteration: bool,
        side_gaps_only: bool,
        max_gaps: int,
    ) -> None:
        """Implementation detail of sort graph. Should be overriden"""
        raise NotImplementedError("not implemented")

    @classmethod
    def _sorting_parameter_check(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int,
        only_one_up_iteration: bool,
        side_gaps_only: bool,
        max_gaps: int,
    ):
        # input parameter validation
        if not (only_one_up_iteration is True or only_one_up_iteration is False):
            raise ValueError(
                f'one_side must be true or false, received "{only_one_up_iteration}"'
            )
        # if only_one_up_iteration:
        #     if ml_graph.layer_count != 2:
        #         raise ValueError(
        #             f"One-sided crossing minimization can only be performed on graphs with exactly 2 layers."
        #             f"Input graph has {ml_graph.layer_count} layers."
        #         )
        elif max_iterations <= 0:
            raise ValueError(f"iterations must be > 0, received {max_iterations}")

        if side_gaps_only:
            if max_gaps != 2:
                raise ValueError(
                    f"If side gaps only, max_gaps must be equal to 2. Got {max_gaps}"
                )
        else:
            if max_gaps <= 0:
                raise ValueError(
                    f"Maximum allowed gaps must be greater than 0. Got {max_gaps}"
                )

    @classmethod
    def sort_2layered_sidegaps_onesided(
        cls,
        ml_graph: MultiLayeredGraph,
    ) -> None:
        assert ml_graph.layer_count == 2
        return cls._sort_graph(
            ml_graph,
            side_gaps_only=True,
            only_one_up_iteration=True,
            max_gaps=2,
            max_iterations=1,
        )

    @classmethod
    def sort_2layered_sidegaps_twosided(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    ) -> None:
        assert ml_graph.layer_count == 2
        return cls._sort_graph(
            ml_graph,
            side_gaps_only=True,
            only_one_up_iteration=False,
            max_gaps=2,
            max_iterations=max_iterations,
        )

    @classmethod
    def sort_2layered_kgaps_onesided(
        cls, ml_graph: MultiLayeredGraph, *, max_gaps: int
    ) -> None:
        assert ml_graph.layer_count == 2
        return cls._sort_graph(
            ml_graph,
            side_gaps_only=True,
            only_one_up_iteration=False,
            max_gaps=max_gaps,
            max_iterations=1,
        )

    @classmethod
    def sort_2layered_kgaps_twosided(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_gaps: int,
        max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    ):
        assert ml_graph.layer_count == 2
        return cls._sort_graph(
            ml_graph,
            side_gaps_only=False,
            max_gaps=max_gaps,
            only_one_up_iteration=False,
            max_iterations=max_iterations,
        )

    @classmethod
    def sort_multilayered_sidegaps(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    ):
        return cls._sort_graph(
            ml_graph,
            side_gaps_only=True,
            max_gaps=2,
            only_one_up_iteration=False,
            max_iterations=max_iterations,
        )

    @classmethod
    def sort_multilayered_kgaps(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_gaps: int,
        max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    ):
        return cls._sort_graph(
            ml_graph,
            side_gaps_only=False,
            max_gaps=max_gaps,
            only_one_up_iteration=False,
            max_iterations=max_iterations,
        )

    @classmethod
    def rearrange_trivial_long_edges(cls, ml_graph: MultiLayeredGraph):
        crossing_count_before = ml_graph.get_total_crossings()

        # for layer_idx in range(ml_graph.layer_count - 1, -1, -1):

        # very naive, can be improved
        for layer_idx in range(ml_graph.layer_count):
            real_nodes_in_layer = [
                n for n in ml_graph.layers_to_nodes[layer_idx] if not n.is_virtual
            ]
            for real_node in real_nodes_in_layer:
                adjacent_virtual_nodes = [
                    n
                    for n in ml_graph.nodes_to_out_edges[real_node]
                    if n is n.is_virtual
                ]
                for i, vnode1 in enumerate(adjacent_virtual_nodes):
                    for j in range(i):
                        cls.untangle_long_edges(
                            ml_graph, vnode1, adjacent_virtual_nodes[j]
                        )
                # ml_graph.layers_to_nodes[layer_idx]

    @classmethod
    def untangle_long_edges(
        cls, ml_graph: MultiLayeredGraph, vnode1: MLGNode, vnode2: MLGNode
    ):
        # nodes must be virtual, and directly connected to real node
        assert vnode1.is_virtual and vnode2.is_virtual
        assert ml_graph.nodes_to_in_edges[vnode1] == ml_graph.nodes_to_in_edges[vnode2]
        curr_layer = ml_graph.layers_to_nodes[vnode1.layer]
        assert curr_layer.index(vnode1) < curr_layer.index(vnode2)

        while vnode1.is_virtual and vnode2.is_virtual:
            vnode1 = next(iter(ml_graph.nodes_to_out_edges[vnode1]))
            vnode2 = next(iter(ml_graph.nodes_to_out_edges[vnode2]))
            curr_layer = ml_graph.layers_to_nodes[vnode1.layer]
            if curr_layer.index(vnode1) > curr_layer.index(vnode2):
