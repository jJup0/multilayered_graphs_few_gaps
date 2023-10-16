import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, NoReturn, TypeAlias, TypeVar

from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def above_below_opposite(above_or_below: Above_or_below_T):
    if above_or_below == "above":
        return "below"
    return "above"


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
        """Given a multi-layered graph, untangle long edges.

        Very naive approach, can be improved.

        Args:
            ml_graph (MultiLayeredGraph): The multi-layered graph for which to untangle edges.
        """
        crossing_count_before = ml_graph.get_total_crossings()

        real_nodes = [n for n in ml_graph.all_nodes_as_list() if not n.is_virtual]
        for real_node in real_nodes:
            adjacent_virtual_nodes = [
                n for n in ml_graph.nodes_to_out_edges[real_node] if n.is_virtual
            ]
            made_switch = True
            debug_ever_made_switch = False
            # if two long edges are swapped, iterate again incase this causes crossings with other edges
            while made_switch:
                made_switch = False
                # untangle each pair of long edges incident to the current node
                for i, vnode1 in enumerate(adjacent_virtual_nodes):
                    for j in range(i):
                        made_switch = made_switch or cls.untangle_long_edges(
                            ml_graph, vnode1, adjacent_virtual_nodes[j]
                        )
                        debug_ever_made_switch |= made_switch

            if debug_ever_made_switch:
                logger.info("Node %s has long edges switched", real_node)

        assert ml_graph.get_total_crossings() <= crossing_count_before

    @classmethod
    def untangle_long_edges(
        cls, ml_graph: MultiLayeredGraph, vnode1: MLGNode, vnode2: MLGNode
    ) -> bool:
        """Given two virtual nodes adjacent to the same real nodes, untangle the long edges that they are a part of.

        Very naive and unoptimized implementation.

        Args:
            ml_graph (MultiLayeredGraph): The graph to which these nodes belong
            vnode1 (MLGNode): A virtual node directly incident to a real node in the layer below.
            vnode2 (MLGNode): A different virtual node directly incident to a real node in the layer below.
        """
        # nodes must be virtual, and directly connected to real node
        assert vnode1.is_virtual and vnode2.is_virtual
        assert ml_graph.nodes_to_in_edges[vnode1] == ml_graph.nodes_to_in_edges[vnode2]

        # traverse up the two long edges until a real node is reached
        prev_vnode1, prev_vnode2 = vnode1, vnode2  # for type checker
        while vnode1.is_virtual and vnode2.is_virtual:
            prev_vnode1, prev_vnode2 = vnode1, vnode2
            vnode1 = next(iter(ml_graph.nodes_to_out_edges[vnode1]))
            vnode2 = next(iter(ml_graph.nodes_to_out_edges[vnode2]))

        last_layer = ml_graph.layers_to_nodes[vnode1.layer]
        vnode1_left_of_vnode2 = last_layer.index(vnode1) < (last_layer.index(vnode2))
        vnode1, vnode2 = prev_vnode1, prev_vnode2
        made_switch = False
        # traverse back down edges to untangle them
        while vnode1.is_virtual:
            assert vnode2.is_virtual
            last_virtual_layer = ml_graph.layers_to_nodes[vnode1.layer]
            vnode1_idx = last_virtual_layer.index(vnode1)
            vnode2_idx = last_virtual_layer.index(vnode2)
            if (vnode1_idx < vnode2_idx) != vnode1_left_of_vnode2:
                # nodes are in order opposite to what they should be, switch them
                last_virtual_layer[vnode1_idx] = vnode2
                last_virtual_layer[vnode2_idx] = vnode1
                made_switch = True

            vnode1 = next(iter(ml_graph.nodes_to_in_edges[vnode1]))
            vnode2 = next(iter(ml_graph.nodes_to_in_edges[vnode2]))

        # both pointers should now point to real node to which the long edges are adjacent
        assert vnode1 is vnode2
        return made_switch


def thesis_side_gaps(
    ml_graph: MultiLayeredGraph,
    *,
    max_iterations: int,
    only_one_up_iteration: bool,
    get_median_or_barycenter: Callable[
        [MultiLayeredGraph, MLGNode, set[MLGNode], dict[MLGNode, int]], float
    ],
) -> None:
    """
    Sorts MultiLayeredGraph using barycenter heuristic, placing nodes
    depending on minimal crossings.

    O(max_iterations * (O(|V_i^{vt}| * |E_i^r|))) time complexity
    """

    for layer_idx, above_or_below in generate_layers_to_above_or_below(
        ml_graph, max_iterations, only_one_up_iteration
    ):
        neighbor_layer_idx = get_layer_idx_above_or_below(layer_idx, above_or_below)

        node_to_neighbors = get_graph_neighbors_from_above_or_below(
            ml_graph, above_or_below
        )

        # O(n)
        prev_layer_indices = ml_graph.nodes_to_indices_at_layer(neighbor_layer_idx)

        curr_layer = ml_graph.layers_to_nodes[layer_idx]
        # sort real and virtual nodes
        # O(|E_i| + O(|V_i| * log(V_i)))
        # in our case either medians or barycenters
        medians_or_barycenters = {
            node: get_median_or_barycenter(
                ml_graph, node, node_to_neighbors[node], prev_layer_indices
            )
            for node in curr_layer
        }
        real_nodes_sorted = sorted(
            (n for n in curr_layer if not n.is_virtual),
            key=lambda node: medians_or_barycenters[node],
        )
        virtual_nodes_sorted = sorted(
            (n for n in curr_layer if n.is_virtual),
            key=lambda node: medians_or_barycenters[node],
        )

        # neighbor prefix
        # O(|E_j|)
        neighbor_layer_degree_prefix_sum = _get_prev_layer_edges_prefix_sum(
            ml_graph, above_or_below, neighbor_layer_idx
        )
        neighbor_layer_total_out_edges = neighbor_layer_degree_prefix_sum[-1]

        # find split index
        # O(|V_i^{vt}|)
        vnode_i = 0
        for vnode_i, vnode in enumerate(virtual_nodes_sorted):
            # virtual node only has one neighbor,
            vnode_neighbor_pos = sum(
                prev_layer_indices[node] for node in node_to_neighbors[vnode]
            )
            # a virtual node has more crossings when placed on the left, if
            # the accumulated edge count up until its neighbor is less than
            # half of all outgoing edges of that layer
            if (
                neighbor_layer_degree_prefix_sum[vnode_neighbor_pos]
                > neighbor_layer_total_out_edges // 2
            ):
                break

        # O(|V_i|)
        ml_graph.layers_to_nodes[layer_idx] = (
            virtual_nodes_sorted[:vnode_i]
            + real_nodes_sorted
            + virtual_nodes_sorted[vnode_i:]
        )


def _get_prev_layer_edges_prefix_sum(
    ml_graph: MultiLayeredGraph,
    above_or_below: Above_or_below_T,
    neighbor_layer_idx: int,
):
    neighbor_layer = ml_graph.layers_to_nodes[neighbor_layer_idx]
    neighbor_layer_degree_prefix_sum = [0]
    neighbor_to_neighbors = get_graph_neighbors_from_above_or_below(
        ml_graph, above_below_opposite(above_or_below)
    )
    for neighbor_layer_node in neighbor_layer:
        neighbor_layer_degree_prefix_sum.append(
            neighbor_layer_degree_prefix_sum[-1]
            + len(neighbor_to_neighbors[neighbor_layer_node])
        )

    return neighbor_layer_degree_prefix_sum
