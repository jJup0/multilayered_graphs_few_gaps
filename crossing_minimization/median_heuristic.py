import statistics
from typing import Literal

from crossing_minimization.utils import (
    DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    lgraph_sorting_algorithm,
    sorting_parameter_check,
)
from crossings.calculate_crossings import crossings_uv_vu
from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

PSEUDO_SORT_DISPLACE_VALUE = 1_000


@lgraph_sorting_algorithm
def few_gaps_median_sort_naive(
    ml_graph: MultiLayeredGraph,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    one_sided: bool = False,
) -> None:
    """Sorts nodes in multilayered graph according to median heuristic.

    Modified in place. Naive virtual node placement.
    Args:
        ml_graph: Graph on which to apply sorting.
    """

    def _sort_layer(
        _layer_idx: int,
        _prev_layer_indices: dict[MLGNode, int],
        node_to_neighbors: dict[MLGNode, set[MLGNode]],
    ):
        nonlocal ml_graph
        layer_before_sorting = ml_graph.layers_to_nodes[_layer_idx][:]
        real_node_median_median = get_real_node_median_median(
            _nodes_at_layer=layer_before_sorting,
            _node_to_neighbors=node_to_neighbors,
            _prev_layer_indices=_prev_layer_indices,
        )

        ml_graph.layers_to_nodes[_layer_idx] = sorted(
            ml_graph.layers_to_nodes[_layer_idx],
            key=lambda node: _get_median_sort_val_naive(
                node=node,
                neighbors=node_to_neighbors[node],
                prev_layer_indices=_prev_layer_indices,
                real_node_median_median=real_node_median_median,
                layer_before_sorting=layer_before_sorting,
            ),
        )

    sorting_parameter_check(
        ml_graph, max_iterations=max_iterations, one_sided=one_sided
    )

    node_to_in_neighbors = ml_graph.nodes_to_in_edges
    node_to_out_neighbors = ml_graph.nodes_to_out_edges
    for _ in range(max_iterations):
        for layer_idx in range(1, ml_graph.layer_count):
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer_idx - 1)
            _sort_layer(layer_idx, prev_layer_indices, node_to_in_neighbors)
        if one_sided:
            return
        for layer_idx in range(ml_graph.layer_count - 2, -1, -1):
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer_idx + 1)
            _sort_layer(layer_idx, prev_layer_indices, node_to_out_neighbors)


@lgraph_sorting_algorithm
def few_gaps_median_sort_improved(
    ml_graph: MultiLayeredGraph,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    one_sided: bool = False,
) -> None:
    """Sorts nodes in multilayered graph according to median heuristic.

    Modified in place. Improved virtual node placement.
    Args:
        ml_graph: Graph on which to apply sorting.
        max_iterations:
          Amount of "up" and "down" cycles to make for sorting. Defaults to 3.
        one_sided:
          Whether to only do one sided crossing minimization or not.
          Defaults to False.
    """

    def _sort_layer(
        _layer_idx: int,
        _prev_layer_indices: dict[MLGNode, int],
        _node_to_neighbors: dict[MLGNode, set[MLGNode]],
        neighbors_above_or_below: Literal["above"] | Literal["below"],
    ):
        nonlocal ml_graph
        # real_node_bary_median = _get_real_node_median_median(
        #     _ml_graph,
        #     _ml_graph.layers_to_nodes[_layer_idx],
        #     _node_to_neighbors,
        #     _prev_layer_indices,
        # )
        layer_before_sorting = ml_graph.layers_to_nodes[_layer_idx][:]
        ml_graph.layers_to_nodes[_layer_idx].sort(
            key=lambda node: _get_median_sort_val_improved(
                layer_before_sorting=layer_before_sorting,
                node=node,
                neighbors=_node_to_neighbors[node],
                prev_layer_indices=_prev_layer_indices,
                ml_graph=ml_graph,
                # real_node_bary_median,
                neighbors_above_or_below=neighbors_above_or_below,
            )
        )

    sorting_parameter_check(
        ml_graph, max_iterations=max_iterations, one_sided=one_sided
    )

    node_to_in_neighbors = ml_graph.nodes_to_in_edges
    node_to_out_neighbors = ml_graph.nodes_to_out_edges

    # arbitrary loop count,TODO pass as parameter?
    for _ in range(max_iterations):
        for layer in range(1, ml_graph.layer_count):
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer - 1)
            _sort_layer(layer, prev_layer_indices, node_to_in_neighbors, "below")
        if one_sided:
            return
        for layer in range(ml_graph.layer_count - 2, -1, -1):
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer + 1)
            _sort_layer(layer, prev_layer_indices, node_to_out_neighbors, "above")


def get_real_node_median_median(
    _nodes_at_layer: list[MLGNode],
    _node_to_neighbors: dict[MLGNode, set[MLGNode]],
    _prev_layer_indices: dict[MLGNode, int],
):
    """Calculate median of medians of real nodes at given layer.

    :param _nodes_at_layer: Nodes of a layer for which to calculate.
    :param _node_to_neighbors: Mapping from nodes to their neighbors.
    :param _prev_layer_indices: Mapping from nodes at "previous" layer to position in layer.
    :return: Median of medians of real nodes at given layer.
    """
    real_node_medians = (
        _unweighted_median(
            layer_before_sorting=_nodes_at_layer,
            node=node,
            neighbors=_node_to_neighbors[node],
            prev_layer_indices=_prev_layer_indices,
        )
        for node in _nodes_at_layer
        if not node.is_virtual
    )

    # should have at least one real node in layer, so list should not be empty
    return statistics.median(real_node_medians)


def _unweighted_median(
    layer_before_sorting: list[MLGNode],
    node: MLGNode,
    neighbors: set[MLGNode],
    prev_layer_indices: dict[MLGNode, int],
) -> float:
    """Calculates "true" median of a node by using its neighbors positions.

    :param layer_before_sorting: Layer with all nodes the way it was before sorting.
    :param node: Node for which to calculate median
    :param neighbors: Set of neighbors.
    :param prev_layer_indices: Dictionary mapping nodes of previous layer to indices.
    :return: Median value of the given node.
    """
    if len(neighbors) == 0:
        # TODO check if this is a viable strategy
        return layer_before_sorting.index(node)

    median = statistics.median(prev_layer_indices[node] for node in neighbors)
    node.text_info = f"median {median}"
    return median


def _get_median_sort_val_naive(
    layer_before_sorting: list[MLGNode],
    node: MLGNode,
    neighbors: set[MLGNode],
    prev_layer_indices: dict[MLGNode, int],
    real_node_median_median: float,
) -> float:
    """Calculate an arbitrary value used to sort node of a layer, using median heuristic.

    :param layer_before_sorting: Layer with nodes before sorting.
    :param node: Node for which to calculate sort value.
    :param neighbors: Neighbors of node.
    :param prev_layer_indices: Mapping from nodes at "previous" layer to position within that layer.
    :param real_node_median_median: Median of real node medians.
    :return: Sort value for node within layer.
    """
    median = _unweighted_median(
        layer_before_sorting, node, neighbors, prev_layer_indices
    )

    if node.is_virtual:
        if median > real_node_median_median:
            median += PSEUDO_SORT_DISPLACE_VALUE
        else:
            median -= PSEUDO_SORT_DISPLACE_VALUE
    node.text_info = f"median {median}"
    return median


def _get_median_sort_val_improved(
    layer_before_sorting: list[MLGNode],
    node: MLGNode,
    neighbors: set[MLGNode],
    prev_layer_indices: dict[MLGNode, int],
    ml_graph: MultiLayeredGraph,
    neighbors_above_or_below: Literal["above"] | Literal["below"],
) -> float:
    median = _unweighted_median(
        layer_before_sorting, node, neighbors, prev_layer_indices
    )

    if node.is_virtual:
        real_node_crossings_if_left = 0
        real_node_crossings_if_right = 0
        for other_node in layer_before_sorting:
            if other_node.is_virtual:
                continue
            left_crossings, right_crossings = crossings_uv_vu(
                ml_graph, node, other_node, neighbors_above_or_below
            )
            real_node_crossings_if_left += left_crossings
            real_node_crossings_if_right += right_crossings

        if real_node_crossings_if_left < real_node_crossings_if_right:
            median -= PSEUDO_SORT_DISPLACE_VALUE
        else:
            median += PSEUDO_SORT_DISPLACE_VALUE

    node.text_info = f"median {median}"
    return median
