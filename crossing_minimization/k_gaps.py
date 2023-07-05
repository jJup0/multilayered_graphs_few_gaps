from functools import cache
from typing import Literal

from crossing_minimization.utils import (
    DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    lgraph_sorting_algorithm,
    sorting_parameter_check,
)
from crossings.calculate_crossings import crossings_for_node_pair
from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph


def _get_crossings_for_vnodes_in_gaps(
    ml_graph: MultiLayeredGraph,
    above_or_below: Literal["above"] | Literal["below"],
    ordered_real_nodes: list[MLGNode],
    virtual_nodes: list[MLGNode],
) -> dict[MLGNode, list[int]]:
    r"""For each virtual node in a given list, try each position it can be placed
    in in a group of real nodes and count crossings.

    Example:
    Numbers represent real nodes, X represents virtual node, ascii art represents crossings:

        1       2___    3   X
        |\____  |  _\__/|  /
        |     \ | /  \__|_/
        4       5       6

    A virtual node can be placed either
    - before real node `1` or
    - between `1` and `2` or
    - between `2` and `3` or
    - after `3` (where it currently is)

    For each placement






    Args:
        ml_graph (MultiLayeredGraph): Multilayered graph in which to count crossings
        above_or_below (Literal["above"] | Literal["below"]): Whether to consider nodes above or below as neighbors.
        ordered_real_nodes (list[MLGNode]): Real nodes in a predetermined order.
        virtual_nodes (list[MLGNode]): List of virtual nodes for which to determine crossings with real nodes.

    Returns:
        dict[MLGNode, list[int]]: A dictionary mapping virtual nodes to crossing counts for each gap in the real nodes.
    """
    virtual_node_to_crossings_in_gap: dict[MLGNode, list[int]] = {}

    for vnode in virtual_nodes:
        # total_crosses = [0] * (len(ordered_real_nodes) + 1)
        right_crosses: list[int] = [0]
        curr_sum = 0
        for real_node in ordered_real_nodes:
            cross_count = crossings_for_node_pair(
                ml_graph, vnode, real_node, above_or_below
            )
            curr_sum += cross_count
            right_crosses.append(curr_sum)

        left_crosses: list[int] = [0]
        curr_sum = 0
        for real_node in reversed(ordered_real_nodes):
            cross_count = crossings_for_node_pair(
                ml_graph, real_node, vnode, above_or_below
            )
            curr_sum += cross_count
            left_crosses.append(curr_sum)
        left_crosses = left_crosses[::-1]

        total_crosses = [l + r for l, r in zip(left_crosses, right_crosses)]
        virtual_node_to_crossings_in_gap[vnode] = total_crosses
    return virtual_node_to_crossings_in_gap


def _find_optimal_gaps(
    ml_graph: MultiLayeredGraph,
    above_or_below: Literal["above"] | Literal["below"],
    ordered_real_nodes: list[MLGNode],
    virtual_nodes: list[MLGNode],
    gaps: int,
):
    INFINITY = 1_000_000_000
    dp = [
        [[INFINITY] * (len(ordered_real_nodes)) for _ in range(len(virtual_nodes))]
        for _ in range(gaps + 1)
    ]

    virtual_node_to_crossings_in_gap = _get_crossings_for_vnodes_in_gaps(
        ml_graph,
        above_or_below,
        ordered_real_nodes=ordered_real_nodes,
        virtual_nodes=virtual_nodes,
    )

    @cache
    def one_gap_crossings(from_vnode_idx: int, to_vnode_idx: int, gap_idx: int) -> int:
        nonlocal virtual_node_to_crossings_in_gap, virtual_nodes
        last_vnode = virtual_nodes[to_vnode_idx]
        res = virtual_node_to_crossings_in_gap[last_vnode][gap_idx]
        if from_vnode_idx == to_vnode_idx:
            return res
        return res + one_gap_crossings(from_vnode_idx, to_vnode_idx - 1, gap_idx)

    def find_crossings(gaps: int, upto_virtual_idx: int, gap_idx: int) -> int:
        nonlocal dp, ml_graph
        pre_computed = dp[gaps][upto_virtual_idx][gap_idx]
        if pre_computed != INFINITY:
            return pre_computed

        if gaps == 1:
            vnode = virtual_nodes[upto_virtual_idx]
            res = virtual_node_to_crossings_in_gap[vnode][gap_idx]
            if upto_virtual_idx > 0:
                res += find_crossings(gaps, upto_virtual_idx - 1, gap_idx)
        else:
            res = INFINITY
            for prev_vnode_idx in range(upto_virtual_idx):
                for prev_gap_idx in range(gap_idx):
                    with_one_fewer_gaps = find_crossings(
                        gaps - 1, prev_vnode_idx, prev_gap_idx
                    )
                    with_one_fewer_gaps += one_gap_crossings(
                        prev_vnode_idx, upto_virtual_idx, prev_gap_idx + 1
                    )
                    res = min(res, with_one_fewer_gaps)

        dp[gaps][upto_virtual_idx][gap_idx] = res
        return res

    return find_crossings(gaps, len(virtual_nodes) - 1, len(ordered_real_nodes))


@lgraph_sorting_algorithm
def k_gaps_barycenter(
    ml_graph: MultiLayeredGraph,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    one_sided: bool = False,
):
    sorting_parameter_check(
        ml_graph, max_iterations=max_iterations, one_sided=one_sided
    )

    def get_sort_value_realnode(
        neighbors: set[MLGNode], prev_layer_indices: dict[MLGNode, int]
    ) -> float:
        neighbor_count = len(neighbors)
        if neighbor_count == 0:
            # TODO check if this is a viable strategy
            return 0
        barycenter = (
            sum(prev_layer_indices[node] for node in neighbors) / neighbor_count
        )
        return barycenter

    node_to_in_neighbors = ml_graph.nodes_to_in_edges
    node_to_out_neighbors = ml_graph.nodes_to_out_edges

    for _ in range(max_iterations):
        for layer_idx in range(1, ml_graph.layer_count):
            layer_real_nodes = [
                n for n in ml_graph.layers_to_nodes[layer_idx] if not n.is_virtual
            ]
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer_idx - 1)
            layer_real_nodes.sort(
                key=lambda node: get_sort_value_realnode(
                    node_to_in_neighbors[node],
                    prev_layer_indices,
                )
            )

            # layer_real_nodes = [n for n in ml_graph.layers_to_nodes[layer_idx] if not n.is_virtual]

        if one_sided:
            return
        for layer_idx in range(ml_graph.layer_count - 2, -1, -1):
            ...
