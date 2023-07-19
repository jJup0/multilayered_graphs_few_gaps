from functools import cache
from typing import Literal

from crossings.calculate_crossings import crossings_for_node_pair
from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph


def _get_crossings_for_vnodes_in_gaps(
    ml_graph: MultiLayeredGraph,
    above_or_below: Literal["above"] | Literal["below"],
    ordered_real_nodes: list[MLGNode],
    virtual_nodes: list[MLGNode],
) -> dict[MLGNode, list[int]]:
    r"""For each virtual node in a given list, try each position it can be placed
    in a group of real nodes and count crossings.

    Example:
    Numbers represent real nodes, X represents virtual node, ascii art represents edges:

        1       2___   _3   X
        |\____  |   \_/ |  /
        |     \ | __/ \_|_/
        4       5       6

    A virtual node can be placed either
    - before real node `1` or
    - between `1` and `2` or
    - between `2` and `3` or
    - after `3` (where it currently is)

    For each placement, calculate the crossings, e.g. placed before node `1` it
    will have 5 crossings, so result[X][0] == 5.
    Repeat for each placement and we get result[X] == [5, 3, 2, 0]

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
                ml_graph, real_node, vnode, above_or_below
            )
            curr_sum += cross_count
            right_crosses.append(curr_sum)

        left_crosses: list[int] = [0]
        curr_sum = 0
        for real_node in reversed(ordered_real_nodes):
            cross_count = crossings_for_node_pair(
                ml_graph, vnode, real_node, above_or_below
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
    ordered_virtual_nodes: list[MLGNode],
    gaps: int,
) -> tuple[int, list[int]]:
    """Given a fixed ordering of real nodes, determine the best distribution of virtual nodes into gaps.

    Args:
        ml_graph (MultiLayeredGraph): Graph in which to find gaps.
        above_or_below (Literal[&quot;above&quot;] | Literal[&quot;below&quot;]):
          Whether to consider nodes above or below the current layer as "neighbors"
        ordered_real_nodes (list[MLGNode]): List of real nodes. Order of elements is decisive.
        virtual_nodes (list[MLGNode]): List of virtual nodes. Order of elements is decisive.
        gaps (int): Amount of gaps virtual nodes may be placed in.

    Returns:
        tuple[int, list[int]]: _description_
    """
    assert gaps > 0

    virtual_node_to_crossings_in_gap = _get_crossings_for_vnodes_in_gaps(
        ml_graph,
        above_or_below,
        ordered_real_nodes=ordered_real_nodes,
        virtual_nodes=ordered_virtual_nodes,
    )

    @cache
    def one_gap_crossings(from_vnode_idx: int, to_vnode_idx: int, gap_idx: int) -> int:
        nonlocal virtual_node_to_crossings_in_gap, ordered_virtual_nodes
        # validation
        assert gap_idx >= 0
        if to_vnode_idx < from_vnode_idx:
            return 0

        last_vnode = ordered_virtual_nodes[to_vnode_idx]
        res = virtual_node_to_crossings_in_gap[last_vnode][gap_idx]
        if from_vnode_idx == to_vnode_idx:
            return res
        return res + one_gap_crossings(from_vnode_idx, to_vnode_idx - 1, gap_idx)

    @cache
    def find_crossings(
        curr_gaps: int, upto_virtual_idx: int, gap_idx: int
    ) -> tuple[int, list[int]]:
        assert curr_gaps > 0
        nonlocal ml_graph, virtual_node_to_crossings_in_gap

        if gap_idx == 0:
            # BASE CASE
            # only one gap allowed, and only first gap allowed. Placing the
            # first upto_virtual_idx nodes into first gap is computed by
            # one_gap_crossings
            crossing_res = one_gap_crossings(0, upto_virtual_idx, 0)
            # the distribution is simply the first upto_virtual_idx + 1
            # virtual nodes are placed in gap 0
            distribution_res = [0] * (upto_virtual_idx + 1)
        elif curr_gaps == 1:
            # find best placement for only being allowed to place in previous gap
            crossing_res, distribution_res = find_crossings(
                1, upto_virtual_idx, gap_idx - 1
            )
            # compare to crossings if all nodes are placed in current
            # rightmost allowed gap
            crossings_in_bunch = one_gap_crossings(0, upto_virtual_idx, gap_idx)
            if crossings_in_bunch < crossing_res:
                # if placing first (upto_virtual_idx + 1) virtual nodes into
                # rightmost allowed gap produces fewer crossings, update result
                crossing_res = crossings_in_bunch
                distribution_res = [gap_idx] * (upto_virtual_idx + 1)
        elif upto_virtual_idx == 0:
            # more than one gap allowed, but only one node, so fetch result for when only one gap allowed
            crossing_res, distribution_res = find_crossings(1, 0, gap_idx)
        else:
            # default result is packing all into one less gap
            crossing_res, distribution_res = find_crossings(
                curr_gaps - 1, upto_virtual_idx, gap_idx
            )
            # more than one gap allowed, find best placement for `gaps`-1 gaps,
            # by iterating from i:= 0 to upto_virtual_idx-1 for the virtual node index
            # # and from j := 0 to gap_idx-1 for the maximum allowed gap index
            # for each iteration, place the remaining upto_virtual_idx - i nodes in the jth gap
            for prev_vnode_idx in range(upto_virtual_idx):
                # find best placement for one fewer gap with the virtual
                # nodes, and few gap-places to use
                with_one_fewer_gaps_crossings, prev_distribution_res = find_crossings(
                    curr_gaps - 1, prev_vnode_idx, gap_idx - 1
                )
                # add crossings of remaining nodes in the currently right-most allowed gap
                crossings_for_new_bunch = one_gap_crossings(
                    prev_vnode_idx + 1, upto_virtual_idx, gap_idx
                )
                with_one_fewer_gaps_crossings += crossings_for_new_bunch

                # if the total crossings are less, update the best result
                if with_one_fewer_gaps_crossings < crossing_res:
                    crossing_res = with_one_fewer_gaps_crossings
                    # make a copy of the previous distribution and place the
                    # remaining virtual nodes in one gap to the right of the
                    # currently rightmost allowed gap-place
                    distribution_res = prev_distribution_res + [gap_idx] * (
                        upto_virtual_idx - prev_vnode_idx
                    )

        # validation, algorithm should be correct now, this may be removed
        assert (
            len(distribution_res) == upto_virtual_idx + 1
        ), f"WARNING: {curr_gaps=}, {upto_virtual_idx=}, {gap_idx=}: {len(distribution_res)=} != {upto_virtual_idx+1=}"

        return crossing_res, distribution_res

    if gaps > len(ordered_real_nodes):
        # no need to check gap count higher than possible gaps in layer
        gaps = len(ordered_real_nodes)
    return find_crossings(gaps, len(ordered_virtual_nodes) - 1, len(ordered_real_nodes))


def k_gaps_sort_layer(
    ml_graph: MultiLayeredGraph,
    layers_to_ordered_real_nodes: list[list[MLGNode]],
    layer_idx: int,
    above_or_below: Literal["above"] | Literal["below"],
    gaps: int,
):
    """
    Given a layer, a gap count and the real nodes of that layer with a predetermined
    order, find the best placement of virtual nodes.
    """

    sorted_virtual_nodes = sorted(
        (n for n in ml_graph.layers_to_nodes[layer_idx] if n.is_virtual),
        key=lambda node: _virtual_node_to_neighbor_position_sorting_func(
            ml_graph, node, above_or_below
        ),
    )
    curr_layer_real_nodes = layers_to_ordered_real_nodes[layer_idx]
    _, vnode_placement = _find_optimal_gaps(
        ml_graph,
        above_or_below,
        curr_layer_real_nodes,
        sorted_virtual_nodes,
        gaps,
    )

    final_node_order: list[MLGNode] = []
    curr_gap_idx = 0
    for gap_idx, vnode in zip(vnode_placement, sorted_virtual_nodes, strict=True):
        for curr_gap_idx in range(curr_gap_idx, gap_idx):
            final_node_order.append(curr_layer_real_nodes[curr_gap_idx])
        curr_gap_idx = gap_idx
        final_node_order.append(vnode)
    for curr_gap_idx in range(curr_gap_idx, len(curr_layer_real_nodes)):
        final_node_order.append(curr_layer_real_nodes[curr_gap_idx])

    if set(ml_graph.layers_to_nodes[layer_idx]) != set(final_node_order):
        print(f"{set(ml_graph.layers_to_nodes[layer_idx])} != {set(final_node_order)}")
        assert False

    ml_graph.layers_to_nodes[layer_idx] = final_node_order


def _virtual_node_to_neighbor_position_sorting_func(
    ml_graph: MultiLayeredGraph,
    vnode: MLGNode,
    neighbors_are_above_or_below: Literal["above"] | Literal["below"],
) -> int:
    assert vnode.is_virtual
    neighbors = (
        ml_graph.nodes_to_in_edges[vnode]
        if neighbors_are_above_or_below == "below"
        else ml_graph.nodes_to_out_edges[vnode]
    )
    assert len(neighbors) == 1

    neighbor = next(iter(neighbors))
    neighbor_layer_idx = vnode.layer - 1 if "below" else vnode.layer + 1
    return ml_graph.layers_to_nodes[neighbor_layer_idx].index(neighbor)
