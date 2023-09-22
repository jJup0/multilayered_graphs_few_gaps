from typing import Literal

from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph


def edges_cross(
    edge1_source_idx: int,
    edge1_destination_idx: int,
    edge2_source_pos: int,
    edge2_destination_idx: int,
) -> bool:
    """Calculates whether edges cross or not.

    Args:
        edge1_source_idx: Relative position of edge 1 source.
        edge1_destination_idx: Relative position of edge 1 destination.
        edge2_source_pos: Relative position of edge 2 source.
        edge2_destination_idx: Relative position of edge 2 destination.

    Returns:
        True if edges cross, else false.
    """
    # two edges tw and uv cross if and only if (x(t) - x(u))(x(w) - x(v)) is negative
    return (edge2_source_pos - edge1_source_idx) * (
        edge2_destination_idx - edge1_destination_idx
    ) < 0


def crossings_for_node_pair(
    ml_graph: MultiLayeredGraph,
    u: MLGNode,
    v: MLGNode,
    above_or_below: Literal["above"] | Literal["below"],
):
    if not (above_or_below == "above" or above_or_below == "below"):
        raise ValueError(f'{above_or_below} must be "above" or "below"')

    if not u.layer == v.layer:
        raise ValueError("Nodes must be on same layer")

    # v_idx must set to any number bigger than u_idx
    u_idx = 1
    v_idx = 2

    crossings = 0

    if above_or_below == "above":
        edges_to_use = ml_graph.nodes_to_out_edges
        nodes_to_index = ml_graph.nodes_to_indices_at_layer(u.layer + 1)
    else:
        edges_to_use = ml_graph.nodes_to_in_edges
        nodes_to_index = ml_graph.nodes_to_indices_at_layer(u.layer - 1)

    u_neighbor_idxs = [nodes_to_index[n] for n in edges_to_use[u]]
    v_neighbor_idxs = [nodes_to_index[n] for n in edges_to_use[v]]
    for u_neighbor_idx in u_neighbor_idxs:
        for v_neighbor_idx in v_neighbor_idxs:
            crossings += edges_cross(u_idx, u_neighbor_idx, v_idx, v_neighbor_idx)

    return crossings


def crossings_uv_vu(
    ml_graph: MultiLayeredGraph,
    u: MLGNode,
    v: MLGNode,
    above_or_below: Literal["above"] | Literal["below"],
):
    uv = crossings_for_node_pair(ml_graph, u, v, above_or_below)
    vu = crossings_for_node_pair(ml_graph, v, u, above_or_below)
    return uv, vu
