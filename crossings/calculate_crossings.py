from typing import Literal

from multilayered_graph.multilayered_graph import MultiLayeredGraph, MLGNode


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
    assert u.layer == v.layer

    # TODO do not fetch all indices, just of layer of u and v and layer above and below
    nodes_to_index = ml_graph.nodes_to_index_within_layer()

    # v_idx must set to any number bigger than u_idx
    u_idx = 1
    v_idx = 2

    crossings = 0

    if above_or_below == "above":
        u_out_neighbor_idxs = [
            nodes_to_index[n] for n in ml_graph.nodes_to_out_edges[u]
        ]
        v_out_neighbors_idxs = [
            nodes_to_index[n] for n in ml_graph.nodes_to_out_edges[v]
        ]
        for u_neighbor_idx in u_out_neighbor_idxs:
            for v_neighbor_idx in v_out_neighbors_idxs:
                crossings += edges_cross(u_idx, u_neighbor_idx, v_idx, v_neighbor_idx)
    elif above_or_below == "below":
        u_in_neighbor_idxs = [nodes_to_index[n] for n in ml_graph.nodes_to_in_edges[u]]
        v_in_neighbors_idxs = [nodes_to_index[n] for n in ml_graph.nodes_to_in_edges[v]]
        for u_neighbor_idx in u_in_neighbor_idxs:
            for v_neighbor_idx in v_in_neighbors_idxs:
                crossings += edges_cross(u_idx, u_neighbor_idx, v_idx, v_neighbor_idx)
    else:
        raise ValueError(f'{above_or_below} must be "above" or "below"')

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
