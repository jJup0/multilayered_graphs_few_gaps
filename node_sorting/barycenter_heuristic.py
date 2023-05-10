from multilayered_graph.multilayered_graph import MultiLayeredGraph, MLGNode

PSEUDO_SORT_DISPLACE_VALUE = 1_000


def few_gaps_barycenter_sort(ml_graph: MultiLayeredGraph) -> None:
    """Sorts nodes in multilayered graph according to barycenter heuristic.

    Modified in place.
    Args:
        ml_graph: Graph on which to apply sorting.
    """

    node_to_in_neighbors = ml_graph.nodes_to_in_edges()

    for layer in range(1, ml_graph.layer_count):
        prev_layer_indices = {
            node: index
            for index, node in enumerate(ml_graph.layers_to_nodes[layer - 1])
        }
        real_node_bary_median = _get_real_node_barycenter_median(
            ml_graph.layers_to_nodes[layer], node_to_in_neighbors, prev_layer_indices
        )
        ml_graph.layers_to_nodes[layer].sort(
            key=lambda node: _get_pseudo_barycenter(
                node,
                node_to_in_neighbors[node],
                prev_layer_indices,
                real_node_bary_median,
            )
        )


def _get_real_node_barycenter_median(
    nodes_at_layer: list[MLGNode],
    node_to_in_neighbors: dict[MLGNode, set[MLGNode]],
    prev_layer_indices: dict[MLGNode, int],
):
    real_node_barycenters = []
    for node in nodes_at_layer:
        if node.is_virtual:
            continue
        real_node_barycenters.append(
            _get_real_node_barycenter(
                node, node_to_in_neighbors[node], prev_layer_indices
            )
        )

    # should have at least one real node in layer, so list should not be empty
    real_node_barycenters.sort()
    real_node_bary_median = real_node_barycenters[len(real_node_barycenters) // 2]
    return real_node_bary_median


def _get_real_node_barycenter(
    node: MLGNode, neighbors: set[MLGNode], prev_layer_indices: dict[MLGNode, int]
) -> float:
    neighbor_count = len(neighbors)
    barycenter = sum(prev_layer_indices[node] for node in neighbors) / neighbor_count
    node.text_info = f"bary {barycenter:.5}"
    return barycenter


def _get_pseudo_barycenter(
    node: MLGNode,
    neighbors: set[MLGNode],
    prev_layer_indices: dict[MLGNode, int],
    real_node_bary_median: float,
) -> float:
    # TODO get barycenter as accurate fraction to avoid
    #   floating point errors, and ensure stable sorting

    neighbor_count = len(neighbors)
    barycenter = sum(prev_layer_indices[node] for node in neighbors) / neighbor_count

    if node.is_virtual:
        if barycenter > real_node_bary_median:
            barycenter += PSEUDO_SORT_DISPLACE_VALUE
        else:
            barycenter -= PSEUDO_SORT_DISPLACE_VALUE
    node.text_info = f"bary {barycenter:.5}"
    return barycenter
