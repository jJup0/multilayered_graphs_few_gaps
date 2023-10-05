from crossing_minimization.utils import (
    Above_or_below_T,
    generate_layers_to_above_or_below,
)
from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

LayerIdx = int

Positions = dict[LayerIdx, dict[MLGNode, int]]


def initialize_positions(graph: MultiLayeredGraph, x0: int = 0) -> Positions:
    positions: Positions = {}
    for layer in range(graph.layer_count):
        positions[layer] = {}
        for idx, node in enumerate(graph.layers_to_nodes[layer]):
            positions[layer][node] = x0 + idx + 1
    return positions


def placement_procedure(
    graph: MultiLayeredGraph,
    positions: Positions,
    priorities: dict[MLGNode, int],
    layer_idx: LayerIdx,
    above_or_below: Above_or_below_T,
):
    nodes_in_layer = graph.layers_to_nodes[layer_idx]

    for index_in_layer, node in enumerate(nodes_in_layer):
        # get neighbor
        if above_or_below == "below":
            neighbors = graph.nodes_to_in_edges[node]
        else:
            neighbors = graph.nodes_to_out_edges[node]

        # get target position
        neighbors_positions = [positions[layer_idx][n] for n in neighbors]
        target_position = round(sum(neighbors_positions) / len(neighbors_positions))

        # TODO IMPLEMENT THIS
        # push other node if priority allows
        # priority = priorities[node]
        # positions[layer_idx][node] = target_position


# Assign priorities to nodes (in this example, using node degrees as priorities)
def get_priorities(graph: MultiLayeredGraph) -> dict[MLGNode, int]:
    priorities: dict[MLGNode, int] = {}
    for node in graph.all_nodes_as_list():
        if node.is_virtual:
            priorities[node] = 1_000_000_000
        else:
            priorities[node] = len(graph.nodes_to_in_edges[node]) + len(
                graph.nodes_to_out_edges[node]
            )
    return priorities


ml_graph = MultiLayeredGraph()

priorities = get_priorities(ml_graph)
positions = initialize_positions(ml_graph)

for layer_idx, above_or_below in generate_layers_to_above_or_below(ml_graph, 2, False):
    placement_procedure(ml_graph, positions, priorities, layer_idx, above_or_below)
