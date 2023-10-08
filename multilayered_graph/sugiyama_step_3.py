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
            neigbor_layer_idx = layer_idx - 1
        else:
            neighbors = graph.nodes_to_out_edges[node]
            neigbor_layer_idx = layer_idx + 1

        # get target position
        curr_position = positions[layer_idx][node]
        _neighbors_positions = [positions[neigbor_layer_idx][n] for n in neighbors]
        target_position = round(sum(_neighbors_positions) / len(_neighbors_positions))
        # already set here?
        positions[node] = target_position
        curr_node_priority = priorities[node]
        if target_position < curr_position:
            other_node_idx = index_in_layer
            squish_position = target_position
            for other_node_idx in range(index_in_layer - 1, -1, -1):
                other_node = nodes_in_layer[other_node_idx]
                other_node_position = positions[layer_idx][other_node]
                if other_node_position < target_position:
                    nodes_in_layer[other_node_idx + 1]
                if priorities[other_node] > curr_node_priority:
                    break

        # TODO IMPLEMENT THIS
        # push other node if priority allows
        # priority = priorities[node]
        # positions[layer_idx][node] = target_position


"""
I have n objects with a predefined order: order = [obj1, obj2, ...]
Each object has an ideal position: ipos[obj] = x
Each object has a priority: prio[obj] = y
The object with the highest priority is placed first and cannot be displaced anymore afterwards.
"""


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
