from dataclasses import dataclass

from crossing_minimization.utils import (
    Above_or_below_T,
    generate_layers_to_above_or_below,
)
from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

LayerIdx = int

Positions_T = dict[LayerIdx, dict[MLGNode, int]]

PRIORITY_VIRTUAL = 1_000_000_000
PRIORITY_INF = 1_000_000_001


def initialize_positions(graph: MultiLayeredGraph, x0: int = 0) -> Positions_T:
    positions: Positions_T = {}
    for layer in range(graph.layer_count):
        positions[layer] = {}
        for idx, node in enumerate(graph.layers_to_nodes[layer]):
            positions[layer][node] = x0 + idx + 1
    return positions


# assign priorities to nodes (in this example, using node degrees as priorities)
def get_priorities(graph: MultiLayeredGraph) -> dict[MLGNode, int]:
    priorities: dict[MLGNode, int] = {}
    for node in graph.all_nodes_as_list():
        if node.is_virtual:
            priorities[node] = PRIORITY_VIRTUAL
        else:
            priorities[node] = len(graph.nodes_to_in_edges[node]) + len(
                graph.nodes_to_out_edges[node]
            )
    return priorities


@dataclass(kw_only=True)
class MLGNodeInPlacement:
    mlg_node: MLGNode
    order_index: int
    x_position: int
    target_position: int
    priority: int


def optimize_positions(
    ordered_nodes_for_placement_with_dummies: list[MLGNodeInPlacement],
    positions: Positions_T,
    layer_idx: LayerIdx,
) -> None:
    """Modifiys `positions[layer_idx]` in place."""

    # sort by priority, and remove dummy nodes
    pl_nodes_by_priority = sorted(
        ordered_nodes_for_placement_with_dummies, key=lambda n: n.priority, reverse=True
    )[2:]

    # get new positions
    for plnode in pl_nodes_by_priority:
        i = -1_000_000
        squish_pos = None
        # TODO BUG SWITCH AROUND min() and max()!!!
        if plnode.x_position < plnode.target_position:
            # find first node with higher priority
            for i in range(plnode.order_index - 1, -1, -1):
                other_pl_node = ordered_nodes_for_placement_with_dummies[i]
                if other_pl_node.priority > plnode.priority:
                    squish_pos = other_pl_node.x_position
                    break

            # squish positions to left
            if squish_pos is None:
                plnode.x_position = plnode.target_position
            else:
                plnode.x_position = min(
                    plnode.target_position, squish_pos + (plnode.order_index - i)
                )
            for j in range(plnode.order_index - 1, -1, -1):
                ordered_nodes_for_placement_with_dummies[j].x_position = min(
                    ordered_nodes_for_placement_with_dummies[j].x_position,
                    plnode.x_position - (plnode.order_index - j),
                )

        elif plnode.x_position > plnode.target_position:
            # analogous to above if-block
            for i in range(
                plnode.order_index + 1, len(ordered_nodes_for_placement_with_dummies)
            ):
                other_pl_node = ordered_nodes_for_placement_with_dummies[i]
                if other_pl_node.priority > plnode.priority:
                    squish_pos = other_pl_node.x_position
                    break

            if squish_pos is None:
                plnode.x_position = plnode.target_position
            else:
                plnode.x_position = max(
                    plnode.target_position, squish_pos - (i - plnode.order_index)
                )
            for j in range(
                plnode.order_index + 1, len(ordered_nodes_for_placement_with_dummies)
            ):
                ordered_nodes_for_placement_with_dummies[j].x_position = max(
                    ordered_nodes_for_placement_with_dummies[j].x_position,
                    plnode.x_position + (j - plnode.order_index),
                )

    for plnode in pl_nodes_by_priority:
        positions[layer_idx][plnode.mlg_node] = plnode.x_position


def get_ordered_nodes_for_placement_with_dummies(
    graph: MultiLayeredGraph,
    positions: Positions_T,
    priorities: dict[MLGNode, int],
    layer_idx: LayerIdx,
    above_or_below: Above_or_below_T,
) -> list[MLGNodeInPlacement]:
    ordered_nodes_for_placement_with_dummies: list[MLGNodeInPlacement] = []

    dummy_node = MLGNode(layer=layer_idx, name="dummy")
    ordered_nodes_for_placement_with_dummies.append(
        MLGNodeInPlacement(
            mlg_node=dummy_node,
            order_index=-1,
            x_position=-1,
            target_position=-1,
            priority=PRIORITY_INF,
        )
    )

    nodes_in_layer = graph.layers_to_nodes[layer_idx]
    for order_index, node in enumerate(nodes_in_layer):
        # get neighbor
        if above_or_below == "below":
            neighbors = graph.nodes_to_in_edges[node]
            neigbor_layer_idx = layer_idx - 1
        else:
            neighbors = graph.nodes_to_out_edges[node]
            neigbor_layer_idx = layer_idx + 1

        curr_position = positions[layer_idx][node]

        _neighbors_positions = [positions[neigbor_layer_idx][n] for n in neighbors]
        if _neighbors_positions:
            target_position = round(
                sum(_neighbors_positions) / len(_neighbors_positions)
            )
        else:
            target_position = curr_position

        ordered_nodes_for_placement_with_dummies.append(
            MLGNodeInPlacement(
                mlg_node=node,
                order_index=order_index,
                x_position=curr_position,
                target_position=target_position,
                priority=priorities[node],
            )
        )

    ordered_nodes_for_placement_with_dummies.append(
        MLGNodeInPlacement(
            mlg_node=dummy_node,
            order_index=len(nodes_in_layer),
            x_position=1_000_000,
            target_position=1_000_000,
            priority=PRIORITY_INF,
        )
    )

    return ordered_nodes_for_placement_with_dummies


def get_x_positions(ml_graph: MultiLayeredGraph):
    """
    Returns a mapping from layer to node to x-axis positions.
    Does not modify ml_graph
    """
    priorities = get_priorities(ml_graph)
    positions = initialize_positions(ml_graph)

    for layer_idx, above_or_below in generate_layers_to_above_or_below(
        ml_graph, 2, False
    ):
        ordered_nodes_for_placement_with_dummies = (
            get_ordered_nodes_for_placement_with_dummies(
                ml_graph, positions, priorities, layer_idx, above_or_below
            )
        )
        optimize_positions(
            ordered_nodes_for_placement_with_dummies, positions, layer_idx
        )

    return positions
