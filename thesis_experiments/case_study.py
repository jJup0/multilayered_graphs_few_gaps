import copy
import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from crossing_minimization.utils import GraphSorter
from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

RANDOMNESS_SEED = None


def create_case_study_graph() -> MultiLayeredGraph:
    """Constructs the graph from the sugiyama paper.

    Returns:
        MultiLayeredGraph: The sugiyama paper case study graph as a "MultiLayeredGraph" object.
    """
    global RANDOMNESS_SEED
    nodes_as_numbers = [
        [8, 24, 1, 35, 30],
        [7, 23],
        [6, 15, 22, 29],
        [5, 21, 20, 28],
        [4, 19],
        [39, 41, 38, 40],
        [42, 26, 3, 16, 17, 18, 11, 14, 37, 13, 12, 43, 36, 32, 34],
        [9, 25, 27, 2, 10, 31, 33],
        # [8, 24, 1, 35, 30], # TODO add these numbers back in but add 100 or something to make them distinguishable from first layer
    ]

    num_to_neighbors = {
        8: [7],
        24: [27, 23],
        1: [9, 25, 2, 15, 10, 23, 31],
        35: [5, 22],
        30: [29, 33],
        # l2
        7: [6],
        23: [22],
        6: [5],
        15: [16, 14, 39, 20],
        22: [21],
        29: [18, 41, 28, 12, 34],
        5: [4],
        21: [19],
        20: [19],
        28: [19],
        4: [42, 26, 3, 11, 39],
        19: [17, 13, 40, 36],
        39: [37],
        41: [37],
        38: [37, 43],
        40: [37, 43],
        42: [9],
        26: [25],
        3: [2],
        16: [2],
        17: [2],
        18: [2],
        11: [10],
        14: [10],
        37: [],
        13: [10],
        12: [10],
        43: [],  # 35
        36: [],  # 35
        32: [31],
        34: [33],
        9: [],  # 8
        25: [],  # 24
        27: [],  # 24
        2: [],  # 1
        10: [],  # 1
        31: [],  # 30
        33: [],  # 30
    }

    node_number_to_node: dict[int, MLGNode] = {}
    ml_graph = MultiLayeredGraph(9)

    def add_node_as_number(layer: int, number: int):
        nonlocal ml_graph, node_number_to_node
        node = ml_graph.add_real_node(layer, str(number))
        # layers_and_number_to_node[layer, number] = node
        assert number not in node_number_to_node, f"{number=} already in number_to_node"
        node_number_to_node[number] = node

    for layer_idx, layer_nodes in enumerate(nodes_as_numbers):
        for num in layer_nodes:
            add_node_as_number(layer_idx, num)

    for num, neighbor_nums in num_to_neighbors.items():
        for neighbor_num in neighbor_nums:
            ml_graph.add_edge(
                node_number_to_node[num], node_number_to_node[neighbor_num]
            )

    # shuffle nodes if randomness seed is set
    if RANDOMNESS_SEED is not None:
        random.seed(RANDOMNESS_SEED)
        for layer_idx in range(ml_graph.layer_count):
            random.shuffle(ml_graph.layers_to_nodes[layer_idx])

    return ml_graph


@dataclass
class NamedGraphAndParams:
    name: str
    graph: MultiLayeredGraph
    Sorter: type[GraphSorter]
    side_gaps: bool
    max_gaps: int


def draw_graph(_g: MultiLayeredGraph, ax: Any | None = None):
    nx_graph = _g.to_networkx_graph()
    pos = _g.nodes_to_integer_relative_coordinates()
    coords = sorted(pos.values(), key=lambda x: x[1])
    logging.debug("coords: %s", coords)
    size_map = [0 if node.is_virtual else 300 for node in nx_graph.nodes]
    labels_map = {node: "" if node.is_virtual else node.name for node in nx_graph.nodes}

    if ax is not None:
        nx.draw(nx_graph, pos, labels=labels_map, node_size=size_map, ax=ax)
    else:
        nx.draw(nx_graph, pos, labels=labels_map, node_size=size_map)


def annotate_and_save_graph(named_graph: NamedGraphAndParams, save_dir: str):
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    # if "unlimited" in named_graph.name:
    #     annotation_coords = (
    #         xmin + (xmax - xmin) * 0.6,
    #         ymin + (ymax - ymin) * 0.05,
    #     )
    # else:
    annotation_coords = (
        xmin + (xmax - xmin) * 0.8,
        ymin + (ymax - ymin) * 0.01,
    )
    plt.annotate(
        f"crossings: {named_graph.graph.get_total_crossings()}", annotation_coords
    )
    plt.savefig(os.path.join(save_dir, named_graph.name), dpi=300)
    plt.clf()


def perform_case_study():
    # import here to avoid import cycle
    from crossing_minimization.barycenter_heuristic import BarycenterImprovedSorter
    from crossing_minimization.gurobi_int_lin import GurobiSorter

    ml_graph = create_case_study_graph()

    named_graphs = [
        NamedGraphAndParams(
            name="Barycenter side gaps",
            graph=copy.deepcopy(ml_graph),
            Sorter=BarycenterImprovedSorter,
            side_gaps=True,
            max_gaps=2,
        ),
        NamedGraphAndParams(
            name="Barycenter two gaps",
            graph=copy.deepcopy(ml_graph),
            Sorter=BarycenterImprovedSorter,
            side_gaps=False,
            max_gaps=2,
        ),
        NamedGraphAndParams(
            name="Barycenter unlimited gaps",
            graph=copy.deepcopy(ml_graph),
            Sorter=BarycenterImprovedSorter,
            side_gaps=False,
            max_gaps=100,
        ),
        NamedGraphAndParams(
            name="Gurobi side gaps",
            graph=copy.deepcopy(ml_graph),
            Sorter=GurobiSorter,
            side_gaps=True,
            max_gaps=2,
        ),
        NamedGraphAndParams(
            name="Gurobi two gaps",
            graph=copy.deepcopy(ml_graph),
            Sorter=GurobiSorter,
            side_gaps=False,
            max_gaps=2,
        ),
        NamedGraphAndParams(
            name="Gurobi unlimited gaps",
            graph=copy.deepcopy(ml_graph),
            Sorter=GurobiSorter,
            side_gaps=False,
            max_gaps=100,
        ),
    ]
    max_iterations = 2
    only_one_up_iteration = False

    for named_graph in named_graphs:
        named_graph.Sorter.sort_graph(
            named_graph.graph,
            max_iterations=max_iterations,
            only_one_up_iteration=only_one_up_iteration,
            side_gaps_only=named_graph.side_gaps,
            max_gaps=named_graph.max_gaps,
        )

    save_dir = os.path.realpath(os.path.dirname(__file__))
    while "saved_plots" not in os.listdir(save_dir) and save_dir != "/":
        save_dir = os.path.dirname(save_dir)

    save_dir = os.path.join(
        save_dir,
        "saved_plots",
        f"case_study_long_edge_switch{RANDOMNESS_SEED if RANDOMNESS_SEED is not None else 'unshuffled'}",
    )
    graph_obj_dir = os.path.join(save_dir, "graph_objects")
    os.makedirs(graph_obj_dir, exist_ok=True)

    for named_graph in named_graphs:
        GraphSorter.rearrange_trivial_long_edges(named_graph.graph)
        named_graph.graph.serialize_proprietary(
            os.path.join(graph_obj_dir, f"{named_graph.name}.json")
        )
        draw_graph(named_graph.graph)
        annotate_and_save_graph(named_graph, save_dir)


if __name__ == "__main__":
    perform_case_study()
