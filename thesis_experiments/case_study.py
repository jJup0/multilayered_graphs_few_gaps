import copy
import logging
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

nodes_as_numbers = [
    [8, 24, 1, 35, 30],
    [7, 23],
    [6, 15, 22, 29],
    [5, 21, 20, 28],
    [4, 19],
    [39, 41, 38, 40],
    [42, 26, 3, 16, 17, 18, 11, 14, 37, 13, 12, 43, 36, 32, 34],
    [9, 25, 27, 2, 10, 31, 33],
    # [8, 24, 1, 35, 30],
]


all_numbers = [num for layer in nodes_as_numbers for num in layer]
all_numbers_set = set(all_numbers)
# # print(len(all_numbers), len(all_numbers_set))
# all_numbers_counter = Counter(all_numbers)
# for num, count in all_numbers_counter.items():
#     if count > 1:
#         print(num, count)

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


def add_node_as_number(layer: int, number: int):
    global ml_graph_k_gaps, node_number_to_node
    node = ml_graph_k_gaps.add_real_node(layer, str(number))
    # layers_and_number_to_node[layer, number] = node
    assert number not in node_number_to_node, f"{number=} already in number_to_node"
    node_number_to_node[number] = node


node_number_to_node: dict[int, MLGNode] = {}
ml_graph_k_gaps = MultiLayeredGraph(9)

for layer_idx, layer_nodes in enumerate(nodes_as_numbers):
    for num in layer_nodes:
        add_node_as_number(layer_idx, num)

for num, neighbor_nums in num_to_neighbors.items():
    for neighbor_num in neighbor_nums:
        ml_graph_k_gaps.add_edge(
            node_number_to_node[num], node_number_to_node[neighbor_num]
        )

ml_graph_many_gaps = copy.deepcopy(ml_graph_k_gaps)
ml_graph_side_gaps = copy.deepcopy(ml_graph_k_gaps)

use_ilp = False
if use_ilp:
    from crossing_minimization.gurobi_int_lin import GurobiSorter

    GurobiSorter.sort_graph(
        ml_graph_k_gaps,
        max_iterations=3,
        only_one_up_iteration=False,
        side_gaps_only=False,
        max_gaps=2,
    )
else:
    from crossing_minimization.barycenter_heuristic import BarycenterImprovedSorter

    iterations = 10
    BarycenterImprovedSorter.sort_graph(
        ml_graph_side_gaps,
        max_iterations=iterations,
        only_one_up_iteration=False,
        side_gaps_only=True,
        max_gaps=2,
    )
    BarycenterImprovedSorter.sort_graph(
        ml_graph_k_gaps,
        max_iterations=iterations,
        only_one_up_iteration=False,
        side_gaps_only=False,
        max_gaps=2,
    )
    BarycenterImprovedSorter.sort_graph(
        ml_graph_many_gaps,
        max_iterations=iterations,
        only_one_up_iteration=False,
        side_gaps_only=False,
        max_gaps=100,
    )


def draw_graph(_g: MultiLayeredGraph, ax: Any):
    nx_graph = _g.to_networkx_graph()
    pos = _g.nodes_to_integer_relative_coordinates()
    coords = sorted(pos.values(), key=lambda x: x[1])
    logging.debug("coords: %s", coords)
    size_map = [0 if node.is_virtual else 300 for node in nx_graph.nodes]
    labels_map = {node: "" if node.is_virtual else node.name for node in nx_graph.nodes}

    nx.draw(nx_graph, pos, labels=labels_map, node_size=size_map, ax=ax)


fig, axs = plt.subplots(1, 3, figsize=(10, 5))

draw_graph(ml_graph_side_gaps, axs[0])
draw_graph(ml_graph_k_gaps, axs[1])
draw_graph(ml_graph_many_gaps, axs[2])
plt.show()
