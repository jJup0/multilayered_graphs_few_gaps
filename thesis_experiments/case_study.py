from collections import Counter

from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

g = MultiLayeredGraph(9)
layers_and_number_to_node: dict[tuple[int, int], MLGNode] = {}


def add_node_as_number(layer: int, number: int):
    global g
    node = g.add_real_node(layer, str(number))
    layers_and_number_to_node[layer, number] = node


# l1
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
# for layer_idx, layer_nodes in enumerate(nodes_as_numbers):
#     for num in layer_nodes:
#         add_node_as_number(layer_idx, num)

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
    6: [],
    15: [],
    22: [],
    29: [],
    5: [],
    21: [],
    20: [],
    28: [],
    4: [],
    19: [],
    39: [],
    41: [],
    38: [],
    40: [],
    42: [],
    26: [],
    3: [],
    16: [],
    17: [],
    18: [],
    11: [],
    14: [],
    37: [],
    13: [],
    12: [],
    43: [],
    36: [],
    32: [],
    34: [],
    9: [],
    25: [],
    27: [],
    2: [],
    10: [],
    31: [],
    33: [],
}
