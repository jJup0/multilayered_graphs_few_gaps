# cli script
# expects 5 arguments:
# graph_count, nodes_per_layer, virtual_node_count, regular_edge_density, out_dir
# example:
# python -m thesis_experiments.generate_2_layer_test_instances 1 5 5 0.1 thesis_experiments/local_tests

import os
import random
import sys

from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

# try:
#     # Parsing argument
#     argument_list = sys.argv[1:]
#     options = ""
#     long_options = ["two_sided"]
#     options_and_values, normal_args = getopt.getopt(
#         argument_list, options, long_options
#     )
# except getopt.error as err:
#     print(str(err))
#     exit()

# two_sided = None


# # checking each argument
# for current_argument, current_value in options_and_values:
#     if current_argument == "--two_sided":
#         two_sided = True


def generate_oscm_graph(
    nodes_per_layer: int,
    virtual_node_count: int,
    regular_edge_density: float,
    # *,
    # two_sided: bool = False
) -> MultiLayeredGraph:
    assert nodes_per_layer >= 0, f"Invalid {nodes_per_layer=}"
    assert virtual_node_count >= 0, f"Invalid {virtual_node_count=}"
    assert 0 <= regular_edge_density <= 1, f"Invalid {regular_edge_density=}"

    regular_edges_count = int(nodes_per_layer * nodes_per_layer * regular_edge_density)

    ml_graph = MultiLayeredGraph(2)
    for _ in range(nodes_per_layer):
        ml_graph.add_real_node(0)
    for _ in range(nodes_per_layer):
        ml_graph.add_real_node(1)

    all_edges: list[tuple[MLGNode, MLGNode]] = [
        (n1, n2)
        for n1 in ml_graph.layers_to_nodes[0]
        for n2 in ml_graph.layers_to_nodes[1]
    ]
    random.shuffle(all_edges)
    for i in range(regular_edges_count):
        n1, n2 = all_edges[i]
        ml_graph.add_edge(n1, n2)  # type: ignore # protected

    # create virtual nodes first
    vnodes_l1: set[MLGNode] = set()
    vnodes_l2: set[MLGNode] = set()
    for i in range(virtual_node_count):
        vnodes_l1.add(ml_graph.add_virtual_node(1, ""))
        vnodes_l2.add(ml_graph.add_virtual_node(1, ""))

    # then add edges for each virtual node
    all_nodes_l1 = list(ml_graph.layers_to_nodes[0])
    all_nodes_l2 = list(ml_graph.layers_to_nodes[1])
    while vnodes_l1:
        vnode1 = vnodes_l1.pop()
        neighbor_in_l2 = random.choice(all_nodes_l2)
        ml_graph.add_edge(vnode1, neighbor_in_l2)
        if neighbor_in_l2 is vnodes_l2:
            all_nodes_l2.remove(neighbor_in_l2)
            vnodes_l2.remove(neighbor_in_l2)

    while vnodes_l2:
        vnode2 = vnodes_l2.pop()
        neighbor_in_l1 = random.choice(all_nodes_l2)
        ml_graph.add_edge(neighbor_in_l1, vnode2)
        if neighbor_in_l1 is vnodes_l1:
            all_nodes_l1.remove(neighbor_in_l1)
            vnodes_l1.remove(neighbor_in_l1)

    return ml_graph


if __name__ == "__main__":
    (
        _py_file_name,
        graph_count,
        nodes_per_layer,
        virtual_node_count,
        regular_edge_density,
        out_dir,
    ) = sys.argv

    graph_count = int(graph_count)
    nodes_per_layer = int(nodes_per_layer)
    virtual_node_count = int(virtual_node_count)
    regular_edge_density = float(regular_edge_density)

    graph_count_str_len = len(str(graph_count))
    for i in range(graph_count):
        g = generate_oscm_graph(
            nodes_per_layer, virtual_node_count, regular_edge_density
        )
        file_name = (
            f"lg_r={nodes_per_layer}"
            + f"_v={virtual_node_count}"
            + f"_p={regular_edge_density}"
            + f"_id={str(i).zfill(graph_count_str_len)}.json"
        )
        g.serialize_proprietary(os.path.realpath(os.path.join(".", out_dir, file_name)))
