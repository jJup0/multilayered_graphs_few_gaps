# cli script
# expects 5 arguments:
# graph_count, nodes_per_layer, virtual_node_count, regular_edge_density, out_dir
# example:
# python -m thesis_experiments.generate_2_layer_test_instances 1 5 5 0.1 thesis_experiments/local_tests

import os
import random
import sys

from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph


def generate_oscm_graph(
    nodes_per_layer: int,
    virtual_node_ratio: float,
    average_node_degree: float,
) -> MultiLayeredGraph:
    assert 0 <= nodes_per_layer, f"Invalid {nodes_per_layer=}"
    assert 0 <= virtual_node_ratio <= 1, f"Invalid {virtual_node_ratio=}"
    assert 0 <= average_node_degree, f"Invalid {average_node_degree=}"

    virtual_nodes_per_layer = int(nodes_per_layer * virtual_node_ratio)
    real_nodes_per_layer = nodes_per_layer - virtual_nodes_per_layer

    # real nodes
    ml_graph = MultiLayeredGraph(2)
    for layer_idx in [0, 1]:
        for _ in range(real_nodes_per_layer):
            ml_graph.add_real_node(layer_idx)

    # edges between real nodes
    all_edges: list[tuple[MLGNode, MLGNode]] = [
        (n1, n2)
        for n1 in ml_graph.layers_to_nodes[0]
        for n2 in ml_graph.layers_to_nodes[1]
    ]
    random.shuffle(all_edges)
    regular_edges_count = int(real_nodes_per_layer * average_node_degree)
    for i in range(regular_edges_count):
        n1, n2 = all_edges[i]
        ml_graph.add_edge(n1, n2)

    # create virtual nodes
    vnodes_l1: set[MLGNode] = set()
    vnodes_l2: set[MLGNode] = set()
    for i in range(virtual_nodes_per_layer):
        vnodes_l1.add(ml_graph.add_virtual_node(0, f"{i}"))
        vnodes_l2.add(ml_graph.add_virtual_node(1, f"{i}"))

    # then add edges for each virtual node
    def check_vnodes_have_single_neighbor(curr_node: MLGNode | None = None):
        for vnode in [n for n in ml_graph.all_nodes_as_list() if n.is_virtual]:
            assert (
                len(ml_graph.nodes_to_in_edges[vnode])
                + len(ml_graph.nodes_to_out_edges[vnode])
            ) <= 1, f"{vnode=} got second neighbor. || {curr_node=}"

    all_nodes_l2 = list(ml_graph.layers_to_nodes[1])
    for vnode1 in vnodes_l1:
        neighbor_in_l2 = random.choice(all_nodes_l2)
        ml_graph.add_edge(vnode1, neighbor_in_l2)
        if neighbor_in_l2 in vnodes_l2:
            all_nodes_l2.remove(neighbor_in_l2)  # O(n) operation
            vnodes_l2.remove(neighbor_in_l2)
        check_vnodes_have_single_neighbor(vnode1)

    real_nodes_l1 = [n for n in ml_graph.layers_to_nodes[0] if not n.is_virtual]
    for vnode2 in vnodes_l2:
        neighbor_in_l1 = random.choice(real_nodes_l1)
        ml_graph.add_edge(neighbor_in_l1, vnode2)
        check_vnodes_have_single_neighbor(vnode2)

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
        os.makedirs(out_dir, exist_ok=True)
        g.serialize_proprietary(os.path.realpath(os.path.join(".", out_dir, file_name)))
