# cli script
# expects 5 arguments:
# graph_count, nodes_per_layer, virtual_node_count, regular_edge_density, out_dir
# example:
# python -m thesis_experiments.generate_2_layer_test_instances 1 5 5 0.1 thesis_experiments/local_tests

import os
import random
import sys

from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

out_dir = os.path.join(os.path.dirname(__file__), "graph_instances")


def generate_oscm_graph(
    nodes_per_layer: int,
    virtual_node_count: int,
    regular_edge_density: float,
) -> MultiLayeredGraph:
    assert nodes_per_layer >= 0, f"Invalid {nodes_per_layer=}"
    assert virtual_node_count >= 0, f"Invalid {virtual_node_count=}"
    assert 0 <= regular_edge_density <= 1, f"Invalid {regular_edge_density=}"

    regular_edges_count = int(nodes_per_layer * nodes_per_layer * regular_edge_density)

    ml_graph = MultiLayeredGraph(2)
    l1_nodes: list[MLGNode] = []
    l2_nodes: list[MLGNode] = []
    for _ in range(nodes_per_layer):
        l1_nodes.append(ml_graph.add_real_node(0))
    for _ in range(nodes_per_layer):
        l2_nodes.append(ml_graph.add_real_node(1))

    all_edges: list[tuple[MLGNode, MLGNode]] = [
        (n1, n2) for n1 in l1_nodes for n2 in l2_nodes
    ]
    random.shuffle(all_edges)
    for i in range(regular_edges_count):
        n1, n2 = all_edges[i]
        ml_graph.add_edge(n1, n2)  # type: ignore # protected

    for i in range(virtual_node_count):
        neighbor = random.choice(l1_nodes)
        v_node = ml_graph.add_virtual_node(1, f"{neighbor}_id={i}")
        l2_nodes.append(v_node)
        ml_graph.add_edge(neighbor, v_node)

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
