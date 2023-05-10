import random

from multilayered_graph.multilayered_graph import MultiLayeredGraph


def generate_multilayer_graph(layers_count: int, node_count: int, edge_density: float, long_edge_probability: float,
                              *, randomness_seed: int | None = None) -> MultiLayeredGraph:
    if (layers_count < 0 or
            node_count < 0 or
            node_count < layers_count or
            edge_density < 0 or
            edge_density > 1 or
            long_edge_probability < 0 or
            long_edge_probability > 1):
        # TODO better value error message
        raise ValueError("Invalid args passed")

    if randomness_seed is not None:
        random.seed(randomness_seed)

    ml_graph = MultiLayeredGraph(layers_count)

    _generate_random_nodes(ml_graph, layers_count=layers_count, node_count=node_count)

    _generate_random_edges(ml_graph, layers_count=layers_count, node_count=node_count, edge_density=edge_density,
                           long_edge_probability=long_edge_probability)

    return ml_graph


def _generate_random_nodes(ml_graph: MultiLayeredGraph, /, layers_count: int, node_count: int):
    # ensure at least one node per layer
    for layer in range(layers_count):
        ml_graph.add_real_node(layer)
    for _ in range(node_count - layers_count):
        layer = random.randint(0, layers_count - 1)
        ml_graph.add_real_node(layer)


def _generate_random_edges(ml_graph: MultiLayeredGraph, /, layers_count: int, node_count: int, edge_density: float,
                           long_edge_probability: float):
    # runs in O(node_count^2), can optimize to be linear

    total_edges_to_generate: int = int((node_count * (node_count - 1)) * edge_density)
    long_edges_to_generate: int = int(total_edges_to_generate * long_edge_probability)
    short_edges_to_generate: int = total_edges_to_generate - long_edges_to_generate

    # SHORT EDGES
    all_possible_short_edges = [(n1, n2)
                                for lower_layer_idx in range(layers_count - 1)
                                for n1 in ml_graph.layers_to_nodes[lower_layer_idx]
                                for n2 in ml_graph.layers_to_nodes[lower_layer_idx + 1]]
    random.shuffle(all_possible_short_edges)
    if short_edges_to_generate > len(all_possible_short_edges):
        short_edges_to_generate = len(all_possible_short_edges)
        print("EDGE DENSITY IS LOWER THAN GIVEN, AS NOT ENOUGH NODES ARE AVAILABLE. ALL POSSIBLE SHORT EDGES GENERATED")
    for i in range(short_edges_to_generate):
        n1, n2 = all_possible_short_edges[i]
        ml_graph.add_edge(n1, n2)

    # LONG EDGES
    all_previous_nodes = []
    all_possible_long_edges = []
    for lower_layer_idx in range(layers_count):
        curr_layer_nodes = ml_graph.layers_to_nodes[lower_layer_idx]
        all_possible_long_edges.extend((n1, n2) for n1 in all_previous_nodes for n2 in curr_layer_nodes)
        all_previous_nodes.extend(curr_layer_nodes)
    random.shuffle(all_possible_long_edges)
    if long_edges_to_generate > len(all_possible_long_edges):
        long_edges_to_generate = len(all_possible_long_edges)
        print("EDGE DENSITY IS LOWER THAN GIVEN, AS NOT ENOUGH NODES ARE AVAILABLE. ALL POSSIBLE LONG EDGES GENERATED")
    for i in range(long_edges_to_generate):
        n1, n2 = all_possible_long_edges[i]
        ml_graph.add_edge(n1, n2)
