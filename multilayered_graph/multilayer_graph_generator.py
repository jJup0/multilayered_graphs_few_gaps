import random

from multilayered_graph.multilayered_graph import (
    MLGNode,
    MLGNodeEdgeType,
    MultiLayeredGraph,
)


def generate_two_layer_graph(
    layer1_count: int,
    layer2_count: int,
    virtual_nodes_count: int,
    regular_edges_count: int | None = None,
    regular_edge_density: float | None = None,
) -> MultiLayeredGraph:
    if (regular_edge_density is not None) and (not (0 <= regular_edge_density <= 1)):
        raise ValueError(f"Invalid {regular_edge_density=}")

    if regular_edges_count is None:
        if regular_edge_density is None:
            raise ValueError(
                f"regular_edge_density or regular_edges_count need to be given"
            )
        regular_edges_count = int(layer1_count * layer2_count * regular_edge_density)

    ml_graph = MultiLayeredGraph(2)
    l1_nodes: list[MLGNode] = []
    l2_nodes: list[MLGNode] = []
    for _ in range(layer1_count):
        l1_nodes.append(ml_graph.add_real_node(0))
    for _ in range(layer2_count):
        l2_nodes.append(ml_graph.add_real_node(1))
    for i in range(virtual_nodes_count):
        neighbor = random.choice(l1_nodes)
        v_node = ml_graph.add_virtual_node(1, f"{neighbor}_id={i}")
        l2_nodes.append(v_node)
        ml_graph.add_edge(neighbor, v_node)

    for _ in range(regular_edges_count):
        n1 = random.choice(l1_nodes)
        n2 = random.choice(l2_nodes)
        ml_graph.add_edge(n1, n2)

    return ml_graph


def generate_multilayer_graph(
    layers_count: int,
    node_count: int,
    edge_density: float,
    long_edge_probability: float,
    *,
    randomness_seed: int | None = None,
) -> MultiLayeredGraph:
    if (
        layers_count < 0
        or node_count < 0
        or node_count < layers_count
        or edge_density < 0
        or edge_density > 1
        or long_edge_probability < 0
        or long_edge_probability > 1
    ):
        raise ValueError("Invalid arguments passed.")

    if randomness_seed is not None:
        random.seed(randomness_seed)

    ml_graph = MultiLayeredGraph(layers_count)

    _generate_random_nodes(ml_graph, layers_count=layers_count, node_count=node_count)

    _generate_random_edges(
        ml_graph,
        layers_count=layers_count,
        node_count=node_count,
        edge_density=edge_density,
        long_edge_probability=long_edge_probability,
    )

    return ml_graph


def random_graph_to_short_str(
    layers_count: int,
    node_count: int,
    edge_density: float,
    long_edge_probability: float,
    randomness_seed: int | None = None,
) -> str:
    short = f"random-{layers_count}L_{node_count}N_{edge_density:.2}p_{long_edge_probability:.2}l"
    if randomness_seed is not None:
        short += f"_{randomness_seed}s"
    return short


def random_graph_to_long_str(
    layers_count: int,
    node_count: int,
    edge_density: float,
    long_edge_probability: float,
    randomness_seed: int | None = None,
) -> str:
    long = (
        f"Random graph, {layers_count} layers, {node_count} nodes, {edge_density:.2f} edge density, "
        f"{long_edge_probability:.2f} long edge density"
    )
    if randomness_seed is not None:
        long += f", seed: {randomness_seed}"
    return long


def random_graph_to_short_long_str(
    layers_count: int,
    node_count: int,
    edge_density: float,
    long_edge_probability: float,
    randomness_seed: int | None = None,
) -> tuple[str, str]:
    short = random_graph_to_short_str(
        layers_count, node_count, edge_density, long_edge_probability, randomness_seed
    )
    long = random_graph_to_long_str(
        layers_count, node_count, edge_density, long_edge_probability, randomness_seed
    )
    return short, long


def _generate_random_nodes(
    ml_graph: MultiLayeredGraph, /, layers_count: int, node_count: int
):
    # ensure at least one node per layer
    for layer in range(layers_count):
        ml_graph.add_real_node(layer)
    for _ in range(node_count - layers_count):
        layer = random.randint(0, layers_count - 1)
        ml_graph.add_real_node(layer)


def _generate_random_edges(
    ml_graph: MultiLayeredGraph,
    /,
    layers_count: int,
    node_count: int,
    edge_density: float,
    long_edge_probability: float,
):
    # runs in O(node_count^2), can optimize to be linear

    total_edges_to_generate: int = int((node_count * (node_count - 1)) * edge_density)
    long_edges_to_generate: int = int(total_edges_to_generate * long_edge_probability)
    short_edges_to_generate: int = total_edges_to_generate - long_edges_to_generate

    # SHORT EDGES
    all_possible_short_edges = [
        (n1, n2)
        for lower_layer_idx in range(layers_count - 1)
        for n1 in ml_graph.layers_to_nodes[lower_layer_idx]
        for n2 in ml_graph.layers_to_nodes[lower_layer_idx + 1]
    ]
    random.shuffle(all_possible_short_edges)
    if short_edges_to_generate > len(all_possible_short_edges):
        short_edges_to_generate = len(all_possible_short_edges)
        print(
            "EDGE DENSITY IS LOWER THAN GIVEN, AS NOT ENOUGH NODES ARE AVAILABLE. ALL POSSIBLE SHORT EDGES GENERATED"
        )

    for upper_layer_idx in range(1, layers_count):
        for n2 in ml_graph.layers_to_nodes[upper_layer_idx]:
            n1 = random.choice(ml_graph.layers_to_nodes[upper_layer_idx - 1])
            # HERE_ID=min_one_edge
            ml_graph.add_edge(n1, n2)

    # TODO bug check: adding a node at HERE_ID=min_one_edge, should be excluded from adding here?
    for i in range(short_edges_to_generate):
        n1, n2 = all_possible_short_edges[i]
        ml_graph.add_edge(n1, n2)

    # LONG EDGES
    all_previous_nodes: list[MLGNode] = []
    all_possible_long_edges: list[MLGNodeEdgeType] = []
    for lower_layer_idx in range(layers_count):
        curr_layer_nodes = ml_graph.layers_to_nodes[lower_layer_idx]
        all_possible_long_edges.extend(
            (n1, n2) for n1 in all_previous_nodes for n2 in curr_layer_nodes
        )
        all_previous_nodes.extend(curr_layer_nodes)
    random.shuffle(all_possible_long_edges)
    if long_edges_to_generate > len(all_possible_long_edges):
        long_edges_to_generate = len(all_possible_long_edges)
        print(
            "EDGE DENSITY IS LOWER THAN GIVEN, AS NOT ENOUGH NODES ARE AVAILABLE. ALL POSSIBLE LONG EDGES GENERATED"
        )
    for i in range(long_edges_to_generate):
        n1, n2 = all_possible_long_edges[i]
        ml_graph.add_edge(n1, n2)
