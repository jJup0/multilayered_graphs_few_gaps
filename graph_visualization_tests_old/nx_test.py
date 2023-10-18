# type: ignore
"""Module is unused, used to be for testing how networkx works/looks."""
import random

import matplotlib.pyplot as plt
import networkx as nx


def test_mlg():
    from multilayered_graph.multilayer_graph_generator import generate_multilayer_graph

    mlgraph = generate_multilayer_graph(7, 24, 0.2, 0.5)

    G = mlgraph.to_networkx_graph()
    pos = mlgraph.nodes_to_integer_relative_coordinates()

    # color_map = ["green"] + ["blue"] * (len(G) - 1)
    # size_map = [1000] + [300] * (len(G) - 1)
    size_map = [0 if node.is_virtual else 300 for node in G.nodes]
    labels_map = {node: "" if node.is_virtual else node.name for node in G.nodes}

    nx.draw(G, pos, labels=labels_map, node_size=size_map)  # node_color=color_map)
    # nx.draw(G, pos, with_labels=True, node_size=size_map) #  with_labels=True, node_color=color_map)
    plt.show()


def test_random_clustered():
    jds = []
    node_count = 20
    iedge_count = 0
    tri_count = 0
    for _ in range(node_count - 1):
        iedges = random.randint(1, node_count - 5)
        tris = random.randint(0, 2)
        iedge_count += iedges
        tri_count += tris
        jds.append((iedges, tris))

    jds.append((2 + iedge_count % 2, 3 - tri_count % 3))

    G = nx.random_clustered_graph(jds)
    # remove multi-edges
    G = nx.Graph(G)

    # remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    nx.draw(G)
    plt.show()


if __name__ == "__main__":
    test_random_clustered()
