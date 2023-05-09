import matplotlib.pyplot as plt
import networkx as nx

from multilayered_graph.multilayer_graph_generator import generate_multilayer_graph

mlgraph = generate_multilayer_graph(7, 24, 0.2, 0.5)

G = mlgraph.to_networkx_graph()
pos = mlgraph.nodes_positions()

# color_map = ["green"] + ["blue"] * (len(G) - 1)
# size_map = [1000] + [300] * (len(G) - 1)
size_map = [0 if node.is_virtual else 300 for node in G.nodes]
labels_map = {node: "" if node.is_virtual else node.name for node in G.nodes}

nx.draw(G, pos, labels=labels_map, node_size=size_map)  # node_color=color_map)
# nx.draw(G, pos, with_labels=True, node_size=size_map) #  with_labels=True, node_color=color_map)
plt.show()
