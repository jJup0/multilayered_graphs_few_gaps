from multilayered_graph.multilayer_graph_generator import generate_multilayer_graph
from node_sorting.barycenter_heuristic import few_gaps_barycenter_sort

ml_graph = generate_multilayer_graph(4, 12, 0.2, 0.5, randomness_seed=1)

first_node = ml_graph.all_nodes_as_list()[0]

pgv_graph = ml_graph.to_pygraphviz_graph()
# pos = ml_graph.nodes_positions()

# print(pgv_graph.string())  # print to screen
# pgv_graph.write("simple.dot")  # write to simple.dot
pgv_graph.draw("plain.svg")

few_gaps_barycenter_sort(ml_graph)
pgv_graph_barycenter = ml_graph.to_pygraphviz_graph()
pgv_graph_barycenter.draw("barycenter.svg")

pgv_graph.layout(prog="dot")
pgv_graph.draw("dot.svg")

x = pgv_graph.nodes()
print(x)
print(x[0])
print(type(x[0]))
