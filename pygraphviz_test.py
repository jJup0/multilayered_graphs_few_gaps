from multilayer_graph_generator import generate_multilayer_graph

ml_graph = generate_multilayer_graph(4, 12, 0.2, 0.5, randomness_seed=1)

first_node = ml_graph.all_nodes_as_list()[0]

pgv_graph = ml_graph.to_pygraphviz_graph()
pos = ml_graph.nodes_positions()

# print(pgv_graph.string())  # print to screen
pgv_graph.write("simple.dot")  # write to simple.dot
pgv_graph.draw("simple.png")
