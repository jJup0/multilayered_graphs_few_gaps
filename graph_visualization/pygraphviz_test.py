from crossing_minimization.barycenter_heuristic import few_gaps_barycenter_smart_sort
from crossing_minimization.median_heuristic import few_gaps_median_sort_improved
from multilayered_graph.multilayer_graph_generator import generate_multilayer_graph
from multilayered_graph.multilayered_graph import MultiLayeredGraph


def draw_graph():
    ml_graph = generate_multilayer_graph(5, 16, 0.2, 0.5, randomness_seed=1)

    # first_node = ml_graph.all_nodes_as_list()[0]
    #
    # pgv_graph = ml_graph.to_pygraphviz_graph()
    # pos = ml_graph.nodes_positions()

    few_gaps_barycenter_smart_sort(ml_graph)
    few_gaps_median_sort_improved(ml_graph)
    pgv_graph_median = ml_graph.to_pygraphviz_graph()
    pgv_graph_median.draw("median.svg")  # type: ignore # type of draw partially unknown

    # print(pgv_graph.string())  # print to screen
    # pgv_graph.write("simple.dot")  # write to simple.dot
    # pgv_graph.draw("plain.svg")
    # pgv_graph.layout(prog="dot")
    # pgv_graph.draw("dot.svg")


def draw_presentation_graph():
    # print(pgv_graph.string())  # print to screen
    # pgv_graph.write("simple.dot")  # write to simple.dot
    # pgv_graph.draw("plain.svg")
    ml_graph = MultiLayeredGraph(layer_count=3)
    node_f = ml_graph.add_real_node(0, "f")
    node_d = ml_graph.add_real_node(0, "d")
    node_h = ml_graph.add_real_node(0, "h")
    node_i = ml_graph.add_real_node(0, "i")

    node_j = ml_graph.add_real_node(1, "j")
    node_e = ml_graph.add_real_node(1, "e")
    node_g = ml_graph.add_real_node(1, "g")

    node_a = ml_graph.add_real_node(2, "a")
    node_b = ml_graph.add_real_node(2, "b")
    node_c = ml_graph.add_real_node(2, "c")

    ml_graph.add_edge(node_f, node_j)
    ml_graph.add_edge(node_h, node_j)
    ml_graph.add_edge(node_i, node_j)
    ml_graph.add_edge(node_h, node_g)
    ml_graph.add_edge(node_d, node_e)

    ml_graph.add_edge(node_d, node_a)
    ml_graph.add_edge(node_i, node_b)
    ml_graph.add_edge(node_i, node_c)

    ml_graph.add_edge(node_e, node_a)
    ml_graph.add_edge(node_e, node_b)
    ml_graph.add_edge(node_e, node_c)

    pgv_graph = ml_graph.to_pygraphviz_graph()

    pgv_graph.layout(prog="dot")
    pgv_graph.draw("presentation_dot.svg")  # type: ignore # type of draw partially unknown


if __name__ == "__main__":
    draw_presentation_graph()
