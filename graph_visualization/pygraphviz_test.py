from multilayered_graph.multilayer_graph_generator import generate_multilayer_graph
from node_sorting.barycenter_heuristic import (
    few_gaps_barycenter_smart_sort,
)
from node_sorting.median_heuristic import few_gaps_median_sort_improved


def draw_graph():
    ml_graph = generate_multilayer_graph(5, 16, 0.2, 0.5, randomness_seed=1)

    # first_node = ml_graph.all_nodes_as_list()[0]
    #
    # pgv_graph = ml_graph.to_pygraphviz_graph()
    # pos = ml_graph.nodes_positions()

    few_gaps_barycenter_smart_sort(ml_graph)
    few_gaps_median_sort_improved(ml_graph)
    pgv_graph_median = ml_graph.to_pygraphviz_graph()
    pgv_graph_median.draw("median.svg")

    # print(pgv_graph.string())  # print to screen
    # pgv_graph.write("simple.dot")  # write to simple.dot
    # pgv_graph.draw("plain.svg")
    # pgv_graph.layout(prog="dot")
    # pgv_graph.draw("dot.svg")


if __name__ == "__main__":
    draw_graph()
