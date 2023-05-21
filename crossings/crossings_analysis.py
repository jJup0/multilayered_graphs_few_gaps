import copy
from dataclasses import dataclass
from typing import List, Tuple, Callable

from multilayered_graph.multilayer_graph_generator import generate_multilayer_graph
from multilayered_graph.multilayered_graph import MultiLayeredGraph
from node_sorting.barycenter_heuristic import (
    few_gaps_barycenter_smart_sort,
    few_gaps_barycenter_sort,
)


@dataclass(frozen=True, slots=True)
class NamedAlgorithm:
    name: str
    algorithm: Callable[[MultiLayeredGraph], None]

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.name


@dataclass(frozen=True, slots=True)
class GraphAndType:
    # TODO
    # name: str
    # algorithm: Callable[[MultiLayeredGraph], None]

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.name


def analyse_crossings():
    # graphs_to_test_on: list[MultiLayeredGraph] = []

    algorithms: list[NamedAlgorithm] = [
        NamedAlgorithm("Barycenter smart", few_gaps_barycenter_smart_sort),
        NamedAlgorithm("Barycenter naive", few_gaps_barycenter_sort),
    ]

    algorithms_to_graph_structure_to_crossings: dict[
        MultiLayeredGraph, dict[str, list[int]]
    ] = {alg: {} for alg in algorithms}

    for i in range(100):
        graph: MultiLayeredGraph = generate_multilayer_graph(
            8, 20, 0.2, 0.3, randomness_seed=i
        )
        # graphs_to_test_on.append(graph)
        _temp(graph, algorithms_to_graph_structure_to_crossings)

    for named_alg in algorithms:
        total_crossings = sum(algorithms_to_graph_structure_to_crossings[named_alg])
        runs = len(algorithms_to_graph_structure_to_crossings[named_alg])
        mean_crossings = total_crossings / runs

        print(f"{named_alg} had mean crossing count of {mean_crossings:.2f}")


def _temp(
    algorithms_to_graph_structure_to_crossings: dict[
        NamedAlgorithm, dict[str, list[int]]
    ],
    graph: MultiLayeredGraph,
):
    graph_copies: list[MultiLayeredGraph] = [
        copy.deepcopy(graph)
        for _ in range(len(algorithms_to_graph_structure_to_crossings) - 1)
    ]
    graph_copies.append(graph)

    # for some reason zip has faulty type hinting?
    named_alg: NamedAlgorithm
    graph_copy: MultiLayeredGraph
    for named_alg, graph_copy in zip(
        algorithms_to_graph_structure_to_crossings.keys(), graph_copies, strict=True
    ):
        named_alg.algorithm(graph_copy)
        crossing_count = graph_copy.get_total_crossings()
        algorithms_to_graph_structure_to_crossings[named_alg].append(crossing_count)
        # print(f"{named_alg.name} produced {crossing_count} crossings")


if __name__ == "__main__":
    analyse_crossings()
