import copy
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from crossing_minimization.barycenter_heuristic import (
    few_gaps_barycenter_smart_sort,
    few_gaps_barycenter_sort_naive,
)
from crossing_minimization.constrained_implementation import few_gaps_constrained_paper
from crossing_minimization.gurobi_int_lin import few_gaps_gurobi
from crossing_minimization.median_heuristic import (
    few_gaps_median_sort_improved,
    few_gaps_median_sort_naive,
)
from multilayered_graph import multilayer_graph_generator
from multilayered_graph.multilayered_graph import MultiLayeredGraph


@dataclass(frozen=True, slots=True)
class NamedAlgorithm:
    name: str
    algorithm: Callable[[MultiLayeredGraph], None]

    def __eq__(self, other: Any):
        return self is other

    def __str__(self):
        return self.name


@dataclass(frozen=True, slots=True)
class GraphAndType:
    graph: MultiLayeredGraph
    type_name: str

    def __eq__(self, other: Any):
        return self is other

    def __str__(self):
        return self.type_name


class CrossingsAnalyser:
    def __init__(self):
        self.algorithms: list[NamedAlgorithm] = [
            NamedAlgorithm("Barycenter naive", few_gaps_barycenter_sort_naive),
            NamedAlgorithm("Barycenter improved", few_gaps_barycenter_smart_sort),
            # NamedAlgorithm("Barycenter split", few_gaps_barycenter_split), # identical to improved
            NamedAlgorithm("Median naive", few_gaps_median_sort_naive),
            NamedAlgorithm("Median improved", few_gaps_median_sort_improved),
            NamedAlgorithm(
                "Constrained nodes (paper impl)", few_gaps_constrained_paper
            ),
            NamedAlgorithm("Gurobi", few_gaps_gurobi),
        ]
        self.algs_graphtype_crossings: dict[NamedAlgorithm, dict[str, list[int]]] = {}
        self.graph_type_names: dict[str, str] = {}
        self.timings: dict[str, list[float]] = {
            # "name_gen": [],
            "graph_gen": [],
            # "reorg": [],
            "deep_copies": [],
        }
        for named_alg in self.algorithms:
            self.timings[named_alg.name] = []

    def analyse_crossings(self):
        self.algs_graphtype_crossings = {
            alg: defaultdict(list) for alg in self.algorithms
        }

        try:
            for i in range(4):
                # random_params = {
                #     "node_count": 30,
                #     "layers_count": 4,
                #     "edge_density": 0.02,
                #     "long_edge_probability": 0.3,
                # }
                two_layer_params = {
                    "layer1_count": 20,
                    "layer2_count": 10,
                    "virtual_nodes_count": 20,
                    "regular_edges_count": 400,
                }
                # two_layer_params = {
                #     "layer1_count": 3,
                #     "layer2_count": 3,
                #     "virtual_nodes_count": 3,
                #     "regular_edges_count": 8,
                # }
                # self._test_random_graph(**random_params, randomness_seed=i)
                #
                # random_params["long_edge_probability"] = 0.7
                # self._test_random_graph(**random_params, randomness_seed=i)
                #
                # random_params["layers_count"] = 2

                # self._reorganize_graph_and_count_crossings(
                #     self._gen_random_graph(**random_params, randomness_seed=i),
                #     self.algs_graphtype_crossings,
                # )
                self._reorganize_graph_and_count_crossings(
                    self._gen_two_layer_graph(**two_layer_params),
                    self.algs_graphtype_crossings,
                )
                #
                # random_params["layers_count"] = 8
                # self._test_random_graph(**random_params, randomness_seed=i)
                print(f"Round {i}")
        except KeyboardInterrupt:
            print(f"Keyboard interrupt, stopping")
            # stop and show results so far

        for graph_type in self.graph_type_names.values():
            print(f'For graph type "{graph_type}":')
            for named_alg in self.algorithms:
                total_crossings = sum(
                    self.algs_graphtype_crossings[named_alg][graph_type]
                )
                runs = len(self.algs_graphtype_crossings[named_alg][graph_type])
                mean_crossings = total_crossings / runs

                print(
                    f"\t{str(named_alg):<30} had mean crossing count of {mean_crossings:>8.2f}"
                )

        total_time_seconds = sum(sum(t) for t in self.timings.values()) / 1_000_000_000
        print(f"Total time taken: {total_time_seconds:.2f}s")
        for timing_name, times in self.timings.items():
            total_time_ms = sum(times) / 1_000_000
            # avg_time = total_time_ms / len(times)
            avg_time_str = f"{total_time_ms / len(times):.3f}"
            print(f"{timing_name:>30} took avg: {avg_time_str:>7}ms")

    def _gen_two_layer_graph(
        self,
        layer1_count: int,
        layer2_count: int,
        virtual_nodes_count: int,
        regular_edges_count: int,
    ) -> GraphAndType:
        t1 = time.perf_counter_ns()

        ml_graph = multilayer_graph_generator.generate_two_layer_graph(
            layer1_count=layer1_count,
            layer2_count=layer2_count,
            virtual_nodes_count=virtual_nodes_count,
            regular_edges_count=regular_edges_count,
        )
        name = f"2-layer random {layer1_count} x {layer2_count}; v={virtual_nodes_count}; e={regular_edges_count}"

        self.graph_type_names[name] = name

        self.timings["graph_gen"].append(time.perf_counter_ns() - t1)
        return GraphAndType(graph=ml_graph, type_name=name)

    def _gen_random_graph(
        self,
        layers_count: int,
        node_count: int,
        edge_density: float,
        long_edge_probability: float,
        *,
        randomness_seed: int | None = None,
    ):
        (
            graph_short_str,
            graph_long_str,
        ) = multilayer_graph_generator.random_graph_to_short_long_str(
            layers_count,
            node_count,
            edge_density,
            long_edge_probability,
            None,
        )
        self.graph_type_names[graph_short_str] = graph_long_str
        perf_timer_name_gen__graph_gen = time.perf_counter_ns()

        graph: MultiLayeredGraph = multilayer_graph_generator.generate_multilayer_graph(
            layers_count=layers_count,
            node_count=node_count,
            edge_density=edge_density,
            long_edge_probability=long_edge_probability,
            randomness_seed=randomness_seed,
        )
        perf_timer_graph_gen__reorg = time.perf_counter_ns()
        self.timings["graph_gen"].append(
            perf_timer_graph_gen__reorg - perf_timer_name_gen__graph_gen
        )
        graph_and_type = GraphAndType(
            graph=graph, type_name=self.graph_type_names[graph_short_str]
        )
        return graph_and_type

    def _reorganize_graph_and_count_crossings(
        self,
        graph_and_type: GraphAndType,
        algorithms_to_graph_structure_to_crossings: dict[
            NamedAlgorithm, dict[str, list[int]]
        ],
    ):
        perf_timer_start__deep_copy = time.perf_counter_ns()

        graph = graph_and_type.graph
        graph_type = graph_and_type.type_name
        graph_copies: list[MultiLayeredGraph] = [
            copy.deepcopy(graph)
            for _ in range(len(algorithms_to_graph_structure_to_crossings) - 1)
        ]
        graph_copies.append(graph)

        self.timings["deep_copies"].append(
            time.perf_counter_ns() - perf_timer_start__deep_copy
        )

        perf_timer_end_prev_alg = time.perf_counter_ns()

        # for some reason zip has faulty type hinting, so type hint here:
        named_alg: NamedAlgorithm
        graph_copy: MultiLayeredGraph
        for named_alg, graph_copy in zip(
            algorithms_to_graph_structure_to_crossings.keys(), graph_copies, strict=True
        ):
            named_alg.algorithm(graph_copy)
            crossing_count = graph_copy.get_total_crossings()
            algorithms_to_graph_structure_to_crossings[named_alg][graph_type].append(
                crossing_count
            )
            perf_timer_end_curr_alg = time.perf_counter_ns()
            self.timings[named_alg.name].append(
                perf_timer_end_curr_alg - perf_timer_end_prev_alg
            )
            perf_timer_end_prev_alg = perf_timer_end_curr_alg

            # graph_copy.to_pygraphviz_graph().draw(f"{named_alg.name}-re.svg")
            # print(f"{named_alg.name} produced {crossing_count} crossings")


if __name__ == "__main__":
    CrossingsAnalyser().analyse_crossings()
