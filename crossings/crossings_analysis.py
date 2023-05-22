import copy
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from multilayered_graph import multilayer_graph_generator
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
    graph: MultiLayeredGraph
    type_name: str

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.type_name


class CrossingsAnalyser:
    def __init__(self):
        self.algorithms: list[NamedAlgorithm] = [
            NamedAlgorithm("Barycenter smart", few_gaps_barycenter_smart_sort),
            NamedAlgorithm("Barycenter naive", few_gaps_barycenter_sort),
        ]
        self.algs_graphtype_crossings: dict[NamedAlgorithm, dict[str, list[int]]] = {}
        self.graph_type_names: dict[str, str] = {}
        self.timings: dict[str, list[float]] = {
            "name_gen": [],
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

        for i in range(20):
            random_params = {
                "node_count": 20,
                "layers_count": 8,
                "edge_density": 0.1,
                "long_edge_probability": 0.3,
            }

            self._test_random_graph(**random_params, randomness_seed=i)

            random_params["long_edge_probability"] = 0.7
            self._test_random_graph(**random_params, randomness_seed=i)

            random_params["layers_count"] = 2
            self._test_random_graph(**random_params, randomness_seed=i)
            print(f"Round {i}")

        for graph_type in self.graph_type_names.values():
            for named_alg in self.algorithms:
                total_crossings = sum(
                    self.algs_graphtype_crossings[named_alg][graph_type]
                )
                runs = len(self.algs_graphtype_crossings[named_alg][graph_type])
                mean_crossings = total_crossings / runs

                print(
                    f'{named_alg} for graph type "{graph_type}" had mean crossing count of {mean_crossings:.2f}'
                )

        total_time_seconds = sum(sum(t) for t in self.timings.values()) / 1_000_000_000
        print(f"Total time taken: {total_time_seconds:.2f}s")
        for timing_name, times in self.timings.items():
            total_time_ms = sum(times) / 1_000_000
            # avg_time = total_time_ms / len(times)
            avg_time_str = f"{total_time_ms / len(times):.3f}"
            print(f"{timing_name:>20} took avg: {avg_time_str:>7}ms")

    def _test_random_graph(
        self,
        layers_count: int,
        node_count: int,
        edge_density: float,
        long_edge_probability: float,
        *,
        randomness_seed: int | None = None,
    ):
        perf_timer_start__name_gen = time.perf_counter_ns()
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

        graph_and_type = GraphAndType(
            graph=graph, type_name=self.graph_type_names[graph_short_str]
        )
        # graphs_to_test_on.append(graph)
        self._reorganize_graph_and_count_crossings(
            graph_and_type, self.algs_graphtype_crossings
        )
        # perf_timer_reorg__end = time.perf_counter_ns()

        # note timings
        self.timings["name_gen"].append(
            perf_timer_name_gen__graph_gen - perf_timer_start__name_gen
        )
        self.timings["graph_gen"].append(
            perf_timer_graph_gen__reorg - perf_timer_name_gen__graph_gen
        )
        # self.timings["reorg"].append(
        #     perf_timer_reorg__end - perf_timer_graph_gen__reorg
        # )

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
            # print(f"{named_alg.name} produced {crossing_count} crossings")


if __name__ == "__main__":
    CrossingsAnalyser().analyse_crossings()
