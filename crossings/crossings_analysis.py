import copy
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from crossing_minimization.barycenter_heuristic import (
    few_gaps_barycenter_smart_sort,
    few_gaps_barycenter_sort_naive,
)
from crossing_minimization.gurobi_int_lin import few_gaps_gurobi_wrapper
from crossing_minimization.k_gaps import k_gaps_barycenter
from crossing_minimization.median_heuristic import (
    few_gaps_median_sort_improved,
    few_gaps_median_sort_naive,
)
from crossings.crossing_analysis_visualization import (
    DataSet,
    GraphLabels,
    draw_crossing_analysis_graph,
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
        # todo test two layer cross minimization
        self.algorithms: list[NamedAlgorithm] = [
            NamedAlgorithm("Barycenter naive", few_gaps_barycenter_sort_naive),
            NamedAlgorithm("Barycenter improved", few_gaps_barycenter_smart_sort),
            NamedAlgorithm("Median naive", few_gaps_median_sort_naive),
            NamedAlgorithm("Median improved", few_gaps_median_sort_improved),
            NamedAlgorithm("Gurobi", few_gaps_gurobi_wrapper),
        ]
        self.algs_graphtype_time_ns: dict[
            NamedAlgorithm, dict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        self.algs_graphtype_crossings: dict[
            NamedAlgorithm, dict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        self.graph_type_names: list[str] = []

        # time taken to perform actions
        self.timings: dict[str, list[float]] = {
            "graph_gen": [],
            "deep_copies": [],
        }
        # for named_alg in self.algorithms:
        #     self.timings[named_alg.name] = []

    def analyse_crossings(self):
        # clear previous data
        self.algs_graphtype_time_ns.clear()
        self.algs_graphtype_crossings.clear()

        # run_parameters[i] = (layer2_count, layer1_count, virtual_nodes_count, regular_edges_count)
        two_layer_graph_parameters: list[tuple[int, int, int, int]] = [
            (7, 7, 7, 15),
            (10, 10, 10, 20),
            (10, 3, 10, 15),
            (10, 40, 10, 10),
            (40, 10, 10, 10),
        ]

        one_sided_algorithm_kwargs = {"one_sided_if_two_layers": True}
        two_sided_algorithm_kwargs = {
            "max_iterations": 3,
            "one_sided_if_two_layers": False,
        }
        for alg_kwargs in (one_sided_algorithm_kwargs, two_sided_algorithm_kwargs):
            try:
                for i, (
                    l1_count,
                    l2_count,
                    vnode_count,
                    reg_edges_count,
                ) in enumerate(two_layer_graph_parameters):
                    random_2_layer_graph = self._generate_random_two_layer_graph(
                        layer1_count=l1_count,
                        layer2_count=l2_count,
                        virtual_nodes_count=vnode_count,
                        regular_edges_count=reg_edges_count,
                    )
                    for algorithm in self.algorithms:
                        self._minimize_and_count_crossings(
                            random_2_layer_graph,
                            algorithm,
                            alg_kwargs,
                        )

                    print(f"Round {i}")
            except KeyboardInterrupt:
                # stop and show results so far
                print(f"Keyboard interrupt, stopping")

            self._print_crossing_results()

    def analyze_crossings_for_graph_two_layer(self):
        # clear previous data
        self.algs_graphtype_time_ns.clear()
        self.algs_graphtype_crossings.clear()

        # algorithms_to_test = [alg for alg in self.algorithms if alg.name != "Gurobi"]
        algorithms_to_test = [alg for alg in self.algorithms]

        two_sided_algorithm_kwargs = {
            "max_iterations": 3,
            "one_sided_if_two_layers": False,
        }

        iterations = 10
        rounds = 5
        scaling = [1 + 0.2 * round_nr for round_nr in range(rounds)]
        base_real_nodes = 8
        base_virtual_nodes = 4
        regular_edges_density = 0.1

        graph_x_values: list[float] = []
        for round_nr in range(rounds):
            real_nodes = int(base_real_nodes * scaling[round_nr])
            graph_x_values.append(real_nodes)
            virtual_nodes = int(base_virtual_nodes * scaling[round_nr])
            regular_edge_count = int(real_nodes * real_nodes * regular_edges_density)
            for iteration in range(iterations):
                print(
                    f"Round {round_nr}, iteration {iteration}. {real_nodes=}, {regular_edge_count=}"
                )

                random_2_layer_graph = self._generate_random_two_layer_graph(
                    layer1_count=real_nodes,
                    layer2_count=real_nodes,
                    virtual_nodes_count=virtual_nodes,
                    regular_edges_count=regular_edge_count,
                    override_name=f"Round {round_nr}",
                )

                for algorithm in algorithms_to_test:
                    self._minimize_and_count_crossings(
                        random_2_layer_graph,
                        algorithm,
                        two_sided_algorithm_kwargs,
                    )

        # print(dict(self.algs_graphtype_crossings))
        # print(dict(self.algs_graphtype_time_ns))
        data_sets: list[DataSet] = []
        for algorithm in algorithms_to_test:
            y_values: list[float] = []
            for round_nr in range(rounds):
                graph_name = f"Round {round_nr}"
                crossings = self.algs_graphtype_crossings[algorithm][graph_name]
                y_values.append(sum(crossings) / len(crossings))

            data_sets.append(DataSet(algorithm.name, y_values))

        graph_labels = GraphLabels(
            "# real nodes (half the amount of virtual nodes)",
            "crossings",
            "Crossings after minimization",
        )
        draw_crossing_analysis_graph(graph_x_values, data_sets, graph_labels)

    def test_correctness_k_gaps(self):
        # clear previous data
        self.algs_graphtype_time_ns.clear()
        self.algs_graphtype_crossings.clear()

        rounds_count = 10
        for round_nr in range(rounds_count):
            print(f"round {round_nr}")
            for gap_count in range(3, 5):
                random_2_layer_graph = self._generate_random_two_layer_graph(
                    layer1_count=50,
                    layer2_count=50,
                    virtual_nodes_count=30,
                    regular_edges_count=500,
                    override_name=f"Round {round_nr}, {gap_count=}",
                )
                g1 = copy.deepcopy(random_2_layer_graph.graph)
                g2 = copy.deepcopy(random_2_layer_graph.graph)

                t1 = time.perf_counter()
                k_gaps_barycenter(g1, one_sided_if_two_layers=True, gaps=gap_count)
                t2 = time.perf_counter()
                k_gaps_barycenter(g2, one_sided_if_two_layers=True, gaps=gap_count)
                t3 = time.perf_counter()

                if g1.get_crossings_per_layer() != g2.get_crossings_per_layer():
                    print(
                        f"DIFFERENT CROSSINGS!!!! {g1.get_crossings_per_layer()} != {g2.get_crossings_per_layer()}"
                    )
                else:
                    print(
                        f"got same result {g1.get_crossings_per_layer()} == {g2.get_crossings_per_layer()}"
                    )
                    print(f"superfluous took {t2-t1}s, simple took {t3-t2}")

    def _print_crossing_results(self):
        for graph_type in self.graph_type_names:
            print(f'For graph type "{graph_type}":')
            for named_alg in self.algorithms:
                total_crossings = sum(
                    self.algs_graphtype_crossings[named_alg][graph_type]
                )
                actual_runs = len(self.algs_graphtype_crossings[named_alg][graph_type])
                mean_crossings = total_crossings / actual_runs

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

    def _generate_random_two_layer_graph(
        self,
        layer1_count: int,
        layer2_count: int,
        virtual_nodes_count: int,
        regular_edges_count: int,
        *,
        override_name: str | None = None,
    ) -> GraphAndType:
        t1 = time.perf_counter_ns()

        ml_graph = multilayer_graph_generator.generate_two_layer_graph(
            layer1_count=layer1_count,
            layer2_count=layer2_count,
            virtual_nodes_count=virtual_nodes_count,
            regular_edges_count=regular_edges_count,
        )
        if override_name is None:
            name = f"2-layer random {layer1_count} x {layer2_count}; v={virtual_nodes_count}; e={regular_edges_count}"
        else:
            name = override_name

        self.graph_type_names.append(name)

        self.timings["graph_gen"].append(time.perf_counter_ns() - t1)
        return GraphAndType(graph=ml_graph, type_name=name)

    def _generate_random_graph(
        self,
        layers_count: int,
        node_count: int,
        edge_density: float,
        long_edge_probability: float,
        *,
        randomness_seed: int | None = None,
    ):
        graph_long_str = multilayer_graph_generator.random_graph_to_long_str(
            layers_count,
            node_count,
            edge_density,
            long_edge_probability,
            None,
        )
        self.graph_type_names.append(graph_long_str)
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
        graph_and_type = GraphAndType(graph=graph, type_name=graph_long_str)
        return graph_and_type

    def _minimize_and_count_crossings(
        self,
        graph_and_type: GraphAndType,
        named_alg: NamedAlgorithm,
        algorithm_kwargs: dict[str, Any],
    ):
        original_graph = graph_and_type.graph

        graph_copy: MultiLayeredGraph = copy.deepcopy(original_graph)
        self.time_algorithm_and_count_crossings(
            named_alg, graph_and_type, algorithm_kwargs
        )

        # optional validation step, can be left out if correctness is guaranteed
        try:
            self.assert_sorted_graph_valid(original_graph, graph_copy, 2)
        except AssertionError as err:
            print(f"WARNING!!! {named_alg.name} sorted the graph invalidly: {err}")

    def assert_sorted_graph_valid(
        self,
        orignal_graph: MultiLayeredGraph,
        sorted_graph: MultiLayeredGraph,
        gaps_allowed: int,
        only_side_gaps: bool = False,
    ):
        """Given an original graph, check if it's crossing-minimized graph is a valid permutation.

        Args:
            orignal_graph (MultiLayeredGraph): The original graph.
            sorted_graph (MultiLayeredGraph): The crossing-minimized graph.
            gaps_allowed (int): The amount of gaps allowed in the crossing-minimized graph.
        """

        def assert_with_message(evaluation: bool, msg: str):
            if not evaluation:
                raise AssertionError(msg)

        if gaps_allowed < 1:
            raise ValueError("At least one gap must be allowed")

        if only_side_gaps and gaps_allowed > 2:
            raise ValueError(
                "If only side gaps are allowed, gaps allowed per layer may not exceed 2"
            )

        assert_with_message(
            orignal_graph.layer_count == sorted_graph.layer_count, "Layer count differs"
        )
        for layer in range(orignal_graph.layer_count):
            og_nodes = orignal_graph.layers_to_nodes[layer]
            sorted_nodes = sorted_graph.layers_to_nodes[layer]
            assert_with_message(
                len(og_nodes) == len(sorted_nodes),
                f"Node count at {layer=} differs. {len(og_nodes)} != {len(sorted_nodes)}",
            )

            og_nodes_set = set(node.name for node in og_nodes)
            sorted_nodes_set = set(node.name for node in sorted_nodes)
            assert_with_message(og_nodes_set == sorted_nodes_set, f"Node names differ")

            gaps = 0
            # if only side gaps allowed, then make dummy-previous-node virtual, to enforce side gap
            prev_node_type_is_virtual = only_side_gaps
            for node in sorted_nodes:
                if node.is_virtual:
                    if not prev_node_type_is_virtual:
                        prev_node_type_is_virtual = True
                        gaps += 1
                else:
                    prev_node_type_is_virtual = False
            if only_side_gaps:
                if gaps == gaps_allowed:
                    assert_with_message(
                        prev_node_type_is_virtual == True,
                        f"Produced {gaps} as algorithm should, however they were not placed on the sides",
                    )
            assert_with_message(
                gaps <= gaps_allowed,
                f"Produced {gaps} gaps, but only {gaps_allowed} gaps allowed",
            )

    def time_algorithm_and_count_crossings(
        self,
        named_alg: NamedAlgorithm,
        graph_and_type: GraphAndType,
        algorithm_kwargs: dict[str, Any],
    ):
        graph = graph_and_type.graph
        graph_name = graph_and_type.type_name

        start_ns = time.perf_counter_ns()
        named_alg.algorithm(graph, **algorithm_kwargs)
        total_time_ns = time.perf_counter_ns() - start_ns

        self.algs_graphtype_time_ns[named_alg][graph_name].append(total_time_ns)

        crossing_count = graph.get_total_crossings()
        self.algs_graphtype_crossings[named_alg][graph_name].append(crossing_count)


if __name__ == "__main__":
    # CrossingsAnalyser().analyse_crossings()
    # CrossingsAnalyser().analyze_crossings_for_graph_two_layer()
    CrossingsAnalyser().test_correctness_k_gaps()
