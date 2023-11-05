# run using `python -m crossings.crossings_analysis`
import copy
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TypedDict

from crossing_analysis_old.crossing_analysis_visualization import (
    DataSet,
    GraphLabels,
    draw_crossing_analysis_graph,
)
from crossing_minimization.barycenter_heuristic import (
    BarycenterImprovedSorter,
    BarycenterNaiveSorter,
    BarycenterThesisSorter,
)
from crossing_minimization.gurobi_int_lin import GurobiHeuristicSorter, GurobiSorter
from crossing_minimization.median_heuristic import (
    ImprovedMedianSorter,
    NaiveMedianSorter,
    ThesisMedianSorter,
)
from crossing_minimization.utils import GraphSorter
from multilayered_graph import multilayer_graph_generator
from multilayered_graph.multilayered_graph import MultiLayeredGraph

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)
DRAW_GRAPH = False


class SortGraphArgs(TypedDict):
    max_iterations: int
    only_one_up_iteration: bool
    side_gaps_only: bool
    max_gaps: int


@dataclass(frozen=False, slots=True)
class GraphAndType:
    graph: MultiLayeredGraph
    type_name: str

    def __eq__(self, other: Any):
        return self is other

    def __str__(self):
        return self.type_name


class CrossingsAnalyser:
    def __init__(self):
        self.algorithms: list[type[GraphSorter]] = [
            GurobiSorter,
            BarycenterImprovedSorter,
            BarycenterNaiveSorter,
            ImprovedMedianSorter,
            NaiveMedianSorter,
        ]
        self.algs_graphtype_time_ns: dict[
            type[GraphSorter], dict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        self.algs_graphtype_crossings: dict[
            type[GraphSorter], dict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        self.graph_type_names: set[str] = set()

        # time taken to perform actions
        self.timings: dict[str, list[float]] = {}
        # for algorithm in self.algorithms:
        #     self.timings[GraphSorter_type.algorithm_name] = []

    def compare_improved_to_thesis_sorter(self) -> None:
        two_layer_graph_parameters: list[tuple[int, int, int, int]] = [
            # (10, 10, 5, 15),
            (20, 20, 10, 50),
            # (70, 70, 35, 150),
        ]
        two_layer_graph_parameters *= 40
        oscm_sg_algorithm_kwargs: SortGraphArgs = {
            "only_one_up_iteration": True,
            "side_gaps_only": True,
            "max_iterations": 1,
            "max_gaps": 2,
        }
        # oscm_kg_algorithm_kwargs: SortGraphArgs = {
        #     "only_one_up_iteration": True,
        #     "side_gaps_only": False,
        #     "max_iterations": 1,
        #     "max_gaps": 3,
        # }
        # two_sided_algorithm_kwargs: SortGraphArgs = {
        #     "only_one_up_iteration": False,
        #     "side_gaps_only": True,
        #     "max_iterations": 5,
        #     "max_gaps": 2,
        # }
        for i, (
            l1_count,
            l2_count,
            vnode_count,
            reg_edges_count,
        ) in enumerate(two_layer_graph_parameters):
            for sort_kwargs in (
                oscm_sg_algorithm_kwargs,
                # oscm_kg_algorithm_kwargs,
                # two_sided_algorithm_kwargs,
            ):
                for ThesisSorter, OGSorter in (
                    (BarycenterThesisSorter, BarycenterImprovedSorter),
                    (ThesisMedianSorter, ImprovedMedianSorter),
                    (GurobiHeuristicSorter, GurobiSorter),
                ):
                    logger.debug("%s vs %s", ThesisSorter.__name__, OGSorter.__name__)
                    random_2_layer_graph = self._generate_random_two_layer_graph(
                        layer1_count=l1_count,
                        layer2_count=l2_count,
                        virtual_nodes_count=vnode_count,
                        regular_edges_count=reg_edges_count,
                    )

                    thesis_min_graph = random_2_layer_graph
                    og_min_graph = copy.deepcopy(random_2_layer_graph)
                    self._minimize_and_count_crossings(
                        thesis_min_graph,
                        ThesisSorter,
                        sort_kwargs,
                    )
                    self._minimize_and_count_crossings(
                        og_min_graph,
                        OGSorter,
                        sort_kwargs,
                    )

                    thesis_crossings = thesis_min_graph.graph.get_total_crossings()
                    og_crossings = og_min_graph.graph.get_total_crossings()
                    if thesis_crossings != og_crossings:
                        logger.warning(
                            "THESIS MINIMIZATION CAUSED MORE CROSSINGS: %s > %s",
                            thesis_crossings,
                            og_crossings,
                        )
                    else:
                        logger.debug(
                            "Crossings: %s == %s", thesis_crossings, og_crossings
                        )

                    for layer_idx in range(thesis_min_graph.graph.layer_count):
                        assert str(
                            thesis_min_graph.graph.layers_to_nodes[layer_idx]
                        ) == str(og_min_graph.graph.layers_to_nodes[layer_idx])
                        logger.debug(
                            "Layer %d: both graphs have identical orders", layer_idx
                        )

                logger.info("Round %d", i)

    def analyse_crossings_side_gaps(self):
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

        one_sided_algorithm_kwargs: SortGraphArgs = {
            "only_one_up_iteration": True,
            "side_gaps_only": True,
            "max_iterations": 1,
            "max_gaps": 2,
        }
        two_sided_algorithm_kwargs: SortGraphArgs = {
            "only_one_up_iteration": False,
            "side_gaps_only": True,
            "max_iterations": 3,
            "max_gaps": 2,
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

        # algorithms_to_test = [alg for alg in self.algorithms if alg.algorithm_name != "Gurobi"]
        algorithms_to_test = [alg for alg in self.algorithms]

        two_sided_algorithm_kwargs: SortGraphArgs = {
            "max_iterations": 3,
            "only_one_up_iteration": False,
            "side_gaps_only": True,
            "max_gaps": 2,
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

                for algorithm_class in algorithms_to_test:
                    self._minimize_and_count_crossings(
                        random_2_layer_graph,
                        algorithm_class,
                        two_sided_algorithm_kwargs,
                    )

        # print(dict(self.algs_graphtype_crossings))
        # print(dict(self.algs_graphtype_time_ns))
        data_sets: list[DataSet] = []
        for algorithm_class in algorithms_to_test:
            y_values: list[float] = []
            for round_nr in range(rounds):
                graph_name = f"Round {round_nr}"
                crossings = self.algs_graphtype_crossings[algorithm_class][graph_name]
                y_values.append(sum(crossings) / len(crossings))

            data_sets.append(DataSet(algorithm_class.algorithm_name, y_values))

        graph_labels = GraphLabels(
            "# real nodes (half the amount of virtual nodes)",
            "crossings",
            "Crossings after minimization",
        )
        draw_crossing_analysis_graph(graph_x_values, data_sets, graph_labels)

    def analyze_crossings_k_gaps(self):
        # algorithms_to_test = [alg for alg in self.algorithms if alg != GurobiSorter]
        algorithms_to_test = self.algorithms
        algorithm_kwargs_kgaps: SortGraphArgs = {
            "max_iterations": 1,
            "only_one_up_iteration": True,
            "side_gaps_only": False,
            "max_gaps": 3,
        }
        try:
            for round_nr in range(1):
                graph_and_type = self._generate_random_two_layer_graph(
                    layer1_count=10,
                    layer2_count=10,
                    virtual_nodes_count=10,
                    regular_edges_count=20,
                )

                for algorithm in algorithms_to_test:
                    sorted_graph = self._minimize_and_count_crossings(
                        graph_and_type, algorithm, algorithm_kwargs_kgaps
                    )
                    if DRAW_GRAPH:
                        sorted_graph.to_pygraphviz_graph().draw(f"0{round_nr}-kgaps-{algorithm.__name__}.svg", "svg")  # type: ignore # unknown

                print(f"{round_nr=}")
        except KeyboardInterrupt:
            # stop and show results so far
            print(f"Keyboard interrupt, stopping")
        self._print_crossing_results(algorithms_to_test)

    def temp_test_gurobi_side_gaps(self):
        algorithm_kwargs_kgaps: SortGraphArgs = {
            "max_iterations": 1,
            "only_one_up_iteration": True,
            "side_gaps_only": True,
            "max_gaps": 2,
        }

        run_count = 20

        for round_nr in range(run_count):
            graph_and_type = self._generate_random_two_layer_graph(
                layer1_count=8,
                layer2_count=10,
                virtual_nodes_count=10,
                regular_edges_count=20,
            )
            sorted_graph = self._minimize_and_count_crossings(
                graph_and_type,
                GurobiSorter,
                {
                    **algorithm_kwargs_kgaps,
                    "temp_debug_use_reduced_model_virtual_nodes": False,  # type: ignore
                },
            )
            sorted_graph_reduced = self._minimize_and_count_crossings(
                graph_and_type,
                GurobiSorter,
                {
                    **algorithm_kwargs_kgaps,
                    "temp_debug_use_reduced_model_virtual_nodes": True,  # type: ignore
                },
            )
            assert (
                sorted_graph.get_total_crossings()
                == sorted_graph_reduced.get_total_crossings()
            )
            print(f"{round_nr=}, same crossings")

    def _print_crossing_results(
        self, algorithms: list[type[GraphSorter]] | None = None
    ):
        if algorithms is None:
            algorithms = self.algorithms
        for graph_type in self.graph_type_names:
            print(f'For graph type "{graph_type}":')
            for GraphSorter_type in algorithms:
                total_crossings = sum(
                    self.algs_graphtype_crossings[GraphSorter_type][graph_type]
                )
                actual_runs = len(
                    self.algs_graphtype_crossings[GraphSorter_type][graph_type]
                )
                mean_crossings = total_crossings / actual_runs

                print(
                    f"\t{str(GraphSorter_type.__name__):<30} had mean crossing count of {mean_crossings:>8.2f}"
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

        self.graph_type_names.add(name)

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
        self.graph_type_names.add(graph_long_str)
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
        original_graph_and_type: GraphAndType,
        GraphSorter_type: type[GraphSorter],
        algorithm_kwargs: SortGraphArgs,
    ) -> MultiLayeredGraph:
        """Minimizes crossings, and returns the crossing-minimized graph.

        Also check validity of sorted graph e.g. if amount of gaps is complied with.

        Args:
            graph_and_type (GraphAndType): Graph to sort.
            GraphSorter_type (type[GraphSorter]): Sorter with which to sort nodes.
            algorithm_kwargs (SortGraphArgs): Keyword arguments to pass to sorter.

        Returns:
            MultiLayeredGraph: A copy of the graph with nodes sorted
        """
        original_graph = original_graph_and_type.graph

        graph_copy: MultiLayeredGraph = copy.deepcopy(original_graph)
        self._time_algorithm_and_count_crossings(
            GraphSorter_type,
            GraphAndType(graph_copy, original_graph_and_type.type_name),
            algorithm_kwargs,
        )

        # optional validation step, can be left out if correctness is guaranteed
        try:
            self._assert_sorted_graph_valid(
                original_graph,
                graph_copy,
                gaps_allowed=algorithm_kwargs["max_gaps"],
                only_side_gaps=algorithm_kwargs["side_gaps_only"],
            )
            # print(f"graph valid")
        except AssertionError as err:
            print(
                f"WARNING!!! {GraphSorter_type.algorithm_name} sorted the graph invalidly: {err}"
            )
        return graph_copy

    def _assert_sorted_graph_valid(
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

        if gaps_allowed < 1:
            raise ValueError("At least one gap must be allowed")

        if only_side_gaps and gaps_allowed > 2:
            raise ValueError(
                "If only side gaps are allowed, gaps allowed per layer may not exceed 2"
            )

        assert (
            orignal_graph.layer_count == sorted_graph.layer_count
        ), "Layer count differs"
        for layer in range(orignal_graph.layer_count):
            og_nodes = orignal_graph.layers_to_nodes[layer]
            sorted_nodes = sorted_graph.layers_to_nodes[layer]
            assert len(og_nodes) == len(
                sorted_nodes
            ), f"Node count at {layer=} differs. {len(og_nodes)} != {len(sorted_nodes)}"

            og_nodes_set = set(node.name for node in og_nodes)
            sorted_nodes_set = set(node.name for node in sorted_nodes)
            assert og_nodes_set == sorted_nodes_set, f"Node names differ"

            gaps = 0
            # if only side gaps allowed, then make dummy-previous-node virtual, to enforce side gap
            prev_node_type_is_virtual = False
            for node in sorted_nodes:
                if node.is_virtual:
                    if not prev_node_type_is_virtual:
                        prev_node_type_is_virtual = True
                        gaps += 1
                else:
                    prev_node_type_is_virtual = False

            assert (
                gaps <= gaps_allowed
            ), f"Produced {gaps} gaps, but only {gaps_allowed} gaps allowed"

            if only_side_gaps:
                only_side_violation_msg = f"Produced {gaps} as algorithm should, however they were not placed on the sides"
                if gaps == 1:
                    assert (
                        sorted_nodes[0].is_virtual or sorted_nodes[-1].is_virtual
                    ), only_side_violation_msg
                elif gaps == 2:
                    assert (
                        sorted_nodes[0].is_virtual and sorted_nodes[-1].is_virtual
                    ), only_side_violation_msg

    def _time_algorithm_and_count_crossings(
        self,
        sorter_class: type[GraphSorter],
        graph_and_type: GraphAndType,
        algorithm_kwargs: SortGraphArgs,
    ):
        graph = graph_and_type.graph
        graph_name = graph_and_type.type_name

        start_ns = time.perf_counter_ns()
        sorter_class.sort_graph(graph, **algorithm_kwargs)
        total_time_ns = time.perf_counter_ns() - start_ns

        self.algs_graphtype_time_ns[sorter_class][graph_name].append(total_time_ns)

        crossing_count = graph.get_total_crossings()
        self.algs_graphtype_crossings[sorter_class][graph_name].append(crossing_count)


if __name__ == "__main__":
    # CrossingsAnalyser().analyse_crossings_side_gaps()
    # CrossingsAnalyser().analyze_crossings_for_graph_two_layer()
    # CrossingsAnalyser().temp_test_gurobi_side_gaps()
    # CrossingsAnalyser().temp_test_gurobi_k_gaps()
    CrossingsAnalyser().compare_improved_to_thesis_sorter()
