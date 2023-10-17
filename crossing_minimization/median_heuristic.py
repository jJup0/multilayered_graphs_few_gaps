import statistics
from abc import abstractmethod

from crossing_minimization.k_gaps import k_gaps_sort_whole_graph
from crossing_minimization.utils import (
    DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    Above_or_below_T,
    GraphSorter,
    generate_layers_to_above_or_below,
    get_graph_neighbors_from_above_or_below,
    get_layer_idx_above_or_below,
    thesis_side_gaps,
)
from crossings.calculate_crossings import crossings_uv_vu
from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

PSEUDO_SORT_DISPLACE_VALUE = 1_000_000


class AbstractMedianSorter(GraphSorter):
    @classmethod
    def _sort_graph(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int,
        only_one_up_iteration: bool,
        side_gaps_only: bool,
        max_gaps: int,
    ) -> None:
        if side_gaps_only:
            return cls._median_side_gaps(
                ml_graph,
                max_iterations=max_iterations,
                only_one_up_iteration=only_one_up_iteration,
            )
        return k_gaps_sort_whole_graph(
            ml_graph,
            max_iterations=max_iterations,
            only_one_up_iteration=only_one_up_iteration,
            max_gaps=max_gaps,
            get_median_or_barycenter=get_median,
        )

    @classmethod
    @abstractmethod
    def _median_side_gaps(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int,
        only_one_up_iteration: bool,
    ) -> None:
        ...


class ThesisMedianSorter(AbstractMedianSorter):
    @classmethod
    def _median_side_gaps(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int,
        only_one_up_iteration: bool,
    ):
        thesis_side_gaps(
            ml_graph,
            max_iterations=max_iterations,
            only_one_up_iteration=only_one_up_iteration,
            get_median_or_barycenter=get_median,
        )


class ImprovedMedianSorter(AbstractMedianSorter):
    """Deprecated Sorter"""

    algorithm_name = "Median improved"

    @classmethod
    def _median_side_gaps(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
        only_one_up_iteration: bool = False,
    ) -> None:
        """Sorts nodes in multilayered graph according to median heuristic.

        Modified in place. Improved virtual node placement.
        Args:
            ml_graph: Graph on which to apply sorting.
            max_iterations:
            Amount of "up" and "down" cycles to make for sorting. Defaults to 3.
            only_one_up_iteration:
            Whether to only do one sided crossing minimization or not.
            Defaults to False.
        """

        layer_to_real_nodes: dict[int, list[MLGNode]] = {}
        for layer_idx, nodes in ml_graph.layers_to_nodes.items():
            real_nodes = [n for n in nodes if not n.is_virtual]
            layer_to_real_nodes[layer_idx] = real_nodes

        for layer, above_or_below in generate_layers_to_above_or_below(
            ml_graph,
            max_iterations=max_iterations,
            only_one_up_iteration=only_one_up_iteration,
        ):
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(
                get_layer_idx_above_or_below(layer, above_or_below)
            )
            node_to_neighbors = get_graph_neighbors_from_above_or_below(
                ml_graph, above_or_below
            )
            ml_graph.layers_to_nodes[layer] = sorted(
                ml_graph.layers_to_nodes[layer],
                key=lambda node: cls._get_pseudo_median_sort_val_improved(
                    node=node,
                    neighbors=node_to_neighbors[node],
                    prev_layer_indices=prev_layer_indices,
                    ml_graph=ml_graph,
                    above_or_below=above_or_below,  # type: ignore # for some reason saying above_or_below is arbitrary string
                    real_nodes_at_layer=layer_to_real_nodes[layer],
                ),
            )

    @classmethod
    def _get_pseudo_median_sort_val_improved(
        cls,
        node: MLGNode,
        neighbors: set[MLGNode],
        prev_layer_indices: dict[MLGNode, int],
        ml_graph: MultiLayeredGraph,
        real_nodes_at_layer: list[MLGNode],
        above_or_below: Above_or_below_T,
    ) -> float:
        median = get_median(ml_graph, node, neighbors, prev_layer_indices)

        if node.is_virtual:
            real_node_crossings_if_left = 0
            real_node_crossings_if_right = 0
            for real_node in real_nodes_at_layer:
                left_crossings, right_crossings = crossings_uv_vu(
                    ml_graph, node, real_node, above_or_below
                )
                real_node_crossings_if_left += left_crossings
                real_node_crossings_if_right += right_crossings

            if real_node_crossings_if_left < real_node_crossings_if_right:
                median -= PSEUDO_SORT_DISPLACE_VALUE
            else:
                median += PSEUDO_SORT_DISPLACE_VALUE

        return median


class NaiveMedianSorter(AbstractMedianSorter):
    """Deprecated Sorter"""

    algorithm_name = "Median naive"

    @classmethod
    def _median_side_gaps(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int,
        only_one_up_iteration: bool,
    ) -> None:
        """Sorts nodes in multilayered graph according to median heuristic.

        Modified in place. Naive virtual node placement.
        Args:
            ml_graph: Graph on which to apply sorting.
        """

        def _sort_layer(
            _layer_idx: int,
            _prev_layer_indices: dict[MLGNode, int],
            node_to_neighbors: dict[MLGNode, set[MLGNode]],
        ):
            nonlocal ml_graph
            layer_before_sorting = ml_graph.layers_to_nodes[_layer_idx][:]
            real_node_median_median = cls.get_real_node_median_median(
                ml_graph,
                nodes_at_layer=layer_before_sorting,
                node_to_neighbors=node_to_neighbors,
                prev_layer_indices=_prev_layer_indices,
            )

            ml_graph.layers_to_nodes[_layer_idx] = sorted(
                ml_graph.layers_to_nodes[_layer_idx],
                key=lambda node: cls._get_pseudo_median_sort_val_naive(
                    ml_graph=ml_graph,
                    node=node,
                    neighbors=node_to_neighbors[node],
                    prev_layer_indices=_prev_layer_indices,
                    real_node_median_median=real_node_median_median,
                ),
            )

        node_to_in_neighbors = ml_graph.nodes_to_in_edges
        node_to_out_neighbors = ml_graph.nodes_to_out_edges
        for _ in range(max_iterations):
            for layer_idx in range(1, ml_graph.layer_count):
                prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer_idx - 1)
                _sort_layer(layer_idx, prev_layer_indices, node_to_in_neighbors)
            if only_one_up_iteration:
                return
            for layer_idx in range(ml_graph.layer_count - 2, -1, -1):
                prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer_idx + 1)
                _sort_layer(layer_idx, prev_layer_indices, node_to_out_neighbors)

    @classmethod
    def get_real_node_median_median(
        cls,
        ml_graph: MultiLayeredGraph,
        nodes_at_layer: list[MLGNode],
        node_to_neighbors: dict[MLGNode, set[MLGNode]],
        prev_layer_indices: dict[MLGNode, int],
    ) -> float:
        """Calculate median of medians of real nodes at given layer.

        Args:
            ml_graph (MultiLayeredGraph): _description_
            nodes_at_layer (list[MLGNode]): Nodes of a layer for which to calculate.
            node_to_neighbors (dict[MLGNode, set[MLGNode]]): Mapping from nodes to their neighbors.
            prev_layer_indices (dict[MLGNode, int]): Mapping from nodes at "previous" layer to position in layer.

        Returns:
            int: Median of medians of real nodes at given layer.
        """
        real_node_medians = (
            get_median(
                ml_graph=ml_graph,
                node=node,
                neighbors=node_to_neighbors[node],
                prev_layer_indices=prev_layer_indices,
            )
            for node in nodes_at_layer
            if not node.is_virtual
        )

        # should have at least one real node in layer, so list should not be empty
        return statistics.median(real_node_medians)

    @classmethod
    def _get_pseudo_median_sort_val_naive(
        cls,
        ml_graph: MultiLayeredGraph,
        node: MLGNode,
        neighbors: set[MLGNode],
        prev_layer_indices: dict[MLGNode, int],
        real_node_median_median: float,
    ) -> float:
        """Calculate an arbitrary value used to sort node of a layer, using median heuristic.

        Args:
            ml_graph (list[MLGNode]): Layer with nodes before sorting.
            node (MLGNode): Node for which to calculate sort value.
            neighbors (set[MLGNode]): Neighbors of node.
            prev_layer_indices (dict[MLGNode, int]): Mapping from nodes at "previous" layer to position within that layer.
            real_node_median_median (float): Median of real node medians.

        Returns:
            float: Sort value for node within layer.
        """

        median = get_median(ml_graph, node, neighbors, prev_layer_indices)

        if node.is_virtual:
            if median > real_node_median_median:
                median += PSEUDO_SORT_DISPLACE_VALUE
            else:
                median -= PSEUDO_SORT_DISPLACE_VALUE
        return median


def get_median(
    ml_graph: MultiLayeredGraph,
    node: MLGNode,
    neighbors: set[MLGNode],
    prev_layer_indices: dict[MLGNode, int],
) -> float:
    """Calculates "true" median of a node by using its neighbors positions.

    Args:
        layer_before_sorting (list[MLGNode]): Graph which this node belongs to.
        node (MLGNode): Node for which to calculate median
        neighbors (set[MLGNode]): Set of neighbors.
        prev_layer_indices (dict[MLGNode, int]): Dictionary mapping nodes of previous layer to indices.

    Returns:
        float: Median value of the given node.
    """
    if len(neighbors) == 0:
        return ml_graph.layers_to_nodes[node.layer].index(node)

    median = statistics.median(prev_layer_indices[node] for node in neighbors)
    return median
