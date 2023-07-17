import statistics
from typing import Callable

from crossing_minimization.k_gaps import k_gaps_sort_layer
from crossing_minimization.utils import (
    DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    Above_or_below_T,
    GraphSorter,
    deprecated_lgraph_sorting,
    generate_layers_to_above_or_below,
    get_graph_neighbors_from_above_or_below,
    get_layer_idx_above_or_below,
)
from crossings.calculate_crossings import crossings_uv_vu
from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

PSEUDO_SORT_DISPLACE_VALUE = 1_000


class AbstractBarycenterSorted(GraphSorter):
    @classmethod
    def k_gaps_barycenter(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int,
        only_one_up_iteration: bool,
        max_gaps: int,
    ):
        layer_to_unordered_real_nodes = [
            [n for n in ml_graph.layers_to_nodes[layer_idx] if not n.is_virtual]
            for layer_idx in range(ml_graph.layer_count)
        ]
        layers_to_above_below = generate_layers_to_above_or_below(
            ml_graph, max_iterations, only_one_up_iteration
        )

        for layer_idx, above_or_below in layers_to_above_below:
            nodes_to_neighbors = get_graph_neighbors_from_above_or_below(
                ml_graph, above_or_below
            )
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(
                get_layer_idx_above_or_below(layer_idx, above_or_below)
            )
            layer_to_unordered_real_nodes[layer_idx].sort(
                key=lambda node: get_actual_barycenter(
                    ml_graph,
                    node,
                    nodes_to_neighbors[node],
                    prev_layer_indices,
                )
            )
            k_gaps_sort_layer(
                ml_graph,
                layers_to_ordered_real_nodes=layer_to_unordered_real_nodes,
                layer_idx=layer_idx,
                above_or_below="below",
                gaps=max_gaps,
            )


class BarycenterImprovedSorter(AbstractBarycenterSorted):
    algorithm_name = "Barycenter improved"

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
            return cls._side_gaps(
                ml_graph,
                max_iterations=max_iterations,
                only_one_up_iteration=only_one_up_iteration,
            )

    @classmethod
    def _side_gaps(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int,
        only_one_up_iteration: bool,
    ) -> None:
        """
        Sorts MultiLayeredGraph using barycenter heuristic, placing nodes
        depending on minimal crossings.
        """

        def _sort_layer_barycenter_improved(
            layer_idx: int,
            above_or_below: Above_or_below_T,
        ):
            nonlocal ml_graph, layer_to_real_nodes

            node_to_neighbors = get_graph_neighbors_from_above_or_below(
                ml_graph, above_or_below
            )
            neighbor_layer_idx = get_layer_idx_above_or_below(layer_idx, above_or_below)
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(neighbor_layer_idx)

            ml_graph.layers_to_nodes[layer_idx] = sorted(
                ml_graph.layers_to_nodes[layer_idx],
                key=lambda node: _get_pseudo_barycenter_improved_placement(
                    node,
                    node_to_neighbors[node],
                    prev_layer_indices,
                    ml_graph,
                    layer_to_real_nodes[layer_idx],
                    above_or_below,
                ),
            )

        layer_to_real_nodes: dict[int, list[MLGNode]] = {}
        for layer_idx, nodes in ml_graph.layers_to_nodes.items():
            real_nodes = [n for n in nodes if not n.is_virtual]
            layer_to_real_nodes[layer_idx] = real_nodes

        for layer_idx, above_or_below in generate_layers_to_above_or_below(
            ml_graph, max_iterations, only_one_up_iteration
        ):
            _sort_layer_barycenter_improved(layer_idx, above_or_below)


class BarycenterNaiveSorter(AbstractBarycenterSorted):
    algorithm_name = "Barycenter naive"

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
            return cls._side_gaps(
                ml_graph,
                max_iterations=max_iterations,
                only_one_up_iteration=only_one_up_iteration,
            )
        return cls.k_gaps_barycenter(
            ml_graph,
            max_iterations=max_iterations,
            only_one_up_iteration=only_one_up_iteration,
            max_gaps=max_gaps,
        )

    @classmethod
    def _side_gaps(
        cls,
        ml_graph: MultiLayeredGraph,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
        only_one_up_iteration: bool = False,
    ) -> None:
        """Sorts nodes in multilayered graph according to barycenter heuristic.

        Modified in place.
        Args:
            ml_graph:
            Graph on which to apply sorting.
            max_iterations:
            Amount of "up" and "down" cycles to make for sorting. Defaults to 3.
            only_one_up_iteration:
            Whether to only do one sided crossing minimization or not.
            Defaults to False.
        """

        def _get_real_node_barycenter_median(
            nodes_at_layer: list[MLGNode],
            _node_to_neighbors: dict[MLGNode, set[MLGNode]],
            _prev_layer_indices: dict[MLGNode, int],
        ):
            nonlocal ml_graph
            real_node_barycenters = (
                get_actual_barycenter(
                    ml_graph, node, _node_to_neighbors[node], _prev_layer_indices
                )
                for node in nodes_at_layer
                if not node.is_virtual
            )

            # should have at least one real node in layer, so list should not be empty
            return statistics.median(real_node_barycenters)

        def _sort_layer(
            _ml_graph: MultiLayeredGraph,
            _layer_idx: int,
            _prev_layer_indices: dict[MLGNode, int],
            _node_to_neighbors: dict[MLGNode, set[MLGNode]],
            # above_or_below: Above_or_below_T,
        ):
            real_node_bary_median = _get_real_node_barycenter_median(
                _ml_graph.layers_to_nodes[_layer_idx],
                _node_to_neighbors,
                _prev_layer_indices,
            )
            layer_before_sorting = _ml_graph.layers_to_nodes[_layer_idx][:]
            _ml_graph.layers_to_nodes[_layer_idx].sort(
                key=lambda node: _get_pseudo_barycenter_naive_virtual_placement(
                    layer_before_sorting,
                    node,
                    _node_to_neighbors[node],
                    _prev_layer_indices,
                    real_node_bary_median,
                )
            )

        nodes_to_in_neighbors = ml_graph.nodes_to_in_edges
        nodes_to_out_neighbors = ml_graph.nodes_to_out_edges

        for _ in range(max_iterations):
            for layer in range(1, ml_graph.layer_count):
                prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer - 1)
                _sort_layer(ml_graph, layer, prev_layer_indices, nodes_to_in_neighbors)
            if only_one_up_iteration:
                return

            for layer in range(ml_graph.layer_count - 2, -1, -1):
                prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer + 1)
                _sort_layer(ml_graph, layer, prev_layer_indices, nodes_to_out_neighbors)


def get_actual_barycenter(
    ml_graph: MultiLayeredGraph,
    node: MLGNode,
    neighbors: set[MLGNode],
    prev_layer_indices: dict[MLGNode, int],
) -> float:
    neighbor_count = len(neighbors)
    if neighbor_count == 0:
        # TODO check if this is a viable strategy
        return ml_graph.layers_to_nodes[node.layer].index(node)
    barycenter = sum(prev_layer_indices[node] for node in neighbors) / neighbor_count
    node.text_info = f"bary {barycenter:.5}"
    return barycenter


def _get_pseudo_barycenter_naive_virtual_placement(
    layer_before_sorting: list[MLGNode],
    node: MLGNode,
    neighbors: set[MLGNode],
    prev_layer_indices: dict[MLGNode, int],
    real_node_bary_median: float,
) -> float:
    # TODO get barycenter as accurate fraction to avoid
    #   floating point errors, and ensure stable sorting

    neighbor_count = len(neighbors)
    if neighbor_count == 0:
        # TODO check if this is a viable strategy
        return layer_before_sorting.index(node)
    barycenter = sum(prev_layer_indices[node] for node in neighbors) / neighbor_count

    if node.is_virtual:
        if barycenter > real_node_bary_median:
            barycenter += PSEUDO_SORT_DISPLACE_VALUE
        else:
            barycenter -= PSEUDO_SORT_DISPLACE_VALUE
    node.text_info = f"bary {barycenter:.5}"
    return barycenter


def _get_pseudo_barycenter_improved_placement(
    node: MLGNode,
    neighbors: set[MLGNode],
    prev_layer_indices: dict[MLGNode, int],
    ml_graph: MultiLayeredGraph,
    real_nodes_at_layer: list[MLGNode],
    above_or_below: Above_or_below_T,
) -> float:
    # TODO get barycenter as accurate fraction to avoid
    #   floating point errors, and ensure stable sorting

    neighbor_count = len(neighbors)
    if neighbor_count == 0:
        # TODO check if this is a viable strategy
        return ml_graph.layers_to_nodes[node.layer].index(node)
    barycenter = sum(prev_layer_indices[node] for node in neighbors) / neighbor_count

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
            barycenter -= PSEUDO_SORT_DISPLACE_VALUE
        else:
            barycenter += PSEUDO_SORT_DISPLACE_VALUE

    node.text_info = f"bary {barycenter:.5}"
    return barycenter


@deprecated_lgraph_sorting
def few_gaps_barycenter_split(_ml_graph: MultiLayeredGraph) -> None:
    """Deprecated, produces the same result as few_gaps_barycenter_smart_sort()."""

    def _sort_layer_barycenter_split(
        ml_graph: MultiLayeredGraph,
        layer_to_real_nodes: dict[int, list[MLGNode]],
        layer_idx: int,
        prev_layer_indices: dict[MLGNode, int],
        node_to_neighbors: dict[MLGNode, set[MLGNode]],
        above_or_below: Above_or_below_T,
    ):
        curr_layer_nodes: list[MLGNode] = ml_graph.layers_to_nodes[layer_idx]
        barycenters = {
            node: get_actual_barycenter(
                ml_graph, node, node_to_neighbors[node], prev_layer_indices
            )
            for node in curr_layer_nodes
        }
        curr_layer_nodes.sort(key=lambda n: barycenters[n])
        # virtual nodes in regular barycenter order
        virtual_nodes = [n for n in curr_layer_nodes if n.is_virtual]
        real_nodes = layer_to_real_nodes[layer_idx]

        place_left = {n: 0 for n in virtual_nodes}
        place_right = {n: 0 for n in virtual_nodes}
        for v_node in virtual_nodes:
            for r_node in real_nodes:
                left_crossings, right_crossings = crossings_uv_vu(
                    ml_graph, v_node, r_node, above_or_below
                )
                place_left[v_node] += left_crossings
                place_right[v_node] += right_crossings

        # start with all virtual nodes on right side. Shift over one by one to left and check min crossings.
        min_idx = 0
        min_crossings = crossings = sum(c for c in place_right.values())
        for i, v_node in enumerate(virtual_nodes, start=1):
            crossings += place_left[v_node] - place_right[v_node]
            if crossings < min_crossings:
                min_idx = i
                min_crossings = crossings

        for i in range(min_idx):
            barycenters[virtual_nodes[i]] -= PSEUDO_SORT_DISPLACE_VALUE
        for i in range(min_idx, len(virtual_nodes)):
            barycenters[virtual_nodes[i]] += PSEUDO_SORT_DISPLACE_VALUE

        # resort with virtual nodes in gaps
        curr_layer_nodes.sort(key=lambda n: barycenters[n])

    _few_gaps_barycenter_base(_ml_graph, _sort_layer_barycenter_split)


@deprecated_lgraph_sorting
def _few_gaps_barycenter_base(
    ml_graph: MultiLayeredGraph,
    sorting_alg: Callable[
        [
            MultiLayeredGraph,
            dict[int, list[MLGNode]],
            int,
            dict[MLGNode, int],
            dict[MLGNode, set[MLGNode]],
            Above_or_below_T,
        ],
        None,
    ],
) -> None:
    """Sorts nodes in multilayered graph according to barycenter heuristic.

    Modified in place. Virtual nodes are place left or right depending on crossings with real nodes.
    Args:
        ml_graph: Graph on which to apply sorting.
    """

    layer_to_real_nodes = {}
    for layer_idx, nodes in ml_graph.layers_to_nodes.items():
        real_nodes = [n for n in nodes if not n.is_virtual]
        layer_to_real_nodes[layer_idx] = real_nodes

    node_to_in_neighbors = ml_graph.nodes_to_in_edges
    node_to_out_neighbors = ml_graph.nodes_to_out_edges
    # arbitrary loop count,TODO pass as parameter?
    for _ in range(3):
        for layer_idx in range(1, ml_graph.layer_count):
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer_idx - 1)
            sorting_alg(
                ml_graph,
                layer_to_real_nodes,
                layer_idx,
                prev_layer_indices,
                node_to_in_neighbors,
                "below",
            )
        for layer_idx in range(ml_graph.layer_count - 2, -1, -1):
            prev_layer_indices = ml_graph.nodes_to_indices_at_layer(layer_idx + 1)
            sorting_alg(
                ml_graph,
                layer_to_real_nodes,
                layer_idx,
                prev_layer_indices,
                node_to_out_neighbors,
                "above",
            )
