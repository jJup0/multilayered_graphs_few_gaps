"""
Implemented algorithm presented in the following paper:
(WARNING: so far the implementation seems to be faulty, producing strange results)

@inproceedings{forster2005fast,
  title={A fast and simple heuristic for constrained two-level crossing reduction},
  author={Forster, Michael},
  booktitle={Graph Drawing: 12th International Symposium, GD 2004, New York,
  NY, USA, September 29-October 2, 2004, Revised Selected Papers 12},
  pages={206--216},
  year={2005},
  organization={Springer}
}
"""
import statistics
from collections import deque
from typing import Literal, TypeAlias

from crossing_minimization.barycenter_heuristic import unweighted_barycenter
from crossing_minimization.utils import (
    DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    deprecated_lgraph_sorting,
    get_layer_idx_above_or_below,
)
from multilayered_graph.multilayered_graph import MLGNode, MultiLayeredGraph

NodeConstraint_T: TypeAlias = tuple[MLGNode, MLGNode]


@deprecated_lgraph_sorting
def few_gaps_constrained_paper(
    ml_graph: MultiLayeredGraph,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS_MULTILAYERED_CROSSING_MINIMIZATION,
    only_one_up_iteration: bool = False,
):
    """Implementation does not seem to place nodes in limited gaps."""

    layers__above_below: list[tuple[int, Literal["above"] | Literal["below"]]] = []
    layers__above_below.extend(
        (layer_idx, "below") for layer_idx in range(1, ml_graph.layer_count)
    )
    # if not only_one_up_iteration:
    #     layers__above_below.extend(
    #         (layer_idx, "above")
    #         for layer_idx in range(ml_graph.layer_count - 2, -1, -1)
    #     )
    #     layers__above_below *= max_iterations

    for layer_idx, above_or_below in layers__above_below:
        constraints = _generate_constraints(ml_graph, layer_idx, above_or_below)
        _constrained_crossing_reduction(
            ml_graph, layer_idx, above_or_below, constraints
        )


def _generate_constraints(
    ml_graph: MultiLayeredGraph,
    layer_idx: int,
    above_or_below: Literal["above"] | Literal["below"],
) -> set[tuple[MLGNode, MLGNode]]:
    nodes = ml_graph.layers_to_nodes[layer_idx]

    prev_layer_idx = get_layer_idx_above_or_below(layer_idx, above_or_below)
    prev_layer_indices = ml_graph.nodes_to_indices_at_layer(prev_layer_idx)

    nodes_to_neighbors = (
        ml_graph.nodes_to_in_edges
        if above_or_below == "below"
        else ml_graph.nodes_to_out_edges
    )

    barycenters = {
        node: unweighted_barycenter(
            ml_graph, node, nodes_to_neighbors[node], prev_layer_indices
        )
        for node in nodes
    }

    real_nodes = [node for node in nodes if not node.is_virtual]
    real_node_barycenters = [barycenters[node] for node in real_nodes]
    median_real_bary = statistics.median(real_node_barycenters)

    virtual_nodes = [node for node in nodes if node.is_virtual]

    constraints: set[tuple[MLGNode, MLGNode]] = set()
    for v_node in virtual_nodes:
        # todo make this decision a parameter passed to the function
        if barycenters[v_node] < median_real_bary:
            constraints.update((v_node, r_node) for r_node in real_nodes)
        else:
            constraints.update((r_node, v_node) for r_node in real_nodes)

    print("constraints:", *sorted(constraints, key=lambda x: str(x[1])), sep="\n")
    return constraints


def _constrained_crossing_reduction(
    ml_graph: MultiLayeredGraph,
    layer_idx: int,
    above_or_below: Literal["above"] | Literal["below"],
    constraints: set[NodeConstraint_T],
):
    # variables starting with an '_' are helper variables not mentioned in the algorithm pseudo code
    _nodes_to_neighbors = (
        ml_graph.nodes_to_in_edges
        if above_or_below == "below"
        else ml_graph.nodes_to_out_edges
    )

    # V1 = ml_graph.layers_to_nodes[_prev_layer_idx]
    V2 = ml_graph.layers_to_nodes[layer_idx]

    # in pseudo code already defined in graph datastructure
    deg = {node: len(_nodes_to_neighbors[node]) for node in V2}

    _prev_layer_idx = get_layer_idx_above_or_below(layer_idx, above_or_below)
    _prev_layer_indices = ml_graph.nodes_to_indices_at_layer(_prev_layer_idx)
    b = {
        node: unweighted_barycenter(
            ml_graph, node, _nodes_to_neighbors[node], _prev_layer_indices
        )
        for node in V2
    }
    L = {node: [node] for node in V2}

    V: set[MLGNode] = set(c[0] for c in constraints).union(
        c[1] for c in constraints
    )  # constrained vertices
    V_dash = {node for node in V2 if node not in V}  # unconstrained vertices

    while True:
        _violated = _find_violated_constraint(V, constraints, _nodes_to_neighbors, b)
        if _violated is None:
            break
        s, t = _violated

        v_c = MLGNode(layer_idx, f"v_c__{s}-{t}", is_virtual=False)
        deg[v_c] = deg[s] + deg[t]
        b[v_c] = (b[s] * deg[s] + b[t] * deg[t]) / deg[v_c]
        L[v_c] = L[s] + L[t]

        # # old optimization
        # _constraint_pop_set: set[NodeConstraint_T] = set()
        # _constraint_add_set: set[NodeConstraint_T] = set()
        # for c in constraints:
        #     c_0, c_1 = c

        #     # can probably remove, as the only case where this is true should be (s, t) right?
        #     if c_0 is s or c_0 is t and c_1 is s or c_1 is t:
        #         _constraint_pop_set.add(c)
        #         # _constraint_add_set.add(v_c, v_c) # no need as removed later anyways
        #     elif c_0 is s or c_0 is t:
        #         _constraint_pop_set.add(c)
        #         _constraint_add_set.add((c_0, v_c))
        #     elif c_1 is s or c_1 is t:
        #         _constraint_pop_set.add(c)
        #         _constraint_add_set.add((v_c, c_1))
        # constraints.update(_constraint_add_set)
        # constraints.difference_update(_constraint_pop_set)
        # constraints.discard((v_c, v_c))
        # V.discard(s)
        # V.discard(t)
        # # if v_c has incident constraints
        # if _constraint_add_set:
        #     V.add(v_c)
        # else:
        #     V_dash.add(v_c)

        constraints = set(
            tuple(node if node is not s and node is not t else v_c for node in c)
            for c in constraints
        )
        constraints.discard((v_c, v_c))
        V.discard(s)
        V.discard(t)
        _v_c_is_constrained = False
        for n1, n2 in constraints:
            if n1 is v_c or n2 is v_c:
                V.add(v_c)
                _v_c_is_constrained = True
                break
        if not _v_c_is_constrained:
            V_dash.add(v_c)

    V_dash_dash = sorted(V.union(V_dash), key=lambda node: b[node])
    other_L: list[MLGNode] = []
    for node in V_dash_dash:
        other_L.extend(L[node])

    # print(ml_graph.layers_to_nodes[layer_idx])
    # print(other_L)
    ml_graph.layers_to_nodes[layer_idx][:] = other_L


def _find_violated_constraint(
    V: set[MLGNode],
    C: set[NodeConstraint_T],
    _nodes_to_incoming_edges: dict[MLGNode, set[MLGNode]],
    _nodes_to_barycenter: dict[MLGNode, float],
) -> tuple[MLGNode, MLGNode] | None:
    b = _nodes_to_barycenter

    S: set[MLGNode] = set()  # active vertices
    I: dict[MLGNode, deque[NodeConstraint_T]] = {}
    for v in V:
        I[v] = deque()
        if len(_nodes_to_incoming_edges[v]) == 0:
            S.add(v)

    while S:
        v = S.pop()
        for c in I[v]:
            s, v = c
            if b[s] > b[v]:
                return c
        for c in C:
            if c[0] is not v:
                continue
            v, t = c
            I[t].appendleft(c)
            if len(I[t]) == len(_nodes_to_incoming_edges[t]):
                S.add(t)

    return None


if __name__ == "__main__":
    pass
