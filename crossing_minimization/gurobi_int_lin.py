import collections
import copy
from functools import cmp_to_key

import gurobipy as gp
from gurobipy import GRB

from crossings.calculate_crossings import crossings_uv_vu
from multilayered_graph.multilayer_graph_generator import generate_multilayer_graph
from multilayered_graph.multilayered_graph import MultiLayeredGraph, MLGNode
from crossing_minimization.barycenter_heuristic import few_gaps_barycenter_smart_sort

gp.LogToConsole = 0


def few_gaps_gurobi(ml_graph: MultiLayeredGraph) -> int:
    layers_to_above_below = []
    layers_to_above_below.extend(
        (layer_idx, "below") for layer_idx in range(1, ml_graph.layer_count)
    )
    layers_to_above_below.extend(
        (layer_idx, "above") for layer_idx in range(ml_graph.layer_count - 2, -1, -1)
    )
    layers_to_above_below *= 3
    for layer_idx, above_below in layers_to_above_below:
        m = gp.Model("Multilayered graph crossing minimization")
        m.Params.LogToConsole = 0

        nodes = ml_graph.layers_to_nodes[layer_idx]

        ordering_gb_vars: dict[tuple[MLGNode, MLGNode], gp.Var] = {}
        # ordering of nodes
        for n1 in nodes:
            for n2 in nodes:
                if n1 is n2:
                    continue

                n1_before_n2 = m.addVar(vtype=GRB.BINARY, name=f"{n1}->{n2}")
                ordering_gb_vars[(n1, n2)] = n1_before_n2

                n2_before_n1 = m.addVar(vtype=GRB.BINARY, name=f"{n2}->{n1}")
                ordering_gb_vars[(n2, n1)] = n2_before_n1

                m.addConstr(n1_before_n2 + n2_before_n1 == 1, f"{n1}<->{n2}")

        # transitivity of order
        for n1 in nodes:
            for n2 in nodes:
                if n1 is n2:
                    continue
                for n3 in nodes:
                    if n3 is n1 or n3 is n2:
                        continue

                    n1__n2 = ordering_gb_vars[(n1, n2)]
                    n2__n3 = ordering_gb_vars[(n2, n3)]
                    n1__n3 = ordering_gb_vars[(n1, n3)]

                    # 1 2 3 -> 1, 1, 1
                    # 1 3 2 -> 1, 0, 1
                    # 2 1 3 -> 0, 1, 1
                    # 2 3 1 -> 0, 1, 0
                    # 3 1 2 -> 1, 0, 0
                    # 3 2 1 -> 0, 0, 0

                    m.addConstr(n1__n2 + n2__n3 - n1__n3 <= 1, f"{n1}->{n2}->{n3}")

        # virtual_nodes are either left of all real nodes, or on the right
        real_nodes = [n for n in nodes if not n.is_virtual]
        virtual_nodes = [n for n in nodes if n.is_virtual]
        virtual_node_gb_vars = {}
        for v_node in virtual_nodes:
            virtual_node_left = m.addVar(vtype=GRB.BINARY, name=f"{v_node}...")
            virtual_node_right = m.addVar(vtype=GRB.BINARY, name=f"...{v_node}")
            virtual_node_gb_vars[(v_node, "left")] = virtual_node_left
            virtual_node_gb_vars[(v_node, "right")] = virtual_node_right
            m.addConstr(virtual_node_left + virtual_node_right == 1, f"...{v_node}...")

            n1_left_of_reals = [ordering_gb_vars[v_node, n2] for n2 in real_nodes]
            m.addConstr(
                sum(n1_left_of_reals) - len(real_nodes) * virtual_node_left == 0,
                f"x{v_node}->->",
            )

            n1_right_of_reals = [ordering_gb_vars[n2, v_node] for n2 in real_nodes]
            m.addConstr(
                sum(n1_right_of_reals) - len(real_nodes) * virtual_node_right == 0,
                f"<-<-{v_node}x",
            )

        # crossing vars
        obj = gp.LinExpr()
        n1__n2_crossings: int
        n2__n1_crossings: int
        for n1 in nodes:
            for n2 in nodes:
                if n1 is n2:
                    continue
                n1__n2_crossings, n2__n1_crossings = crossings_uv_vu(
                    ml_graph, n1, n2, above_below
                )
                # print(f"{(n1__n2_crossings, n2__n1_crossings)=}")
                n1__n2 = ordering_gb_vars[(n1, n2)]
                n2__n1 = ordering_gb_vars[(n2, n1)]
                obj += n1__n2_crossings * n1__n2 + n2__n1_crossings * n2__n1

        # Set objective
        m.setObjective(obj, GRB.MINIMIZE)
        m.update()
        # print(f"{obj=}")

        m.optimize()
        # print(f"Objective val: {m.objVal}")

        gurobi_merge_sort(nodes, ordering_gb_vars)


def gurobi_merge_sort(arr, ordering_gb_vars):
    def merge(_arr1: list[gp.Var], _arr2: list[gp.Var]):
        nonlocal ordering_gb_vars
        a1 = collections.deque(_arr1)
        a2 = collections.deque(arr2)
        res = []

        while a1 and a2:
            if ordering_gb_vars[a1[0], a2[0]].x > 0.5:
                res.append(a1.popleft())
            else:
                res.append(a2.popleft())

        res.extend(a1)
        res.extend(a2)
        return res

    if len(arr) > 1:
        m = len(arr) // 2
        arr1 = gurobi_merge_sort(arr[:m], ordering_gb_vars)
        arr2 = gurobi_merge_sort(arr[m:], ordering_gb_vars)
        arr[:] = merge(arr1, arr2)
    return arr


def sort_nodes_with_gurobi_ordering_BROKEN(
    nodes: list[MLGNode], ordering_gb_vars: dict[tuple[MLGNode, MLGNode], gp.Var]
):
    def compare(n1: MLGNode, n2: MLGNode):
        if ordering_gb_vars[n1, n2].x < 0.5:
            return -1
        return 1

    nodes.sort(key=cmp_to_key(compare))


def _check_correctly_sorted(
    nodes: list[MLGNode], ordering_gb_vars: dict[tuple[MLGNode, MLGNode], gp.Var]
):
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i + 1 :]:
            if ordering_gb_vars[n1, n2].x < 0.9:
                print(f"ERROR {n1} comes before {n2}")
                exit(1)


def _check_transitive_ordering(
    nodes: list[MLGNode], ordering_gb_vars: dict[tuple[MLGNode, MLGNode], gp.Var]
):
    for n1 in nodes:
        for n2 in nodes:
            if n1 is n2:
                continue
            if ordering_gb_vars[n1, n2].x < 0.5:
                continue
            for n3 in nodes:
                if n3 is n1 or n3 is n2:
                    continue
                if ordering_gb_vars[n2, n3].x > 0.5:
                    assert ordering_gb_vars[n1, n3].x > 0.5


def test_gurobi_implementation():
    gurobi_graph = generate_multilayer_graph(5, 20, 0.1, 0.5, randomness_seed=1)
    bary_graph = copy.deepcopy(gurobi_graph)

    # gurobi_graph.to_pygraphviz_graph().draw("before.svg")

    res = few_gaps_gurobi(gurobi_graph)
    # print(f"GUROBI RES = {res}")
    print(f"{gurobi_graph.get_crossings_per_layer()=}")
    # gurobi_graph.to_pygraphviz_graph().draw("gurobi.svg")

    few_gaps_barycenter_smart_sort(bary_graph)
    print(f"{bary_graph.get_crossings_per_layer()=}")
    # bary_graph.to_pygraphviz_graph().draw("bary.svg")


if __name__ == "__main__":
    test_gurobi_implementation()

    # gurobi_graph.get_crossings_per_layer() = [22, 7, 55, 8, 0]
    # bary_graph.get_crossings_per_layer() = [30, 10, 37, 1, 0]

    # l = [random.randint(0, 9999) for _ in range(200)]
    # ls = sorted(l)
    # mergeSort(l, None)
    # print(l == ls)
