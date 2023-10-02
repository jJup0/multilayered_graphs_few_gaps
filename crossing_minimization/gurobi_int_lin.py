import collections
import logging
import warnings

import gurobipy as gp
from gurobipy import GRB

from crossing_minimization.k_gaps import virtual_node_to_neighbor_position_sorting_func
from crossing_minimization.utils import (
    Above_or_below_T,
    GraphSorter,
    generate_layers_to_above_or_below,
)
from crossings.calculate_crossings import crossings_uv_vu
from multilayered_graph.multilayered_graph import (
    MLGNode,
    MLGNodeEdge_T,
    MultiLayeredGraph,
)

logger = logging.getLogger("Gurobi")
logger.setLevel(logging.WARNING)

gp.setParam("LogToConsole", 0)


class GurobiSorter(GraphSorter):
    algorithm_name = "Gurobi"

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
        # two sided only implemented for side gaps
        if (
            side_gaps_only is True
            and only_one_up_iteration is False
            and ml_graph.layer_count == 2
        ):
            # if two sided two layer
            side_gaps_gurobi_two_sided(ml_graph)
            return

        # else use general sorter
        one_sided(
            ml_graph,
            max_iterations=max_iterations,
            only_one_up_iteration=only_one_up_iteration,
            side_gaps_only=side_gaps_only,
            max_gaps=max_gaps,
        )


def side_gaps_gurobi_two_sided(
    ml_graph: MultiLayeredGraph,
) -> None:
    if ml_graph.layer_count != 2:
        raise ValueError(f"ml_graph.layer_count must be 2, got {ml_graph.layer_count}")

    m = gp.Model("Multilayered Graph 2 layer cross minimization")
    l1 = ml_graph.layers_to_nodes[0]
    l2 = ml_graph.layers_to_nodes[1]
    l1_ordering_gb_vars, _, _ = _gen_order_var_and_constraints(m, l1, prefix="l1_")
    l2_ordering_gb_vars, _, _ = _gen_order_var_and_constraints(m, l2, prefix="l2_")

    for nodes_in_layer, ordering_vars_for_layer in (
        (l1, l1_ordering_gb_vars),
        (l2, l2_ordering_gb_vars),
    ):
        real_nodes_layer = [n for n in nodes_in_layer if not n.is_virtual]
        virtual_nodes_layer = [n for n in nodes_in_layer if n.is_virtual]
        _gen_virtual_node_sidegap_vars_and_constraints(
            m,
            virtual_nodes=virtual_nodes_layer,
            real_nodes=real_nodes_layer,
            ordering_gb_vars=ordering_vars_for_layer,
        )

    # create objective function
    obj = gp.LinExpr()
    edges = ml_graph.layers_to_edges[0]
    for e1 in edges:
        n1_l1, n1_l2 = e1
        for e2 in edges:
            n2_l1, n2_l2 = e2
            if n1_l1 is n2_l1 or n1_l2 is n2_l2:
                continue

            ordering_var_for_l1_nodes = l1_ordering_gb_vars[n1_l1, n2_l1]
            ordering_var_for_l2_nodes = l2_ordering_gb_vars[n1_l2, n2_l2]

            crossing_between_nodes = m.addVar(
                vtype=GRB.BINARY, name=f"c_{n1_l1}{n2_l1}{n1_l2}{n2_l2}"
            )

            m.addConstr(
                crossing_between_nodes
                >= ordering_var_for_l1_nodes - ordering_var_for_l2_nodes,
                f"cross_{n1_l1}{n2_l1}{n1_l2}{n2_l2}",
            )
            m.addConstr(
                crossing_between_nodes
                >= ordering_var_for_l2_nodes - ordering_var_for_l1_nodes,
                f"cross_{n1_l2}{n2_l2}{n1_l1}{n2_l1}",
            )

            obj += crossing_between_nodes

    # set objective, update and optimize
    _optimize_model(m, obj)

    # order graph using gurobi variables
    _gurobi_merge_sort(l1, l1_ordering_gb_vars)
    _gurobi_merge_sort(l2, l2_ordering_gb_vars)


def one_sided(
    ml_graph: MultiLayeredGraph,
    *,
    max_iterations: int,
    only_one_up_iteration: bool,
    side_gaps_only: bool,
    max_gaps: int,
) -> None:
    layers_to_above_below = generate_layers_to_above_or_below(
        ml_graph, max_iterations, only_one_up_iteration
    )
    layer_to_model: dict[int, gp.Model] = {}
    layer_to_ordering_vars: dict[int, dict[tuple[MLGNode, MLGNode], gp.Var]] = {}

    gap_constraint_vars = real_node_neighbor_vars = None
    for layer_idx, above_below in layers_to_above_below:
        nodes = ml_graph.layers_to_nodes[layer_idx]

        if layer_idx not in layer_to_model:
            m = gp.Model(
                f"Multilayered graph crossing minimization - layer {layer_idx}"
            )
            m.Params.LogToConsole = 0

            ordering_gp_vars, _, _ = _gen_order_var_and_constraints(m, nodes)
            layer_to_ordering_vars[layer_idx] = ordering_gp_vars

            real_nodes = [n for n in nodes if not n.is_virtual]
            virtual_nodes = [n for n in nodes if n.is_virtual]
            if side_gaps_only:
                _gen_virtual_node_sidegap_vars_and_constraints(
                    m,
                    virtual_nodes=virtual_nodes,
                    real_nodes=real_nodes,
                    ordering_gb_vars=ordering_gp_vars,
                )
            else:
                gap_constraint_vars, real_node_neighbor_vars = _gen_k_gap_constraints(
                    m, nodes, ordering_gp_vars, max_gaps
                )
            _gen_fixed_constraints_for_virtual_order(
                ml_graph, m, ordering_gp_vars, above_below, layer_idx
            )
            layer_to_model[layer_idx] = m

        m = layer_to_model[layer_idx]
        ordering_gp_vars = layer_to_ordering_vars[layer_idx]
        # always update objective function
        obj = gp.LinExpr()
        n1__n2_crossings: int
        n2__n1_crossings: int

        for i, n1 in enumerate(nodes):
            for j in range(i):
                n2 = nodes[j]
                # todo could check if both node are virtual, no need to add
                # to objective function if forced by constraints
                n1__n2_crossings, n2__n1_crossings = crossings_uv_vu(
                    ml_graph, n1, n2, above_below
                )
                n1__n2 = ordering_gp_vars[(n1, n2)]
                n2__n1 = ordering_gp_vars[(n2, n1)]
                obj += n1__n2_crossings * n1__n2 + n2__n1_crossings * n2__n1

        # set objective, update and optimize
        _optimize_model(m, obj)

        _gurobi_log_result_info(
            ml_graph,
            ordering_gp_vars=ordering_gp_vars,
            gap_constraint_vars=gap_constraint_vars,
            real_node_neighbor_vars=real_node_neighbor_vars,
        )

        # order graph using gurobi variables
        _gurobi_merge_sort(nodes, ordering_gp_vars)


def _optimize_model(m: gp.Model, objective: "gp.Objective"):
    # set objective, update and optimize
    m.setObjective(objective, GRB.MINIMIZE)
    m.update()

    time_limit_seconds = 60 * 2
    m.setParam(GRB.Param.TimeLimit, time_limit_seconds)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        pass
    elif m.Status == GRB.TIME_LIMIT:
        warnings.warn("Hit time limit while solving model")
    else:
        warnings.warn(f"ENCOUNTERED ISSUE WHILE SOLVING MODEL: {m.Status}")


def _gurobi_log_result_info(
    ml_graph: MultiLayeredGraph,
    *,
    ordering_gp_vars: dict[MLGNodeEdge_T, gp.Var],
    gap_constraint_vars: list[gp.Var] | None,
    real_node_neighbor_vars: list[gp.Var] | None,
):
    if logger.getEffectiveLevel() > logging.INFO:
        return

    if gap_constraint_vars is not None and real_node_neighbor_vars is not None:
        total_gaps = 0
        for var in gap_constraint_vars:
            total_gaps += var.X
        logger.info(f"{len(gap_constraint_vars)} gap vars. Sum/{total_gaps=}")
        right_gap_var = gap_constraint_vars[-1]
        left_gap_var = gap_constraint_vars[-2]
        logger.info(
            f"{right_gap_var.VarName} = {right_gap_var.X} || {left_gap_var.VarName} = {left_gap_var.X}"
        )

        total_neighbors = 0
        for var in real_node_neighbor_vars:
            total_neighbors += var.X
        logger.info(
            f"{len(real_node_neighbor_vars)} neighbor vars. Sum/{total_neighbors=}"
        )

    # virtual_nodes = [n for n in ml_graph.layers_to_nodes[1] if n.is_virtual]
    real_nodes = [n for n in ml_graph.layers_to_nodes[1] if not n.is_virtual]

    rnodes_left_of_rnode_vars: dict[MLGNode, float] = {
        rnode_right: sum(
            ordering_gp_vars[rnode_left, rnode_right].X
            for rnode_left in real_nodes
            if rnode_left is not rnode_right
        )
        for rnode_right in real_nodes
    }
    logger.info(f"{sorted(rnodes_left_of_rnode_vars.items(), key=lambda x: x[1])}")


def _gen_order_var_and_constraints(m: gp.Model, nodes: list[MLGNode], prefix: str = ""):
    ordering_gb_vars: dict[tuple[MLGNode, MLGNode], gp.Var] = {}
    ordering_constraints_mutex: dict[tuple[MLGNode, MLGNode], gp.Constr] = {}
    for i, n1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            n2 = nodes[j]

            pair_n1_n2 = (n1, n2)
            pair_n2_n1 = (n2, n1)
            n1_before_n2 = m.addVar(vtype=GRB.BINARY, name=f"{prefix}{n1}->{n2}")  # type: ignore # incorrect call arguments
            ordering_gb_vars[pair_n1_n2] = n1_before_n2

            n2_before_n1 = m.addVar(vtype=GRB.BINARY, name=f"{prefix}{n2}->{n1}")  # type: ignore # incorrect call arguments
            ordering_gb_vars[pair_n2_n1] = n2_before_n1

            one_before_other_constraint = m.addConstr(
                n1_before_n2 + n2_before_n1 == 1, f"{prefix}{n1}<->{n2}"
            )
            ordering_constraints_mutex[pair_n1_n2] = ordering_constraints_mutex[
                pair_n2_n1
            ] = one_before_other_constraint

    ordering_constraints_transitivity: dict[
        tuple[MLGNode, MLGNode, MLGNode], gp.Constr
    ] = {}

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

                ordering_constraints_transitivity[(n1, n2, n3)] = m.addConstr(
                    n1__n2 + n2__n3 - n1__n3 <= 1, f"{n1}->{n2}->{n3}"
                )

    return (
        ordering_gb_vars,
        ordering_constraints_mutex,
        ordering_constraints_transitivity,
    )


def _gen_virtual_node_sidegap_vars_and_constraints(
    m: gp.Model,
    virtual_nodes: list[MLGNode],
    real_nodes: list[MLGNode],
    ordering_gb_vars: dict[tuple[MLGNode, MLGNode], gp.Var],
    prefix: str = "",
) -> None:
    # virtual_nodes are either left of all real nodes, or on the right
    virtual_node_gb_vars: dict[MLGNode, gp.Var] = {}
    for v_node in virtual_nodes:
        virtual_node_right = m.addVar(vtype=GRB.BINARY, name=f"{prefix}>>{v_node}")  # type: ignore # incorrect call arguments
        virtual_node_gb_vars[v_node] = virtual_node_right

        n1_right_of_reals = [ordering_gb_vars[n2, v_node] for n2 in real_nodes]
        m.addConstr(
            sum(n1_right_of_reals) - len(real_nodes) * virtual_node_right == 0,
            f"{prefix}<-<-{v_node}x",
        )


def _gen_k_gap_constraints(
    m: gp.Model,
    nodes: list[MLGNode],
    ordering_gp_vars: dict[MLGNodeEdge_T, gp.Var],
    allowed_gaps: int,
    prefix: str = "Gap_",
) -> tuple[list[gp.Var], list[gp.Var]]:
    virtual_nodes = [n for n in nodes if n.is_virtual]
    if not virtual_nodes:
        return [], []
    real_nodes = [n for n in nodes if not n.is_virtual]

    vnodes_left_of_rnode_vars: dict[MLGNode, list[gp.Var]] = {
        rnode: [ordering_gp_vars[vnode, rnode] for vnode in virtual_nodes]
        for rnode in real_nodes
    }
    rnodes_left_of_rnode_vars: dict[MLGNode, list[gp.Var]] = {
        n1: [ordering_gp_vars[n2, n1] for n2 in real_nodes if n2 is not n1]
        for n1 in real_nodes
    }
    gap_constraint_vars: list[gp.Var] = []
    real_node_neighbor_vars: list[gp.Var] = []

    # middle gaps
    for n1 in real_nodes:
        for n2 in real_nodes:
            if n1 is n2:
                continue

            n1_n2_realnode_neighbors = m.addVar(vtype=GRB.BINARY, name=f"{prefix}{n1}{n2}_rnode_neighbors")  # type: ignore # incorrect call arguments
            real_node_neighbor_vars.append(n1_n2_realnode_neighbors)
            # fmt: off
            m.addConstr(
                n1_n2_realnode_neighbors
                #
                # n1 not left neighbor of n2, if n1 comes after n2
                >= ordering_gp_vars[n1, n2] * len(nodes) - (len(nodes))  
                #
                # if n2 is left neighbor of n1, then this sum will only be 1 if n2 is the next real node after n1
                # otherwise it will be 0 or negative
                + sum(rnodes_left_of_rnode_vars[n1]) - sum(rnodes_left_of_rnode_vars[n2]) + 2 
            )
            # fmt: on

            n1_n2_gap = m.addVar(vtype=GRB.BINARY, name=f"{prefix}{n1}__{n2}")  # type: ignore # incorrect call arguments
            gap_constraint_vars.append(n1_n2_gap)

            # a gap between n1 and n2 can only exist if n1 comes before n2
            # not required, maybe improve performance?
            m.addConstr(n1_n2_gap <= ordering_gp_vars[n1, n2])

            # gap between two nodes exists if there are virtual nodes between them
            # but no real nodes
            # fmt: off
            m.addConstr(
                n1_n2_gap
                #
                # will be some small fraction greater than 0 if there are virtual nodes between n1 and n2
                >= (sum(vnodes_left_of_rnode_vars[n2]) - sum(vnodes_left_of_rnode_vars[n1])) / len(virtual_nodes) 
                # 
                # if nodes are real node neighbors then a gap can exist between them, otherwise not
                + n1_n2_realnode_neighbors - 1, # 
                f"{prefix}{n1}<->{n2}",
            )
            # fmt: on

    # outer/side gaps
    left_gap = m.addVar(vtype=GRB.BINARY, name=f"{prefix}leftgap")  # type: ignore # incorrect call arguments
    right_gap = m.addVar(vtype=GRB.BINARY, name=f"{prefix}rightgap")  # type: ignore # incorrect call arguments
    gap_constraint_vars.append(left_gap)
    gap_constraint_vars.append(right_gap)

    # there is a left/right-most gap if any virtual node is the left/right-most node
    for vnode in virtual_nodes:
        left_of_rnode_constraints = [
            ordering_gp_vars[vnode, rnode] for rnode in real_nodes
        ]
        m.addConstr(
            left_gap >= sum(left_of_rnode_constraints) - len(real_nodes) + 1,
            f"{prefix}leftgap_caused_by_{vnode}",
        )
        right_of_rnode_constraints = [
            ordering_gp_vars[rnode, vnode] for rnode in real_nodes
        ]
        m.addConstr(
            right_gap >= sum(right_of_rnode_constraints) - len(real_nodes) + 1,
            f"{prefix}rightgap_caused_by_{vnode}",
        )

    sum_of_gap_constraints = sum(gap_constraint_vars)
    assert not isinstance(
        sum_of_gap_constraints, int
    ), f"{gap_constraint_vars=} should not be empty"
    m.addConstr(sum_of_gap_constraints <= allowed_gaps, f"gap_limit")
    return gap_constraint_vars, real_node_neighbor_vars


def _gurobi_merge_sort(
    arr: list[MLGNode], ordering_gb_vars: dict[tuple[MLGNode, MLGNode], gp.Var]
) -> list[MLGNode]:
    # built-in .sort() with functools.cmp_to_key does not work, so manually implemented merge sort
    def merge(_arr1: list[MLGNode], _arr2: list[MLGNode]) -> list[MLGNode]:
        nonlocal ordering_gb_vars
        a1 = collections.deque(_arr1)
        a2 = collections.deque(_arr2)
        res: list[MLGNode] = []

        while a1 and a2:
            # X gives the gurobi-variable's value after optimization. Since the
            # variable is boolean in our case it will either evaluate to 0 or 1,
            # so compare with any number in between, 0.5 was arbitrarily chosen
            if ordering_gb_vars[a1[0], a2[0]].X > 0.5:
                res.append(a1.popleft())
            else:
                res.append(a2.popleft())

        res.extend(a1)
        res.extend(a2)
        return res

    if len(arr) > 1:
        m = len(arr) // 2
        arr1 = _gurobi_merge_sort(arr[:m], ordering_gb_vars)
        arr2 = _gurobi_merge_sort(arr[m:], ordering_gb_vars)
        arr[:] = merge(arr1, arr2)
    return arr


def _gen_fixed_constraints_for_virtual_order(
    ml_graph: MultiLayeredGraph,
    m: gp.Model,
    ordering_gp_vars: dict[MLGNodeEdge_T, gp.Var],
    above_or_below: Above_or_below_T,
    layer_idx: int,
):
    """
    For a minimal one-sided crossing, the relative order of virtual nodes is fixed.
    Virtual nodes appear in the same order as their neighbors.
    """
    sorted_virtual_nodes = sorted(
        (n for n in ml_graph.layers_to_nodes[layer_idx] if n.is_virtual),
        key=lambda node: virtual_node_to_neighbor_position_sorting_func(
            ml_graph, node, above_or_below
        ),
    )
    for i, vnode1 in enumerate(sorted_virtual_nodes):
        for j in range(i + 1, len(sorted_virtual_nodes)):
            vnode2 = sorted_virtual_nodes[j]
            vn1_vn2_ordering_var = ordering_gp_vars[vnode1, vnode2]
            m.addConstr(vn1_vn2_ordering_var == 1)
            vn2_vn1_ordering_var = ordering_gp_vars[vnode2, vnode1]
            m.addConstr(vn2_vn1_ordering_var == 0)


# class GurobiFullIntegerSorter_stub(GraphSorter):
#     algorithm_name = "Gurobi_integer_variable"

#     @classmethod
#     def _sort_graph(
#         cls,
#         ml_graph: MultiLayeredGraph,
#         *,
#         max_iterations: int,
#         only_one_up_iteration: bool,
#         side_gaps_only: bool,
#         max_gaps: int,
#     ) -> None:
#         layers_to_above_below = generate_layers_to_above_or_below(
#             ml_graph, max_iterations, only_one_up_iteration
#         )
#         layer_to_model: dict[int, gp.Model] = {}
#         layer_to_ordering_vars: dict[int, dict[MLGNode, list[gp.Var]]] = {}

#         for layer_idx, above_below in layers_to_above_below:
#             nodes = ml_graph.layers_to_nodes[layer_idx]

#             if layer_idx not in layer_to_model:
#                 m = gp.Model(
#                     f"Multilayered graph crossing minimization - layer {layer_idx}"
#                 )
#                 m.Params.LogToConsole = 0

#                 (
#                     node_to_all_pos_vars,
#                     _pos_to_all_node_vars,
#                 ) = cls._gen_pos_vars_and_constraints(m, nodes)
#                 layer_to_ordering_vars[layer_idx] = node_to_all_pos_vars

#                 if side_gaps_only:
#                     cls._gen_virtual_node_vars_and_constraints(m, node_to_all_pos_vars)
#                 else:
#                     cls._gen_k_gap_constraints(m, nodes, node_to_all_pos_vars, max_gaps)
#                 layer_to_model[layer_idx] = m

#             m = layer_to_model[layer_idx]
#             ordering_gp_vars = layer_to_ordering_vars[layer_idx]
#             # always update objective function
#             obj = gp.LinExpr()
#             for n1 in nodes:
#                 for n2 in nodes:
#                     if n1 is n2:
#                         continue
#                     above_below = above_below  # just for type check to not complain about unused variable
#                     # # seems not worth implementing because need to make helper variables
#                     # # that model relative order between nodes, which is what the other
#                     # # gurobi model is entirely based on

#                     # get position of variable by summing up
#                     # pos = gp.Var(GRB.INTEGER sum(i * var[i, node] for i in range(len(nodes))))
#                     # crossings_n1_n2 = GRB.BOOLEAN()
#                     # m.addConstr(crossings_n1_n2 >= (pos_node1 - pos_node2) / sth_large? * get_crossings(n1, n2))
#                     # obj += (pos_node1 - pos_node2)

#                     raise NotImplementedError("Crossings objective not implemented")

#             # set objective, update and optimize
#             m.setObjective(obj, GRB.MINIMIZE)
#             m.update()
#             m.optimize()

#             if m.Status != GRB.OPTIMAL:
#                 raise Exception(f"Model is not optimal: {m.Status}")

#             # order graph using gurobi variables
#             cls._sort_nodes(nodes, ordering_gp_vars)

#     @classmethod
#     def _gen_pos_vars_and_constraints(
#         cls, m: gp.Model, nodes: list[MLGNode]
#     ) -> tuple[dict[MLGNode, list[gp.Var]], dict[int, list[gp.Var]]]:
#         node_count = len(nodes)
#         node_to_all_pos_vars: dict[MLGNode, list[gp.Var]] = {node: [] for node in nodes}
#         pos_to_all_node_vars: dict[int, list[gp.Var]] = {
#             i: [] for i in range(node_count)
#         }

#         for node in nodes:
#             node_to_all_pos_vars[node] = []
#             for position in range(node_count):
#                 node_pos_var = m.addVar(vtype=GRB.BINARY, name=f"{node}_at_pos_{position}")  # type: ignore # incorrect call arguments
#                 node_to_all_pos_vars[node].append(node_pos_var)
#                 pos_to_all_node_vars[position].append(node_pos_var)

#             m.addConstr(
#                 gp.quicksum(node_to_all_pos_vars[node]) == 1,
#                 f"{node}_at_exactly_one_position",
#             )

#         for position in range(node_count):
#             m.addConstr(
#                 gp.quicksum(pos_to_all_node_vars[position]) == 1,
#                 f"one_node_at_pos{position}",
#             )

#         return node_to_all_pos_vars, pos_to_all_node_vars

#     @classmethod
#     def _gen_virtual_node_vars_and_constraints(
#         cls,
#         m: gp.Model,
#         node_to_all_pos_vars: dict[MLGNode, list[gp.Var]],
#     ):
#         """Generates constraints that force gap nodes to the sides.

#         Implemented by forcing real nodes in the middle.
#         n := len(nodes)
#         Generates n variables and n^2 constraints.

#         Args:
#             m (gp.Model): Model for which to generate variables and constraints.
#             node_to_all_pos_vars (dict[MLGNode, list[gp.Var]]):
#               Mapping from nodes to their boolean position variables.
#         """
#         node_count = len(node_to_all_pos_vars)
#         real_nodes = [node for node in node_to_all_pos_vars if not node.is_virtual]
#         starting_pos_to_gp_var: dict[int, gp.Var] = {}
#         for real_nodes_starting_pos in range(node_count - len(real_nodes)):
#             curr_starting_pos_gp_var = m.addVar(vtype=GRB.BINARY, name=f"real_nodes_starting_pos_{real_nodes_starting_pos}")  # type: ignore # incorrect call arguments
#             starting_pos_to_gp_var[real_nodes_starting_pos] = curr_starting_pos_gp_var
#             for node in real_nodes:
#                 for node_pos, node_pos_var in enumerate(node_to_all_pos_vars[node]):
#                     m.addConstr(
#                         node_pos * node_pos_var
#                         >= real_nodes_starting_pos * curr_starting_pos_gp_var
#                     )
#                     m.addConstr(
#                         node_pos * node_pos_var
#                         <= (node_count - real_nodes_starting_pos)
#                         * curr_starting_pos_gp_var
#                         + (1 - curr_starting_pos_gp_var) * node_count * 2
#                     )

#         m.addConstr(gp.quicksum(starting_pos_to_gp_var.values()) == 1)

#     @classmethod
#     def _gen_k_gap_constraints(
#         cls,
#         m: gp.Model,
#         nodes: list[MLGNode],
#         node_to_all_pos_vars: dict[MLGNode, list[gp.Var]],
#         allowed_gaps: int,
#     ) -> ...:
#         raise NotImplementedError("Not implemented")

#     @classmethod
#     def _sort_nodes(
#         cls, nodes: list[MLGNode], node_to_all_pos_vars: dict[MLGNode, list[gp.Var]]
#     ):
#         for node in tuple(nodes):
#             for i, pos_var in enumerate(node_to_all_pos_vars[node]):
#                 if pos_var.X == 1:
#                     nodes[i] = node
#                     break
#         assert len(set(nodes)) == len(nodes)


# def _gen_virtual_node_vars_and_constraints_old(
#     m: gp.Model,
#     virtual_nodes: list[MLGNode],
#     real_nodes: list[MLGNode],
#     ordering_gb_vars: dict[tuple[MLGNode, MLGNode], gp.Var],
#     prefix: str = "",
# ):
#     # virtual_nodes are either left of all real nodes, or on the right
#     virtual_node_gb_vars: dict[
#         tuple[MLGNode, Literal["left"] | Literal["right"]], gp.Var
#     ] = {}
#     for v_node in virtual_nodes:
#         virtual_node_left = m.addVar(vtype=GRB.BINARY, name=f"{prefix}{v_node}...")  # type: ignore # incorrect call arguments
#         virtual_node_right = m.addVar(vtype=GRB.BINARY, name=f"{prefix}...{v_node}")  # type: ignore # incorrect call arguments
#         virtual_node_gb_vars[(v_node, "left")] = virtual_node_left
#         virtual_node_gb_vars[(v_node, "right")] = virtual_node_right
#         m.addConstr(
#             virtual_node_left + virtual_node_right == 1, f"{prefix}...{v_node}..."
#         )

#         n1_left_of_reals = [ordering_gb_vars[v_node, n2] for n2 in real_nodes]
#         m.addConstr(
#             sum(n1_left_of_reals) - len(real_nodes) * virtual_node_left == 0,
#             f"{prefix}x{v_node}->->",
#         )

#         n1_right_of_reals = [ordering_gb_vars[n2, v_node] for n2 in real_nodes]
#         m.addConstr(
#             sum(n1_right_of_reals) - len(real_nodes) * virtual_node_right == 0,
#             f"{prefix}<-<-{v_node}x",
#         )

#     return virtual_node_gb_vars
