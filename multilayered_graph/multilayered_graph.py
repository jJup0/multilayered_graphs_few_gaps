from collections import defaultdict
from copy import deepcopy
from typing import Any, TypeAlias

import networkx as nx
import pygraphviz as pgv  # type: ignore # stubfile not found


class InvalidEdgeParamError(ValueError):
    """Can not add edge between two nodes.

    Raised when an edge between a node higher up in the graph and lower node is expected to be created.
    """

    pass


class InvalidLayerError(ValueError):
    """Out-of-bounds layer is passed."""

    pass


class NoNodeExistsError(ValueError):
    """Node to perform operation on does not exist in graph."""

    pass


class MLGNode:
    def __init__(self, layer: int, name: str, is_virtual: bool = False):
        # layer is permanent, TODO refactor: maybe only store layer in graph
        self.layer = layer
        self.name = name
        self.is_virtual = is_virtual
        self.text_info = ""

    def __copy__(self) -> "MLGNode":
        _copy = MLGNode(self.layer, self.name, self.is_virtual)
        _copy.text_info = self.text_info
        return _copy

    def __deepcopy__(self, memo: None = None):
        return self.__copy__()

    def attrs_to_dict(self) -> dict[str, Any]:
        return {"layer": self, "name": self.name, "is_virtual": self.is_virtual}

    def __hash__(self) -> int:
        return hash((self.layer, self.name))

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        # if self.is_virtual:
        #     return ""
        return self.name


MLGNodeEdge_T: TypeAlias = tuple[MLGNode, MLGNode]


class MultiLayeredGraph:
    def __init__(self, layer_count: int = 1):
        if layer_count < 0:
            raise ValueError("Layer count must be larger than 0.")

        self.layer_count = layer_count

        # Maps list of nodes to layer. The order of the nodes in this list is
        # the same order as they are when drawn
        self.layers_to_nodes: defaultdict[int, list[MLGNode]] = defaultdict(list)
        self.layers_to_edges: defaultdict[
            int, set[tuple[MLGNode, MLGNode]]
        ] = defaultdict(set)
        self.nodes_to_in_edges: defaultdict[MLGNode, set[MLGNode]] = defaultdict(set)
        self.nodes_to_out_edges: defaultdict[MLGNode, set[MLGNode]] = defaultdict(set)

    def __copy__(self) -> "MultiLayeredGraph":
        cls = self.__class__
        _copy = cls(self.layer_count)
        _copy.layers_to_nodes = self.layers_to_nodes
        _copy.layers_to_edges = self.layers_to_edges
        _copy.nodes_to_in_edges = self.nodes_to_in_edges
        _copy.nodes_to_out_edges = self.nodes_to_out_edges

        return _copy

    def __deepcopy__(self, memo_dict: dict[Any, Any] | None = None):
        if memo_dict is None:
            memo_dict = {}

        cls = self.__class__
        _copy = cls.__new__(cls)
        memo_dict[id(self)] = _copy
        for k, v in self.__dict__.items():
            setattr(_copy, k, deepcopy(v, memo_dict))
        return _copy

    def add_real_node(self, layer: int, name: str | None = None) -> MLGNode:
        if layer < 0 or layer - 1 > self.layer_count:
            raise InvalidLayerError(f'Invalid layer "{layer}"')
        if name is None:
            name = f"{layer}_{len(self.layers_to_nodes[layer])}"
        node: MLGNode = MLGNode(layer, name)
        self.layers_to_nodes[layer].append(node)
        return node

    def add_virtual_node(self, layer: int, name_appendix: str) -> MLGNode:
        virtual_node_name = f"{layer}_vnode_{name_appendix}"
        virtual_node = MLGNode(layer, virtual_node_name, True)
        self.layers_to_nodes[layer].append(virtual_node)
        return virtual_node

    def add_edge(self, from_lower_node: MLGNode, to_upper_node: MLGNode) -> None:
        """Adds and edge between two nodes in a graph.

        The lower node must be on a lower layer than the upper node.
        Both nodes must exist on the correct layer in the graph.

        Args:
            from_lower_node: Node from which the edge to go.
            to_upper_node: Node to which the edge to go.
        """

        lower_layer = from_lower_node.layer
        upper_layer = to_upper_node.layer
        if lower_layer >= upper_layer:
            raise InvalidEdgeParamError(
                f"Lower node is higher up or at same layer as upper node."
                + f" {from_lower_node.layer} >= {to_upper_node.layer}"
            )
        if from_lower_node not in self.layers_to_nodes[from_lower_node.layer]:
            raise NoNodeExistsError(f"lower_node {from_lower_node} not in graph")
        if to_upper_node not in self.layers_to_nodes[to_upper_node.layer]:
            raise NoNodeExistsError(f"lower_node {to_upper_node} not in graph")

        prev_node = from_lower_node
        for layer_of_vnode in range(lower_layer + 1, upper_layer):
            # create long edge
            virtual_node = self.add_virtual_node(
                layer_of_vnode, f"{from_lower_node.name}_{to_upper_node.name}"
            )
            self._add_short_edge(prev_node, virtual_node)
            prev_node = virtual_node

        self._add_short_edge(prev_node, to_upper_node)

    def _add_short_edge(self, lower_node: MLGNode, upper_node: MLGNode) -> None:
        self.layers_to_edges[lower_node.layer].add((lower_node, upper_node))
        self.nodes_to_out_edges[lower_node].add(upper_node)
        self.nodes_to_in_edges[upper_node].add(lower_node)

    def to_networkx_graph(self) -> nx.Graph:
        nx_graph = nx.Graph()

        all_nodes_with_props = [
            (node, node.attrs_to_dict()) for node in self.all_nodes_as_list()
        ]
        nx_graph.add_nodes_from(all_nodes_with_props)  # type: ignore # partially unknown
        nx_graph.add_edges_from(self.all_edges_as_list())  # type: ignore # partially unknown
        return nx_graph

    def to_pygraphviz_graph(self) -> pgv.AGraph:
        pgv_graph = pgv.AGraph()
        pgv_graph.has_layout = True
        pgv_graph.graph_attr["splines"] = "spline"

        position_scale = 150
        positions = self.nodes_to_integer_relative_coordinates()
        for node in self.all_nodes_as_list():
            pos_x, pos_y = positions[node]
            pos_str = f"{pos_x * position_scale},{pos_y * position_scale}"
            attrs: dict[str, Any] = {"pos": pos_str}
            if node.is_virtual:
                attrs["label"] = ""
                # attrs["style"] = "invis"
                attrs["fixedsize"] = True
                attrs["width"] = 0.01
                attrs["height"] = 0.01

            # attrs["label"] = f"{(pos_x, pos_y)}"
            attrs["label"] = f"{node}"
            if node.text_info:
                attrs["label"] += f"\n{node.text_info}"

            pgv_graph.add_node(node, **attrs)  # type: ignore # partially unknown

        pgv_graph.add_edges_from(self.all_edges_as_list())  # type: ignore # partially unknown
        return pgv_graph

    def all_edges_as_list(self) -> list[tuple[MLGNode, MLGNode]]:
        all_edges: list[MLGNodeEdge_T] = []
        for nodes_at_layer in self.layers_to_edges.values():
            all_edges.extend(nodes_at_layer)

        return all_edges

    def all_nodes_as_list(self) -> list[MLGNode]:
        all_nodes: list[MLGNode] = []
        for layer in range(self.layer_count):
            nodes_at_layer = self.layers_to_nodes[layer]
            all_nodes.extend(nodes_at_layer)
        return all_nodes

    def nodes_to_index_within_layer_for_whole_graph(self) -> dict[MLGNode, int]:
        indices: dict[MLGNode, int] = {}
        for nodes_at_layer in self.layers_to_nodes.values():
            for i, node in enumerate(nodes_at_layer):
                indices[node] = i
        return indices

    def nodes_to_integer_relative_coordinates(self) -> dict[MLGNode, tuple[int, int]]:
        positions: dict[MLGNode, tuple[int, int]] = {}
        for layer, nodes_at_layer in self.layers_to_nodes.items():
            for i, node in enumerate(nodes_at_layer):
                positions[node] = (i, layer)
        return positions

    def get_crossings_per_layer(self) -> list[int]:
        curr_positions = self.nodes_to_index_within_layer_for_whole_graph()
        crossing_count_at_layer: list[int] = []
        for layer in range(self.layer_count):
            edges = list(self.layers_to_edges[layer])
            crossings = 0
            # two edges tw and uv cross if and only if (x(t) - x(U))(x(w) - x(v)) is negative
            for i in range(len(edges)):
                u, v = edges[i]
                u_pos = curr_positions[u]
                v_pos = curr_positions[v]
                for j in range(i + 1, len(edges)):
                    t, w = edges[j]
                    t_pos = curr_positions[t]
                    w_pos = curr_positions[w]
                    edges_cross = (t_pos - u_pos) * (w_pos - v_pos) < 0
                    crossings += edges_cross
            crossing_count_at_layer.append(crossings)
        return crossing_count_at_layer

    def get_total_crossings(self):
        return sum(self.get_crossings_per_layer())

    def nodes_to_indices_at_layer(self, layer_idx: int) -> dict[MLGNode, int]:
        return {
            node: index for index, node in enumerate(self.layers_to_nodes[layer_idx])
        }
