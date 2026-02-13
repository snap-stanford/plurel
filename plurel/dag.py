"""
Create DAGs with various strategies
"""

import abc

import networkx as nx
import numpy as np

from plurel.config import DAGParams
from plurel.utils import set_random_seed


class DAG(abc.ABC):
    """
    Base class for all DAGs.
    """

    def __init__(
        self,
        num_nodes: int,
        dag_params: DAGParams,
        seed: int | None = None,
    ):
        self.seed = seed
        if self.seed:
            set_random_seed(self.seed)
        self.dag_params = dag_params
        self.graph = self.sample(num_nodes=num_nodes)
        self.validate()

    def size(self):
        return len(self.graph.nodes)

    def validate(self):
        assert nx.is_directed_acyclic_graph(self.graph)
        assert nx.is_connected(self.graph.to_undirected(as_view=True))

    def order_edges(self, edges: list):
        return [(src, dst) for (src, dst) in edges if src < dst]

    def add_edge_weights(self, edges: list):
        return [
            (
                src,
                dst,
                {"weight": np.random.randn()},
            )
            for (src, dst) in edges
        ]

    @abc.abstractmethod
    def sample(self, **kwargs) -> nx.DiGraph:
        pass


class ErdosRenyi(DAG):
    """
    Create DAGs with the erdos-renyi structure
    """

    def sample(self, num_nodes: int) -> nx.DiGraph:
        is_connected = False
        while not is_connected:
            p = self.dag_params.er_p_choices.sample_uniform()
            graph = nx.gnp_random_graph(n=num_nodes, p=p, seed=self.seed, directed=True)
            ordered_edges = self.order_edges(edges=graph.edges())
            weighted_edges = self.add_edge_weights(edges=ordered_edges)
            graph = nx.DiGraph(weighted_edges)
            try:
                is_connected = (
                    nx.is_connected(graph.to_undirected(as_view=True))
                    and len(graph.nodes) == num_nodes
                )
            except Exception:
                is_connected = False
        return graph


class BarabasiAlbert(DAG):
    def shuffle_nodes(self, graph):
        shuffled_nodes = list(graph.nodes())
        np.random.shuffle(shuffled_nodes)
        mapping = {old: new for old, new in zip(graph.nodes(), shuffled_nodes)}
        new_graph = nx.relabel_nodes(G=graph, mapping=mapping)
        return new_graph

    def sparsify_leaves(self, graph: nx.DiGraph):
        sinks = [node for node in graph.nodes if graph.out_degree(node) == 0]
        graph = graph.copy()
        for node in sinks:
            incoming_edges = list(graph.in_edges(node))
            if len(incoming_edges) > 1 and np.random.rand() < self.dag_params.ba_sink_edge_dropout:
                edge_idx = np.random.randint(len(incoming_edges))
                edge = incoming_edges[edge_idx]
                graph.remove_edge(*edge)
        return graph

    def flip_leaf_edges(self, graph: nx.DiGraph):
        sinks = [node for node in graph.nodes if graph.out_degree(node) == 0]
        graph = graph.copy()
        for node in sinks:
            incoming_edges = list(graph.in_edges(node, data=True))
            if (
                len(incoming_edges) == 1
                and np.random.rand() < self.dag_params.ba_flip_leaf_edge_prob
            ):
                u, v, ddict = incoming_edges[0]
                if graph.in_degree(v) >= self.dag_params.ba_max_in_degree:
                    continue
                graph.remove_edge(u, v)
                graph.add_edge(v, u, **ddict)
        return graph

    def sample(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.barabasi_albert_graph(n=num_nodes, m=self.dag_params.ba_m, seed=self.seed)
        assert nx.is_connected(graph.to_undirected(as_view=True)), (
            "nx generated a non-connected graph. Adjust parameters to increase connectivity."
        )
        ordered_edges = self.order_edges(edges=graph.edges())
        weighted_edges = self.add_edge_weights(edges=ordered_edges)
        graph = nx.DiGraph(weighted_edges)
        graph = self.shuffle_nodes(graph)
        graph = self.sparsify_leaves(graph)
        graph = self.flip_leaf_edges(graph)
        return graph


class RandomTree(DAG):
    """
    Create DAGs with the random tree structure where edges point away from root node
    """

    def order_edges(self, edges: list):
        return edges

    def sample(self, num_nodes: int) -> nx.DiGraph:
        graph = nx.random_labeled_tree(num_nodes, seed=self.seed)
        assert nx.is_connected(graph.to_undirected(as_view=True)), (
            "nx generated a non-connected graph. Adjust parameters to increase connectivity."
        )
        root_node = np.random.choice(graph.nodes).item()
        directed_edges = list(nx.bfs_edges(graph, root_node))
        ordered_edges = self.order_edges(edges=directed_edges)
        weighted_edges = self.add_edge_weights(edges=ordered_edges)
        graph = nx.DiGraph(weighted_edges)
        return graph


class ReverseRandomTree(RandomTree):
    """
    Create DAGs with the reverse random tree structure where edges point to root node
    """

    def order_edges(self, edges: list):
        return [(dst, src) for (src, dst) in edges]


class WattsStrogatz(DAG):
    def sample(self, num_nodes: int) -> nx.DiGraph:
        p = self.dag_params.ws_rewire_p_choices.sample_uniform()
        k = np.random.choice([2, 4])
        if k > num_nodes:
            k = 2

        G = nx.watts_strogatz_graph(
            n=num_nodes,
            k=k,
            p=p,
            seed=self.seed,
        )

        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for c1, c2 in zip(comps[:-1], comps[1:]):
                u = np.random.choice(list(c1))
                v = np.random.choice(list(c2))
                G.add_edge(u, v)

        ordered_edges = self.order_edges(G.edges())
        weighted_edges = self.add_edge_weights(ordered_edges)
        return nx.DiGraph(weighted_edges)


class Layered(DAG):
    """
    Layered / MLP-style DAG.
    Structure:
      L0 -> L1 -> ... -> L(d-1)
    """

    def _distribute_nodes(self, num_nodes: int, depth: int):
        """
        Returns: List[int] = number of nodes per layer
        Ensures:
        - sum == num_nodes
        - each layer has at least 1 node
        """
        assert depth <= num_nodes

        # start with 1 per layer
        sizes = [1] * depth
        remaining = num_nodes - depth

        # randomly distribute the rest
        for _ in range(remaining):
            i = np.random.randint(depth)
            sizes[i] += 1

        return sizes

    def sample(self, num_nodes: int) -> nx.DiGraph:
        p_drop = self.dag_params.layered_edge_dropout_p
        depth = self.dag_params.layered_depth_choices.sample_uniform()
        depth = min(depth, num_nodes)

        layer_sizes = self._distribute_nodes(
            num_nodes=num_nodes,
            depth=depth,
        )

        # nodes to layers
        layers = []
        cur = 0
        for sz in layer_sizes:
            layer_nodes = list(range(cur, cur + sz))
            layers.append(layer_nodes)
            cur += sz

        # layered edges
        edges = []
        for l in range(len(layers) - 1):
            srcs = layers[l]
            dsts = layers[l + 1]

            for u in srcs:
                for v in dsts:
                    if np.random.rand() < p_drop:
                        continue
                    edges.append((u, v))

        # every node in layer l+1 must have at least one parent
        for l in range(len(layers) - 1):
            srcs = layers[l]
            dsts = layers[l + 1]
            for v in dsts:
                if not any((u, v) in edges for u in srcs):
                    u = np.random.choice(srcs)
                    edges.append((u, v))

        # every node in layer l must have at least one child
        for l in range(len(layers) - 1):
            srcs = layers[l]
            dsts = layers[l + 1]
            for u in srcs:
                if not any((u, v) in edges for v in dsts):
                    v = np.random.choice(dsts)
                    edges.append((u, v))

        weighted_edges = self.add_edge_weights(edges)
        return nx.DiGraph(weighted_edges)


DAG_REGISTRY = {
    "ErdosRenyi": ErdosRenyi,
    "BarabasiAlbert": BarabasiAlbert,
    "RandomTree": RandomTree,
    "ReverseRandomTree": ReverseRandomTree,
    "WattsStrogatz": WattsStrogatz,
    "Layered": Layered,
}
