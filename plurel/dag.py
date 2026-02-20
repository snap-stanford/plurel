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

    def ensure_connected(self, graph: nx.Graph) -> nx.Graph:
        """Ensure the graph is connected by adding edges between disconnected components."""
        if nx.is_connected(graph):
            return graph
        graph = graph.copy()
        comps = list(nx.connected_components(graph))
        for c1, c2 in zip(comps[:-1], comps[1:]):
            u = np.random.choice(list(c1))
            v = np.random.choice(list(c2))
            graph.add_edge(u, v)
        return graph

    def order_edges(self, edges: list):
        return [(src, dst) if src < dst else (dst, src) for (src, dst) in edges]

    def add_edge_weights(self, edges: list):
        return [
            (
                src,
                dst,
                {"weight": np.random.randn()},
            )
            for (src, dst) in edges
        ]

    def _isolated_graph(self, num_nodes: int) -> nx.DiGraph:
        """Return a DiGraph with num_nodes isolated nodes and no edges."""
        g = nx.DiGraph()
        g.add_nodes_from(range(num_nodes))
        return g

    @abc.abstractmethod
    def sample(self, **kwargs) -> nx.DiGraph:
        pass


class ErdosRenyi(DAG):
    """
    Create DAGs with the erdos-renyi structure
    """

    def sample(self, num_nodes: int) -> nx.DiGraph:
        if num_nodes < 2:
            return self._isolated_graph(num_nodes)
        p = self.dag_params.er_p_choices.sample_uniform()
        graph = nx.gnp_random_graph(n=num_nodes, p=p, seed=self.seed, directed=True)
        graph = self.ensure_connected(graph.to_undirected())
        ordered_edges = self.order_edges(edges=graph.edges())
        weighted_edges = self.add_edge_weights(edges=ordered_edges)
        return nx.DiGraph(weighted_edges)


class BarabasiAlbert(DAG):
    def shuffle_nodes(self, graph):
        shuffled_nodes = list(graph.nodes())
        np.random.shuffle(shuffled_nodes)
        mapping = {old: new for old, new in zip(graph.nodes(), shuffled_nodes)}
        graph = nx.relabel_nodes(G=graph, mapping=mapping)
        return graph

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
        if num_nodes < 2:
            return self._isolated_graph(num_nodes)
        m = min(self.dag_params.ba_m, num_nodes - 1)
        graph = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=self.seed)
        graph = self.ensure_connected(graph)
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
        if num_nodes < 2:
            return self._isolated_graph(num_nodes)
        graph = nx.random_labeled_tree(num_nodes, seed=self.seed)
        graph = self.ensure_connected(graph)
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
        if num_nodes < 2:
            return self._isolated_graph(num_nodes)

        p = self.dag_params.ws_rewire_p_choices.sample_uniform()
        k = np.random.choice([2, 4])
        if k > num_nodes:
            k = num_nodes if num_nodes % 2 == 0 else num_nodes - 1
        if k < 2:
            k = 2

        graph = nx.watts_strogatz_graph(
            n=num_nodes,
            k=k,
            p=p,
            seed=self.seed,
        )

        graph = self.ensure_connected(graph)

        ordered_edges = self.order_edges(graph.edges())
        weighted_edges = self.add_edge_weights(ordered_edges)
        return nx.DiGraph(weighted_edges)


class RandomCauchy(DAG):
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def sample(self, num_nodes: int) -> nx.DiGraph:
        if num_nodes < 2:
            return self._isolated_graph(num_nodes)

        A = np.random.standard_cauchy()
        B = np.random.standard_cauchy(num_nodes)
        C = np.random.standard_cauchy(num_nodes)

        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                p_ij = self._sigmoid(A + B[i] + C[j])
                if np.random.rand() < p_ij:
                    edges.append((i, j))

        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(edges)
        graph = self.ensure_connected(graph)
        ordered_edges = self.order_edges(graph.edges())
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
        if num_nodes < 2:
            return self._isolated_graph(num_nodes)
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
    "RandomCauchy": RandomCauchy,
}
