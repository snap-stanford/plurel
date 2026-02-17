import networkx as nx
import numpy as np
from tqdm import tqdm


def assign_cluster_at_levels(num_nodes: int, hierarchy: list):
    num_base_clusters = np.prod(hierarchy)
    nodes_per_cluster = int(np.ceil(num_nodes / num_base_clusters))
    base_cluster_offsets = [
        nodes_per_cluster * (c_idx + 1) for c_idx in range(num_base_clusters - 1)
    ]
    base_cluster_offsets.append(num_nodes)
    cluster_at_levels = np.zeros((num_nodes, len(hierarchy)), dtype=int)
    cluster_node_idx_start = 0
    for c_idx in range(num_base_clusters):
        for l_idx in range(len(hierarchy)):
            fac = np.prod(hierarchy[l_idx + 1 :]) if l_idx + 1 < len(hierarchy) else 1
            cluster_node_idx_end = base_cluster_offsets[c_idx]
            cluster_at_levels[cluster_node_idx_start:cluster_node_idx_end, l_idx] = (
                c_idx // fac
            ) % hierarchy[l_idx]
        cluster_node_idx_start = cluster_node_idx_end
    return cluster_at_levels


def get_probs_at_levels(hierarchy_a: list, hierarchy_b: list):
    assert len(hierarchy_a) == len(hierarchy_b), "only similar hierarchy levels are supported"

    num_levels = len(hierarchy_a)
    probs_at_levels = []
    for l_idx in range(num_levels):
        shape = (hierarchy_a[l_idx], hierarchy_b[l_idx])
        probs = np.random.uniform(0.001, 0.002, size=shape)
        for i in range(max(shape)):
            probs[i % shape[0], i % shape[1]] = 0.9
        probs_at_levels.append(probs)
    return probs_at_levels


def get_nodes_connect_prob(
    node_idx_a: int,
    node_idx_b: int,
    probs_at_levels: list,
    cluster_at_levels_a: list,
    cluster_at_levels_b: list,
):
    num_levels = len(probs_at_levels)
    probs = [
        probs_at_levels[l_idx][
            cluster_at_levels_a[node_idx_a, l_idx],
            cluster_at_levels_b[node_idx_b, l_idx],
        ]
        for l_idx in range(num_levels)
    ]
    return np.prod(probs)


def get_bipartite_hsbm(size_a: int, size_b: int, hierarchy_a: list, hierarchy_b: list):
    assert len(hierarchy_a) == len(hierarchy_b), "only similar hierarchy levels are supported"

    cluster_at_levels_a = assign_cluster_at_levels(num_nodes=size_a, hierarchy=hierarchy_a)
    cluster_at_levels_b = assign_cluster_at_levels(num_nodes=size_b, hierarchy=hierarchy_b)
    probs_at_levels = get_probs_at_levels(hierarchy_a=hierarchy_a, hierarchy_b=hierarchy_b)

    bi_hsbm = nx.DiGraph()

    nodes_a = [f"a{i}" for i in range(size_a)]
    nodes_b = [f"b{j}" for j in range(size_b)]

    for a_idx, a_node in enumerate(nodes_a):
        bi_hsbm.add_node(
            a_node,
            node_idx=a_idx,
            hierarchy=list(cluster_at_levels_a[a_idx]),
        )
    for b_idx, b_node in enumerate(nodes_b):
        bi_hsbm.add_node(
            b_node,
            node_idx=b_idx,
            hierarchy=list(cluster_at_levels_b[b_idx]),
        )

    for b_idx, b_node in tqdm(enumerate(nodes_b), desc="adding edges in bi_hsbm", leave=False):
        probs = np.array(
            [
                get_nodes_connect_prob(
                    node_idx_a=a_idx,
                    node_idx_b=b_idx,
                    probs_at_levels=probs_at_levels,
                    cluster_at_levels_a=cluster_at_levels_a,
                    cluster_at_levels_b=cluster_at_levels_b,
                )
                for a_idx in range(size_a)
            ]
        )
        try:
            probs = probs / probs.sum()
        except ValueError:
            probs = None
        a_idx = np.random.choice(range(size_a), p=probs)
        bi_hsbm.add_edge(nodes_a[a_idx], b_node)

    assert nx.is_bipartite(bi_hsbm)
    return bi_hsbm


def get_bipartite_pl(size_a: int, size_b: int, exponent: float):
    bi_pl = nx.DiGraph()

    nodes_a = [f"a{i}" for i in range(size_a)]
    nodes_b = [f"b{j}" for j in range(size_b)]

    for a_idx, a_node in enumerate(nodes_a):
        bi_pl.add_node(a_node, node_idx=a_idx, bipartite=0)
    for b_idx, b_node in enumerate(nodes_b):
        bi_pl.add_node(b_node, node_idx=b_idx, bipartite=1)

    shuffled_a_indxs = np.arange(size_a)
    np.random.shuffle(shuffled_a_indxs)
    for b_idx, b_node in tqdm(enumerate(nodes_b), desc="adding edges in bi_pl", leave=False):
        probs = np.array(
            [1 - np.pow(shuffled_a_indxs[a_idx] / size_a, exponent) for a_idx in range(size_a)]
        )
        try:
            probs = probs / probs.sum()
        except ValueError:
            probs = None
        a_idx = np.random.choice(range(size_a), p=probs)
        bi_pl.add_edge(nodes_a[a_idx], b_node)

    assert nx.is_bipartite(bi_pl)

    return bi_pl
