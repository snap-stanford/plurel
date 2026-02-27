import warnings

import numpy as np
import pytest

from plurel.config import DAGParams
from plurel.dag import DAG_REGISTRY, EDGE_WEIGHT_SAMPLERS

NODE_COUNTS = [10, 20, 50, 100]


@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize("dag_class", list(DAG_REGISTRY.values()))
@pytest.mark.parametrize("num_nodes", NODE_COUNTS)
def test_dag(seed, dag_class, num_nodes):
    dag = dag_class(num_nodes=num_nodes, dag_params=DAGParams(), seed=seed)
    assert dag.size() == num_nodes, dag.layers

    sources = [node_id for node_id in dag.graph.nodes if dag.graph.in_degree(node_id) == 0]
    assert len(sources) > 0


@pytest.mark.parametrize("dag_name", list(DAG_REGISTRY.keys()))
@pytest.mark.parametrize("seed", range(10))
def test_dag_no_overflow(dag_name, seed):
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        dag = DAG_REGISTRY[dag_name](num_nodes=30, dag_params=DAGParams())
    for _, _, data in dag.graph.edges(data=True):
        assert np.isfinite(data["weight"]), (
            f"{dag_name} seed={seed} produced non-finite edge weight: {data['weight']}"
        )


@pytest.mark.parametrize("dist_name", list(EDGE_WEIGHT_SAMPLERS.keys()))
def test_edge_weight_sampler_finite(dist_name):
    np.random.seed(0)
    sampler = EDGE_WEIGHT_SAMPLERS[dist_name]
    samples = [float(sampler()) for _ in range(10_000)]
    finite = [np.isfinite(s) for s in samples]
    assert all(finite), f"{dist_name} produced {finite.count(False)} non-finite samples"
