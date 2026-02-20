import pytest

from plurel.config import DAGParams
from plurel.dag import DAG_REGISTRY

NODE_COUNTS = [10, 20, 50, 100]


@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize("dag_class", list(DAG_REGISTRY.values()))
@pytest.mark.parametrize("num_nodes", NODE_COUNTS)
def test_dag(seed, dag_class, num_nodes):
    dag = dag_class(num_nodes=num_nodes, dag_params=DAGParams(), seed=seed)
    assert dag.size() == num_nodes, dag.layers

    sources = [node_id for node_id in dag.graph.nodes if dag.graph.in_degree(node_id) == 0]
    assert len(sources) > 0
