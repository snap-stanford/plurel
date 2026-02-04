import itertools

import pytest

from plurel.utils import get_bipartite_hsbm

sizes_a = [10, 30, 50]
sizes_b = [20, 60, 100]
hierarchies_a = [[2, 4], [4, 2], [2, 4]]
hierarchies_b = [[4, 2], [2, 4], [2, 8]]

param_grid = list(itertools.product(sizes_a, sizes_b, hierarchies_a, hierarchies_b))


@pytest.mark.parametrize("size_a, size_b, hierarchy_a, hierarchy_b", param_grid)
def test_get_bipartite_hsbm(size_a, size_b, hierarchy_a, hierarchy_b):
    bi_hsbm = get_bipartite_hsbm(
        size_a=size_a, size_b=size_b, hierarchy_a=hierarchy_a, hierarchy_b=hierarchy_b
    )

    for i in range(size_b):
        assert len(bi_hsbm.in_edges(f"b{i}")) == 1
