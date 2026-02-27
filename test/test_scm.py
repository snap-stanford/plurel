import warnings

import numpy as np
import pytest
from torch_frame import stype

from plurel.config import DAGParams, SCMParams
from plurel.dag import DAG_REGISTRY
from plurel.scm import SCM, SOURCE_GEN_REGISTRY
from plurel.utils import TableType


def _make_scm(seed: int = 0) -> SCM:
    return SCM(
        table_name="t",
        child_table_names=[],
        feature_columns={
            "num_col": {"_stype": stype.numerical, "categories": None},
            "cat_col": {"_stype": stype.categorical, "categories": [0, 1, 2]},
        },
        pkey_col="id",
        fkey_col_to_pkey_table={},
        foreign_scm_info={},
        scm_params=SCMParams(),
        dag_params=DAGParams(),
        seed=seed,
    )


@pytest.mark.parametrize("seed", list(range(5)))
def test_scm(seed):
    scms = []
    table_ids = list(range(len(DAG_REGISTRY)))
    dag_classes = list(DAG_REGISTRY.keys())
    for table_id, dag_class in zip(table_ids, dag_classes):
        child_table_ids = table_ids[table_id + 1 :]
        parent_table_ids = table_ids[:table_id]

        scm = SCM(
            table_name=f"table_{table_id}",
            child_table_names=[f"table_{c_id}" for c_id in child_table_ids],
            feature_columns={
                "feature_0": {"_stype": stype.numerical, "categories": None},
                "feature_1": {
                    "_stype": stype.categorical,
                    "categories": ["active", "inactive"],
                },
            },
            pkey_col="row_idx",
            fkey_col_to_pkey_table={},
            foreign_scm_info={},
            scm_params=SCMParams(),
            dag_params=DAGParams(),
            seed=seed * 100 + table_id,
        )
        df = scm.generate_df(
            num_rows=10,
            table_type=(TableType.Entity if np.random.rand() < 0.5 else TableType.Activity),
        )
        assert len(df) == 10
        scms.append(scm)


@pytest.mark.parametrize("gen_type", list(SOURCE_GEN_REGISTRY.keys()))
def test_source_gen_registry_numerical_finite(gen_type):
    np.random.seed(0)
    factory = SOURCE_GEN_REGISTRY[gen_type]
    gen = factory.make_numerical(
        scm_params=SCMParams(), num_rows=100, table_type=TableType.Activity
    )
    values = [gen.get_value(row_idx=i) for i in range(100)]
    assert all(np.isfinite(v) for v in values), (
        f"source gen '{gen_type}' produced non-finite values"
    )


@pytest.mark.parametrize("seed", range(10))
def test_scm_generate_df_no_inf_nan(seed):
    df = _make_scm(seed=seed).generate_df(num_rows=50, table_type=TableType.Entity)
    float_cols = df.select_dtypes(float)
    assert not float_cols.isin([float("inf"), float("-inf")]).any().any()
    assert not float_cols.isna().any().any()


@pytest.mark.parametrize("seed", range(10))
def test_scm_generate_df_no_overflow_warning(seed):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _make_scm(seed=seed).generate_df(num_rows=50, table_type=TableType.Entity)
