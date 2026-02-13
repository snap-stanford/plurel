import numpy as np
from torch_frame import stype

from plurel.config import DAGParams, SCMParams
from plurel.dag import DAG_REGISTRY
from plurel.scm import SCM
from plurel.utils import TableType


def test_scm():
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
            seed=table_id,
        )
        df = scm.generate_df(
            num_rows=10,
            table_type=(TableType.Entity if np.random.rand() < 0.5 else TableType.Activity),
        )
        assert len(df) == 10
        scms.append(scm)
