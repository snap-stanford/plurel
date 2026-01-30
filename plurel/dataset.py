import os
from typing import Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

import networkx as nx
from relbench.base import Database, Dataset, Table
from torch_frame import stype

from plurel.dag import DAG_REGISTRY
from plurel.scm import SCM
from plurel.utils import set_random_seed, TableType
from plurel.config import Config
from plurel.sql_parser import SQLAlchemySchemaGraphBuilder


class SyntheticDataset(Dataset):
    """
    A synthetic relbench compatible dataset.

    Args:
        seed (int): Random seed for reproducibility of dataset generation.
        config (Config): Config object with constants and sampling choices.
    """

    def __init__(self, seed: int, config: Config):
        self.seed = seed
        self.config = config
        set_random_seed(self.seed)
        self.initialize_timestamps()
        super().__init__(cache_dir=self.config.cache_dir)

    def initialize_timestamps(self):
        start_timestamp = self.config.database_params.min_timestamp
        end_timestamp = self.config.database_params.max_timestamp
        total_days = (end_timestamp - start_timestamp).days

        days = np.sort(np.random.choice(total_days, 2, replace=False))
        self.min_timestamp = start_timestamp + pd.Timedelta(days=int(days[0]))
        self.max_timestamp = start_timestamp + pd.Timedelta(days=int(days[1]))

        timestamps = pd.date_range(start=self.min_timestamp, end=self.max_timestamp)
        val_start_idx = int(len(timestamps) * self.config.val_split)
        test_start_idx = int(len(timestamps) * self.config.test_split)
        self.val_timestamp = timestamps[val_start_idx]
        self.test_timestamp = timestamps[test_start_idx]

    def _get_random_dag_table_relationships(self, num_tables: int):
        """
        Each table will have the following attributes:
        ```py
        {
            "columns":dict[col_name -> {
                "stype": stype,
                "categories": list[str] | None
            }],
            "pkey_col": str | None,
            "fkey_col_to_pkey_table": dict[str, str],
        }
        ```
        """
        dag_class = self.config.database_params.table_layout_choices.sample_uniform()
        dag = DAG_REGISTRY[dag_class](
            num_nodes=num_tables, dag_params=self.config.dag_params, seed=self.seed
        )
        table_relationships = dag.graph

        for table_id in table_relationships.nodes:
            table_relationships.nodes[table_id]["name"] = f"table_{table_id}"

        for table_id in table_relationships.nodes:
            # most tables are narrow with few being wide
            num_cols = self.config.database_params.num_cols_choices.sample_pl()
            feature_cols = [f"feature_{idx}" for idx in range(num_cols)]
            pkey_col = "row_idx"
            fkey_col_to_pkey_table = {
                f"foreign_row_{idx}": table_relationships.nodes[parent_table_id]["name"]
                for idx, parent_table_id in enumerate(
                    sorted(list(table_relationships.predecessors(table_id)))
                )
            }
            fkey_cols = list(fkey_col_to_pkey_table.keys())

            columns = {}
            for col in [pkey_col, *fkey_cols]:
                _stype = stype.categorical
                columns[col] = {
                    "_stype": stype.categorical,
                    "categories": None,  # since these are pk/fk
                }

            for feature_col in feature_cols:
                _stype = self.config.scm_params.col_stype_choices.sample_uniform()
                if _stype == stype.categorical:
                    num_categories = (
                        self.config.scm_params.num_categories_choices.sample_uniform()
                    )
                    categories = list(range(num_categories))
                else:
                    categories = None
                columns[feature_col] = {
                    "_stype": _stype,
                    "categories": categories,
                }

            metadata = {
                "columns": columns,
                "pkey_col": pkey_col,
                "fkey_col_to_pkey_table": fkey_col_to_pkey_table,
            }
            for k, v in metadata.items():
                table_relationships.nodes[table_id][k] = v

        return table_relationships

    def _get_schema_file_table_relationships(self, schema_file: str):
        builder = SQLAlchemySchemaGraphBuilder(sql_file=schema_file)
        builder.load_schema()
        return builder.build_graph()

    def configure_table_relationships(
        self, num_tables: Optional[int] = None, schema_file: Optional[str] = None
    ) -> nx.DiGraph:
        """
        Define the primary -> foreign key relationships between tables as a DAG.

        Args:
            num_tables (int): Number of tables for the layout based on random DAGs.
            schema_file (str): A predefined SQL based schema file.
        """
        assert (
            num_tables or schema_file
        ), "either `num_tables` or `schema_file` must not be None"
        # higher preference to schema file
        if schema_file:
            table_relationships = self._get_schema_file_table_relationships(
                schema_file=schema_file
            )
        elif num_tables:
            table_relationships = self._get_random_dag_table_relationships(
                num_tables=num_tables
            )

        activity_tables = [
            table
            for table in table_relationships.nodes
            if table_relationships.out_degree(table) == 0
        ]
        for table in table_relationships.nodes:
            if table in activity_tables:
                table_type = TableType.Activity
            else:
                table_type = TableType.Entity

            table_relationships.nodes[table]["type"] = table_type
            table_relationships.nodes[table]["num_rows"] = self.get_num_rows(
                table_type=table_type
            )
        return table_relationships

    def get_num_rows(self, table_type):
        if table_type == TableType.Entity:
            num_rows = (
                self.config.database_params.num_rows_entity_table_choices.sample_uniform()
            )
        if table_type == TableType.Activity:
            num_rows = (
                self.config.database_params.num_rows_activity_table_choices.sample_uniform()
            )
        return num_rows

    def implant_nan(self, df, pkey_col, fkey_cols):
        nan_perc = self.config.database_params.column_nan_perc.sample_uniform()
        num_nan_cells = int(np.floor(nan_perc * len(df)))
        for col_name, _type in df.dtypes.items():
            if col_name not in [pkey_col, *fkey_cols] and _type in [float]:
                nan_cells_idx = np.random.choice(
                    df.index, size=num_nan_cells, replace=False
                )
                df.loc[nan_cells_idx, col_name] = np.nan
        return df

    def process_categorical_data(self, df, feature_columns, pkey_col, fkey_cols):
        """As of now only binary classification is supported, so we need to convert int columns for categorical data into boolean."""
        for col_name, _type in df.dtypes.items():
            if col_name not in [pkey_col, *fkey_cols] and _type in [int]:
                feature_column_info = feature_columns[col_name]
                categories = feature_column_info["categories"]
                if not categories:
                    continue
                # convert all numeric categorical columns into boolean
                if type(categories[0]) == int:
                    df[col_name] = (df[col_name] > 0).astype(bool)
                    # drop columns which only have True/False values.
                    if len(df[col_name].unique()) == 1:
                        df.drop(columns=[col_name], inplace=True)
                elif type(categories[0]) == str:
                    df[col_name] = df[col_name].map(lambda i: categories[i])
        return df

    def make_db(self) -> Database:
        set_random_seed(self.seed)
        num_tables = self.config.database_params.num_tables_choices.sample_uniform()
        self.table_relationships = self.configure_table_relationships(
            num_tables=num_tables, schema_file=self.config.schema_file
        )
        topological_gens = list(nx.topological_generations(self.table_relationships))
        table_name_to_scm = {}

        total_gens = len(topological_gens)
        for gen_idx, gen in enumerate(topological_gens):
            for table_id in gen:
                table_type = self.table_relationships.nodes[table_id]["type"]
                table_name = self.table_relationships.nodes[table_id]["name"]
                num_rows = self.table_relationships.nodes[table_id]["num_rows"]
                columns = self.table_relationships.nodes[table_id]["columns"]
                pkey_col = self.table_relationships.nodes[table_id]["pkey_col"]
                fkey_col_to_pkey_table = self.table_relationships.nodes[table_id][
                    "fkey_col_to_pkey_table"
                ]
                feature_columns = {
                    col: col_info
                    for col, col_info in columns.items()
                    if col != pkey_col and col not in fkey_col_to_pkey_table.keys()
                }

                child_table_names = []
                for child_table_id in sorted(
                    list(self.table_relationships.successors(table_id))
                ):
                    child_table_names.append(
                        self.table_relationships.nodes[child_table_id]["name"]
                    )

                foreign_scm_info = {
                    foreign_table_name: table_name_to_scm[foreign_table_name]
                    for foreign_table_name in fkey_col_to_pkey_table.values()
                }
                scm = SCM(
                    table_name=table_name,
                    child_table_names=child_table_names,
                    feature_columns=feature_columns,
                    pkey_col=pkey_col,
                    fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                    foreign_scm_info=foreign_scm_info,
                    scm_params=self.config.scm_params,
                    dag_params=self.config.dag_params,
                )
                if table_type == TableType.Activity:
                    min_timestamp, max_timestamp = (
                        self.min_timestamp,
                        self.max_timestamp,
                    )
                else:
                    min_timestamp, max_timestamp = None, None
                scm.generate_df(
                    num_rows=num_rows,
                    table_type=table_type,
                    min_timestamp=min_timestamp,
                    max_timestamp=max_timestamp,
                )
                table_name_to_scm[table_name] = scm

        table_dict = {}

        for table_id in self.table_relationships.nodes:
            table_name = self.table_relationships.nodes[table_id]["name"]
            df = table_name_to_scm[table_name].df
            pkey_col = self.table_relationships.nodes[table_id]["pkey_col"]
            fkey_col_to_pkey_table = self.table_relationships.nodes[table_id][
                "fkey_col_to_pkey_table"
            ]
            columns = self.table_relationships.nodes[table_id]["columns"]
            feature_columns = {
                col: col_info
                for col, col_info in columns.items()
                if col != pkey_col and col not in fkey_col_to_pkey_table.keys()
            }
            ########## post-processing ###############
            df = self.implant_nan(
                df=df, pkey_col=pkey_col, fkey_cols=list(fkey_col_to_pkey_table.keys())
            )
            df = self.process_categorical_data(
                df=df,
                feature_columns=feature_columns,
                pkey_col=pkey_col,
                fkey_cols=list(fkey_col_to_pkey_table.keys()),
            )
            ##########################################
            time_col = "date" if "date" in df.columns else None
            table_dict[table_name] = Table(
                df=df,
                time_col=time_col,
                pkey_col=pkey_col,
                fkey_col_to_pkey_table=fkey_col_to_pkey_table,
            )

        return Database(table_dict)
