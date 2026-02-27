"""
Generate synthetic tabular data using Structured Causal Models.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.distributions import Beta
from torch_frame import stype
from tqdm import tqdm

from plurel.bipartite import get_bipartite_hsbm
from plurel.config import DAGParams, SCMParams
from plurel.dag import DAG_REGISTRY
from plurel.transforms import MLP, CategoricalDecoder, CategoricalEncoder
from plurel.ts import (
    BetaSourceGenerator,
    CategoricalTSDataGenerator,
    GaussianSourceGenerator,
    IIDCategoricalGenerator,
    MixedSourceGenerator,
    TSDataGenerator,
    UniformSourceGenerator,
)
from plurel.utils import TableType, set_random_seed


class SourceGenFactory:
    """Base class for source data generator factories."""

    def make_numerical(self, scm_params: SCMParams, num_rows: int, table_type: TableType):
        raise NotImplementedError

    def make_categorical(
        self,
        scm_params: SCMParams,
        num_categories: int,
        num_rows: int,
        table_type: TableType,
    ):
        raise NotImplementedError


class TSSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        min_value, max_value = sorted(np.random.uniform(size=2) * scale)
        trend_scale = (
            scm_params.activity_table_ts_trend_scale_choices.sample_uniform()
            if table_type == TableType.Activity
            else scm_params.entity_table_ts_trend_scale
        )
        noise_scale = (
            scm_params.activity_table_ts_noise_scale
            if table_type == TableType.Activity
            else scm_params.entity_table_ts_noise_scale
        )
        cycle_scale = (
            scm_params.activity_table_ts_cycle_scale_choices.sample_uniform()
            if table_type == TableType.Activity
            else scm_params.entity_table_ts_cycle_scale
        )
        return TSDataGenerator(
            num_points=num_rows,
            min_value=min_value,
            max_value=max_value,
            trend_alpha=scm_params.ts_trend_alpha_choices.sample_uniform(),
            trend_scale=trend_scale,
            cycle_frequency=np.ceil(
                num_rows * scm_params.ts_cycle_freq_perc_choices.sample_uniform()
            ),
            cycle_scale=cycle_scale,
            noise_scale=noise_scale,
            ar_rho=scm_params.ts_ar_rho_choices.sample_uniform(),
        )

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return CategoricalTSDataGenerator(
            [
                self.make_numerical(scm_params=scm_params, num_rows=num_rows, table_type=table_type)
                for _ in range(num_categories)
            ]
        )


class UniformSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        low, high = sorted(np.random.uniform(size=2) * scale)
        return UniformSourceGenerator(low=low, high=high)

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class GaussianSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        low, high = sorted(np.random.uniform(size=2) * scale)
        return GaussianSourceGenerator(
            mean=(low + high) / 2,
            std=max((high - low) / 4, 1e-6),
            low=low,
            high=high,
        )

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class BetaSourceGenFactory(SourceGenFactory):
    def make_numerical(self, scm_params, num_rows, table_type):
        scale = float(scm_params.ts_value_scale_choices.sample_uniform())
        low, high = sorted(np.random.uniform(size=2) * scale)
        alpha = float(scm_params.source_beta_alpha_choices.sample_uniform())
        beta = float(scm_params.source_beta_beta_choices.sample_uniform())
        return BetaSourceGenerator(alpha=alpha, beta=beta, scale=high - low, offset=low)

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


class MixedSourceGenFactory(SourceGenFactory):
    _sub_types = ["uniform", "gaussian", "beta"]

    def make_numerical(self, scm_params, num_rows, table_type):
        sub_types = np.random.choice(self._sub_types, size=np.random.randint(2, 4))
        return MixedSourceGenerator(
            [
                SOURCE_GEN_REGISTRY[t].make_numerical(
                    scm_params=scm_params, num_rows=num_rows, table_type=table_type
                )
                for t in sub_types
            ]
        )

    def make_categorical(self, scm_params, num_categories, num_rows, table_type):
        return IIDCategoricalGenerator(num_categories=num_categories)


SOURCE_GEN_REGISTRY: dict[str, SourceGenFactory] = {
    "ts": TSSourceGenFactory(),
    "uniform": UniformSourceGenFactory(),
    "gaussian": GaussianSourceGenFactory(),
    "beta": BetaSourceGenFactory(),
    "mixed": MixedSourceGenFactory(),
}


class PropagationStrategy:
    """Base class for SCM propagation strategies.

    To add a new mode: subclass, implement all abstract methods, and register
    in PROPAGATION_STRATEGY_REGISTRY.
    """

    def __init__(self, scm_params: SCMParams):
        self.scm_params = scm_params
        self.mlp_emb_dim = int(scm_params.mlp_emb_dim_choices.sample_uniform())

    def _get_numerical_encoder(self):
        return MLP(
            scm_params=self.scm_params,
            in_dim=self.scm_params.mlp_in_dim,
            hid_dim=self.mlp_emb_dim,
            out_dim=self.mlp_emb_dim,
        )

    def _get_numerical_decoder(self):
        return MLP(
            scm_params=self.scm_params,
            in_dim=self.mlp_emb_dim,
            hid_dim=self.mlp_emb_dim,
            out_dim=self.scm_params.mlp_out_dim,
        )

    def _get_categorical_encoder(self, num_categories: int):
        return CategoricalEncoder(
            scm_params=self.scm_params,
            num_embeddings=num_categories,
            embedding_dim=self.mlp_emb_dim,
        )

    def _get_categorical_decoder(self, num_categories: int):
        return CategoricalDecoder(
            scm_params=self.scm_params,
            num_embeddings=num_categories,
            embedding_dim=self.mlp_emb_dim,
        )

    def get_encoder(self, _stype: stype, num_categories: int | None = None):
        return {
            stype.numerical: self._get_numerical_encoder,
            stype.categorical: lambda: self._get_categorical_encoder(num_categories=num_categories),
        }[_stype]()

    def get_decoder(self, _stype: stype, num_categories: int | None = None):
        return {
            stype.numerical: self._get_numerical_decoder,
            stype.categorical: lambda: self._get_categorical_decoder(num_categories=num_categories),
        }[_stype]()

    def _get_source_data_gen(self, node_stype, num_categories, num_rows, table_type):
        source_gen_type = self.scm_params.source_gen_type_choices.sample_uniform()
        source_gen_factory = SOURCE_GEN_REGISTRY[source_gen_type]
        return {
            stype.numerical: lambda: source_gen_factory.make_numerical(
                scm_params=self.scm_params, num_rows=num_rows, table_type=table_type
            ),
            stype.categorical: lambda: source_gen_factory.make_categorical(
                scm_params=self.scm_params,
                num_categories=num_categories,
                num_rows=num_rows,
                table_type=table_type,
            ),
        }[node_stype]()

    def make_ts_data_gen(self, node_stype, num_categories, num_rows, table_type):
        raise NotImplementedError

    def make_node_encoder(self, _stype, num_categories):
        raise NotImplementedError

    def make_node_decoder(self, _stype, num_categories):
        raise NotImplementedError

    def make_edge_encoder(self, _stype, num_categories):
        raise NotImplementedError

    def tensorize_source(self, value, _stype) -> torch.Tensor:
        raise NotImplementedError

    def tensorize_col(self, value, _stype) -> torch.Tensor:
        raise NotImplementedError

    def post_generate(self, scm) -> None:
        """Called after all rows are generated. Override for post-processing."""
        pass


class EagerPropagationStrategy(PropagationStrategy):
    """Type-aware strategy: uses type-specific encoders/decoders throughout."""

    def make_ts_data_gen(self, node_stype, num_categories, num_rows, table_type):
        return self._get_source_data_gen(node_stype, num_categories, num_rows, table_type)

    def make_node_encoder(self, _stype, num_categories):
        return self.get_encoder(_stype=_stype, num_categories=num_categories)

    def make_node_decoder(self, _stype, num_categories):
        return self.get_decoder(_stype=_stype, num_categories=num_categories)

    def make_edge_encoder(self, _stype, num_categories):
        return self.get_encoder(_stype=_stype, num_categories=num_categories)

    def tensorize_source(self, value, _stype):
        if _stype == stype.categorical:
            return torch.LongTensor([value])
        return torch.Tensor([value])

    def tensorize_col(self, value, _stype):
        if _stype == stype.categorical:
            return torch.LongTensor([value])
        return torch.Tensor([value])


class LazyPropagationStrategy(PropagationStrategy):
    """Lazy strategy: treats all values as numerical; quantizes categoricals post-hoc."""

    def make_ts_data_gen(self, node_stype, num_categories, num_rows, table_type):
        source_gen_type = self.scm_params.source_gen_type_choices.sample_uniform()
        return SOURCE_GEN_REGISTRY[source_gen_type].make_numerical(
            scm_params=self.scm_params, num_rows=num_rows, table_type=table_type
        )

    def make_node_encoder(self, _stype, num_categories):
        return self.get_encoder(_stype=stype.numerical)

    def make_node_decoder(self, _stype, num_categories):
        return self.get_decoder(_stype=stype.numerical)

    def make_edge_encoder(self, _stype, num_categories):
        return self.get_encoder(_stype=stype.numerical)

    def tensorize_source(self, value, _stype):
        return torch.Tensor([value])

    def tensorize_col(self, value, _stype):
        return torch.Tensor([value])

    def post_generate(self, scm):
        scm._apply_categorical_quantization()


PROPAGATION_STRATEGY_REGISTRY: dict[str, type[PropagationStrategy]] = {
    "type_eager": EagerPropagationStrategy,
    "type_lazy": LazyPropagationStrategy,
}

AGGREGATION_REGISTRY: dict[str, callable] = {
    "sum": lambda s: s.sum(dim=0),
    "max": lambda s: s.max(dim=0).values,
    "product": lambda s: s.prod(dim=0),
    "logexp": lambda s: torch.logsumexp(s, dim=0),
}


class SCM:
    def __init__(
        self,
        table_name: str,
        child_table_names: list[str],
        feature_columns: dict[str, stype],
        pkey_col: str,
        fkey_col_to_pkey_table: dict[str, str],
        foreign_scm_info: dict[str, SCM],
        scm_params: SCMParams,
        dag_params: DAGParams,
        seed: int | None = None,
    ):
        self.table_name = table_name
        self.child_table_names = child_table_names
        self.num_col_nodes = len(feature_columns)
        self.feature_columns = feature_columns
        self.pkey_col = pkey_col
        self.fkey_col_to_pkey_table = fkey_col_to_pkey_table
        self.foreign_scm_info = foreign_scm_info
        self.scm_params = scm_params
        self.dag_params = dag_params
        self.seed = seed
        if self.seed:
            set_random_seed(seed=self.seed)

        self.propagation_mode = self.scm_params.propagation_mode_choices.sample_uniform()
        self.strategy = PROPAGATION_STRATEGY_REGISTRY[self.propagation_mode](
            scm_params=self.scm_params
        )

        self.validate_foreign_scms()
        self.initialize_dag()
        self.initialize_nodes_and_edges()

    def initialize_dag(self):
        dag_class = self.scm_params.scm_layout_choices.sample_uniform()
        dag_class = DAG_REGISTRY[dag_class]
        scm_col_node_perc = self.scm_params.scm_col_node_perc_choices.sample_uniform()
        num_nodes = (
            self.num_col_nodes
            + int(np.ceil(self.num_col_nodes / scm_col_node_perc))
            + 3  # buffer to avoid empty graphs
        )
        self.dag = dag_class(num_nodes=num_nodes, dag_params=self.dag_params, seed=self.seed)
        self.num_edges = len(self.dag.graph.edges)

    def validate_foreign_scms(self):
        for foreign_table_name, scm in self.foreign_scm_info.items():
            assert hasattr(scm, "df"), (
                f"the foreign_scm for table: {foreign_table_name} does not contain the `self.df` attribute"
            )

    def initialize_ts_data_gens(self, num_rows: int, table_type: TableType):
        self.source_node_to_ts_data_gen = {}
        for node in self.source_nodes:
            _stype = self.dag.graph.nodes[node]["_stype"]
            num_categories = self.dag.graph.nodes[node]["num_categories"]
            self.source_node_to_ts_data_gen[node] = self.strategy.make_ts_data_gen(
                node_stype=_stype,
                num_categories=num_categories,
                num_rows=num_rows,
                table_type=table_type,
            )

    def initialize_nodes_and_edges(self):
        self.source_nodes = [
            node for node in self.dag.graph.nodes if self.dag.graph.in_degree(node) == 0
        ]
        self.col_nodes = np.random.choice(
            a=self.dag.graph.nodes, size=self.num_col_nodes, replace=False
        )
        for col_idx, (col_name, col_info) in enumerate(self.feature_columns.items()):
            col_node = self.col_nodes[col_idx]
            self.dag.graph.nodes[col_node]["col_name"] = col_name
            self.dag.graph.nodes[col_node]["_stype"] = col_info["_stype"]
            self.dag.graph.nodes[col_node]["num_categories"] = (
                len(col_info["categories"]) if col_info["categories"] else None
            )

        self._reset_node_attributes()
        self._reset_edge_attributes()

    def _reset_node_attributes(self):
        for node in self.dag.graph.nodes:
            if node in self.col_nodes:
                _stype = self.dag.graph.nodes[node]["_stype"]
                num_categories = self.dag.graph.nodes[node]["num_categories"]
            else:
                _stype = self.scm_params.col_stype_choices.sample_uniform()
                self.dag.graph.nodes[node]["_stype"] = _stype
                num_categories = (
                    self.scm_params.num_categories_choices.sample_uniform()
                    if _stype == stype.categorical
                    else None
                )
                self.dag.graph.nodes[node]["num_categories"] = num_categories

            alpha = float(self.scm_params.node_noise_alpha_choices.sample_uniform())
            beta = float(self.scm_params.node_noise_beta_choices.sample_uniform())
            self.dag.graph.nodes[node]["noise_dist"] = Beta(
                torch.tensor([alpha]), torch.tensor([beta])
            )
            self.dag.graph.nodes[node]["propagation_agg"] = (
                self.scm_params.propagation_agg_choices.sample_uniform()
            )
            self.dag.graph.nodes[node]["decoder"] = self.strategy.make_node_decoder(
                _stype=_stype, num_categories=num_categories
            )
            if node in self.col_nodes:
                self.dag.graph.nodes[node]["collation_encoders"] = {
                    (
                        self.table_name,
                        child_table_name,
                    ): self.strategy.make_node_encoder(_stype=_stype, num_categories=num_categories)
                    for child_table_name in self.child_table_names
                }

    def _reset_edge_attributes(self):
        for parent_node, child_node in self.dag.graph.edges:
            parent_stype = self.dag.graph.nodes[parent_node]["_stype"]
            parent_num_categories = self.dag.graph.nodes[parent_node]["num_categories"]
            self.dag.graph.edges[parent_node, child_node]["encoder"] = (
                self.strategy.make_edge_encoder(
                    _stype=parent_stype, num_categories=parent_num_categories
                )
            )

    def _aggregate_embeddings(
        self, embs: list[torch.Tensor], weights: list[float], mode: str
    ) -> torch.Tensor:
        weighted = [w * e for w, e in zip(weights, embs)]
        stack = torch.stack(weighted, dim=0)  # (n, emb_dim)
        if mode not in AGGREGATION_REGISTRY:
            raise ValueError(f"Unknown aggregation mode: {mode}")
        return AGGREGATION_REGISTRY[mode](stack)

    def propagate(self, row_idx: int, foreign_row_idxs: list[int], foreign_scms: list[SCM]):
        foreign_scms_row_embds: list[list] = []
        for foreign_row_idx, foreign_scm in zip(foreign_row_idxs, foreign_scms):
            foreign_row_embds = foreign_scm.collate_feature_embeddings(
                row_idx=foreign_row_idx, child_table_name=self.table_name
            )
            foreign_scms_row_embds.append(foreign_row_embds)

        topological_gens = nx.topological_generations(self.dag.graph)
        for gen in topological_gens:
            for node in gen:
                node_stype = self.dag.graph.nodes[node]["_stype"]
                if node in self.source_nodes:
                    value = self.source_node_to_ts_data_gen[node].get_value(row_idx=row_idx)
                    self.dag.graph.nodes[node]["value"] = self.strategy.tensorize_source(
                        value=value, _stype=node_stype
                    )
                else:
                    parent_nodes = list(self.dag.graph.predecessors(node))
                    propagation_agg = self.dag.graph.nodes[node]["propagation_agg"]

                    # directly add noise
                    noise_dist = self.dag.graph.nodes[node]["noise_dist"]
                    node_emb = (
                        noise_dist.sample(sample_shape=(self.strategy.mlp_emb_dim,)).squeeze()
                        / self.strategy.mlp_emb_dim
                    )

                    all_embs, all_weights = [], []
                    for parent_node in parent_nodes:
                        parent_attrs = self.dag.graph.nodes[parent_node]
                        encoder = self.dag.graph.edges[parent_node, node]["encoder"]
                        all_embs.append(encoder(parent_attrs["value"]).squeeze())
                        all_weights.append(
                            self.dag.graph.get_edge_data(parent_node, node)["weight"]
                        )
                    for foreign_row_embds in foreign_scms_row_embds:
                        w = 1 / len(foreign_row_embds) if propagation_agg == "sum" else 1.0
                        for foreign_row_embd in foreign_row_embds:
                            all_embs.append(foreign_row_embd)
                            all_weights.append(w)

                    if all_embs:
                        node_emb = node_emb + self._aggregate_embeddings(
                            embs=all_embs, weights=all_weights, mode=propagation_agg
                        )

                    decoder = self.dag.graph.nodes[node]["decoder"]
                    self.dag.graph.nodes[node]["value"] = decoder(node_emb)

    def generate_row(self, row_idx: int):
        row = {self.pkey_col: row_idx}
        foreign_row_idxs = []
        foreign_scms = []
        for fkey_col, foreign_table_name in self.fkey_col_to_pkey_table.items():
            foreign_scm = self.foreign_scm_info[foreign_table_name]
            foreign_scms.append(foreign_scm)
            bi_g = self.bi_fk_pk_graph_map[foreign_table_name]
            parent_node_name = list(bi_g.in_edges(f"b{row_idx}"))[0][0]
            foreign_row_idx = bi_g.nodes[parent_node_name]["node_idx"]
            foreign_row_idxs.append(foreign_row_idx)
            row[fkey_col] = foreign_row_idx

        self.propagate(
            row_idx=row_idx,
            foreign_row_idxs=foreign_row_idxs,
            foreign_scms=foreign_scms,
        )
        for node in sorted(self.col_nodes):
            col_name = self.dag.graph.nodes[node]["col_name"]
            row[col_name] = self.dag.graph.nodes[node]["value"].item()
        return row

    def initialize_bi_fk_pk_graph_map(self):
        self.bi_fk_pk_graph_map = {}
        for foreign_table_name, foreign_scm in tqdm(
            self.foreign_scm_info.items(),
            desc="Generating bi_fk_pk_graphs",
            leave=False,
        ):
            num_levels = self.scm_params.bi_hsbm_levels_choices.sample_uniform()
            hierarchy_a = [
                self.scm_params.bi_hsbm_clusters_per_level_choices.sample_uniform()
                for _ in range(num_levels)
            ]
            hierarchy_b = [
                self.scm_params.bi_hsbm_clusters_per_level_choices.sample_uniform()
                for _ in range(num_levels)
            ]
            bi_g = get_bipartite_hsbm(
                size_a=len(foreign_scm.df),
                size_b=self.num_rows,
                hierarchy_a=hierarchy_a,
                hierarchy_b=hierarchy_b,
            )
            self.bi_fk_pk_graph_map[foreign_table_name] = bi_g

    def _apply_categorical_quantization(self):
        for node in self.col_nodes:
            if self.dag.graph.nodes[node]["_stype"] != stype.categorical:
                continue
            col_name = self.dag.graph.nodes[node]["col_name"]
            num_categories = self.dag.graph.nodes[node]["num_categories"]
            col_values = torch.tensor(self.df[col_name].values, dtype=torch.float32)
            # Sample random boundaries from the data itself
            boundary_indices = torch.randint(0, len(col_values), (num_categories - 1,))
            boundaries = col_values[boundary_indices]
            # Count how many boundaries each value exceeds → class label in [0, num_categories-1]
            classes = (col_values.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)
            # randomly permute class labels to break ordinality
            permute_prob = self.scm_params.cat_label_permute_prob_choices.sample_uniform()
            if np.random.random() < permute_prob:
                perm = torch.randperm(num_categories)
                classes = perm[classes]
            # randomly reverse class labels
            reverse_prob = self.scm_params.cat_label_reverse_prob_choices.sample_uniform()
            if np.random.random() < reverse_prob:
                classes = num_categories - 1 - classes
            self.df[col_name] = classes.numpy()

    def generate_df(
        self,
        num_rows: int,
        table_type: TableType,
        min_timestamp: pd.Timestamp | None = None,
        max_timestamp: pd.Timestamp | None = None,
    ):
        self.num_rows = num_rows
        self.initialize_ts_data_gens(num_rows=num_rows, table_type=table_type)
        self.initialize_bi_fk_pk_graph_map()
        self.df = pd.DataFrame(
            [
                self.generate_row(row_idx=row_idx)
                for row_idx in tqdm(range(self.num_rows), desc="generating rows", leave=False)
            ]
        )
        self.strategy.post_generate(scm=self)
        if min_timestamp and max_timestamp:
            self.df["date"] = pd.date_range(
                start=min_timestamp, end=max_timestamp, periods=num_rows
            )
        return self.df

    def collate_feature_embeddings(self, row_idx: int, child_table_name: int):
        col_to_stype = {}
        col_to_num_categories = {}
        col_to_collation_encoder = {}
        for node in sorted(self.col_nodes):
            col_name = self.dag.graph.nodes[node]["col_name"]
            col_to_stype[col_name] = self.dag.graph.nodes[node]["_stype"]
            col_to_num_categories[col_name] = self.dag.graph.nodes[node]["num_categories"]
            col_to_collation_encoder[col_name] = self.dag.graph.nodes[node]["collation_encoders"][
                (self.table_name, child_table_name)
            ]
        row = self.df.iloc[row_idx].to_dict()
        row_embds = []
        # for p->f embedding propagation
        for col_name, value in row.items():
            if col_name not in col_to_stype:
                continue
            _stype = col_to_stype[col_name]
            value_tensor = self.strategy.tensorize_col(value=value, _stype=_stype)
            encoder = col_to_collation_encoder[col_name]
            row_embds.append(encoder(value_tensor).squeeze())
        return row_embds
