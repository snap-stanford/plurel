import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_frame import stype


@dataclass(frozen=True)
class Choices:
    """
    A sampling utility for configuration parameters.

    Supports two kinds of sampling:
    - "range": Uniform sampling between [value[0], value[1]]
    - "set": Random selection from a discrete set of values

    Args:
        kind: Either "range" or "set"
        value: For "range", a list of [min, max]. For "set", a list of discrete choices.
    """

    kind: Literal["range", "set"]
    value: list[Any] | Any

    def __post_init__(self):
        if self.kind not in ("range", "set"):
            raise ValueError(
                f"Invalid kind of choices '{self.kind}'. Must be either 'range' or 'set'."
            )
        if self.kind == "range":
            if not isinstance(self.value, list):
                raise ValueError(f"'value' of type '{type(self.value)}' is not supported")
            if len(self.value) != 2:
                raise ValueError("'value' must have two elements to support 'range' based sampling")

    def sample_uniform(self, size: int | None = None, replace: bool = False):
        if self.kind == "range":
            if type(self.value[0]) == int:
                return np.random.randint(low=self.value[0], high=self.value[1] + 1, size=size)
            elif type(self.value[0]) == float:
                return np.random.uniform(low=self.value[0], high=self.value[1], size=size)
            else:
                raise ValueError(
                    f"Unsupported data type: {type(self.value[0])} for uniform sampling. The 'value' elements should be either int/float."
                )
        elif self.kind == "set":
            return np.random.choice(self.value, size=size, replace=replace)

    def sample_pl(self, exponent: float = 1, size: int | None = None, replace: bool = False):
        if self.kind == "range":
            if type(self.value[0]) == int:
                low = self.value[0]
                high = self.value[1] + 1
                choices = np.arange(low, high)
                probs = 1 / (choices**exponent)
                probs /= probs.sum()
                return np.random.choice(choices, size=size, replace=replace, p=probs)
            else:
                raise ValueError(
                    f"Unsupported data type: {type(self.value[0])} for power-law sampling. The 'value' elements should be either int/float."
                )
        elif self.kind == "set":
            raise ValueError("'set' kind not supported for power-law sampling")


class RandomFunctionActivation:
    """Composite sine-wave activation: re-randomized on every call."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = np.random.randint(3, 8)
        amplitudes = torch.randn(n)
        freqs = torch.exp(torch.FloatTensor(n).uniform_(math.log(0.1), math.log(10.0)))
        phases = torch.FloatTensor(n).uniform_(0, 2 * math.pi)
        return (amplitudes * torch.sin(freqs * x.unsqueeze(-1) + phases)).sum(dim=-1)


@dataclass(frozen=True)
class DatabaseParams:
    """Parameters controlling database structure and table generation."""

    table_layout_choices: Choices = Choices(
        kind="set",
        value=["BarabasiAlbert", "ReverseRandomTree", "WattsStrogatz"],
    )
    num_tables_choices: Choices = Choices(kind="range", value=[3, 20])
    num_rows_entity_table_choices: Choices = Choices(kind="range", value=[500, 1000])
    num_rows_activity_table_choices: Choices = Choices(kind="range", value=[2000, 5000])
    num_cols_choices: Choices = Choices(kind="range", value=[3, 40])
    min_timestamp: pd.Timestamp = pd.Timestamp("1990-01-01")
    max_timestamp: pd.Timestamp = pd.Timestamp("2025-01-01")
    column_nan_perc_choices: Choices = Choices(kind="range", value=[0.01, 0.1])
    col_transform_choices: Choices = Choices(
        kind="set",
        value=["identity", "rank_uniform", "log", "sqrt", "standardize"],
    )


@dataclass(frozen=True)
class SCMParams:
    """Parameters controlling Structural Causal Models (SCMs)."""

    scm_layout_choices: Choices = Choices(
        kind="set",
        value=[
            "ErdosRenyi",
            "BarabasiAlbert",
            "RandomTree",
            "ReverseRandomTree",
            "Layered",
            "WattsStrogatz",
            "RandomCauchy",
        ],
    )
    scm_col_node_perc_choices: Choices = Choices(kind="range", value=[0.3, 0.9])
    num_categories_choices: Choices = Choices(kind="range", value=[2, 10])
    col_stype_choices: Choices = Choices(kind="set", value=[stype.categorical, stype.numerical])
    initialization_choices: Choices = Choices(
        kind="set",
        value=[
            torch.nn.init.kaiming_normal_,
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.xavier_normal_,
            torch.nn.init.xavier_uniform_,
            torch.nn.init.trunc_normal_,
            lambda x: torch.nn.init.sparse_(x, sparsity=0.5),
        ],
    )
    activation_choices: Choices = Choices(
        kind="set",
        value=[
            F.relu,
            F.elu,
            F.selu,
            F.silu,
            F.softsign,
            F.tanh,
            F.sigmoid,
            F.hardswish,
            F.mish,
            F.leaky_relu,
            F.gelu,
            F.softplus,
            F.hardsigmoid,
            F.hardtanh,
            F.celu,
            F.tanhshrink,
            F.rrelu,
            F.logsigmoid,
            F.softshrink,
            F.hardshrink,
            lambda x: x,
            lambda x: x.abs(),
            torch.sin,
            torch.cos,
            torch.sign,
            lambda x: x.clamp(-1, 1),
            lambda x: torch.log1p(x.abs()) * x.sign(),
            lambda x: (torch.sqrt(x**2 + 1) - 1) / 2 + x,
            RandomFunctionActivation(),
        ],
    )
    node_noise_alpha_choices: Choices = Choices(kind="range", value=[0.5, 5.0])
    node_noise_beta_choices: Choices = Choices(kind="range", value=[0.5, 5.0])

    bi_hsbm_levels_choices: Choices = Choices(kind="range", value=[1, 5])
    bi_hsbm_clusters_per_level_choices: Choices = Choices(kind="range", value=[1, 3])

    ts_trend_alpha_choices: Choices = Choices(kind="range", value=[0.0, 2.0])
    activity_table_ts_trend_scale_choices: Choices = Choices(kind="set", value=[-1, 1])
    entity_table_ts_trend_scale: float = 0.0

    ts_cycle_freq_perc_choices: Choices = Choices(kind="set", value=[i / 10 for i in range(1, 11)])
    activity_table_ts_cycle_scale_choices: Choices = Choices(kind="set", value=[-1, 1])
    entity_table_ts_cycle_scale: float = 0.0

    activity_table_ts_noise_scale: float = 0.05
    entity_table_ts_noise_scale: float = 1

    ts_ar_rho_choices: Choices = Choices(kind="range", value=[0.0, 0.9])
    ts_value_scale_choices: Choices = Choices(kind="set", value=[0.01, 0.1, 1, 10, 100])

    propagation_agg_choices: Choices = Choices(
        kind="set",
        value=["sum", "max", "product", "logexp"],
    )

    mlp_in_dim: int = 1
    mlp_out_dim: int = 1
    mlp_emb_dim: int = 32
    mlp_num_layers_choices: Choices = Choices(kind="range", value=[2, 4])
    mlp_weight_density_choices: Choices = Choices(kind="range", value=[0.3, 1.0])
    source_gen_type_choices: Choices = Choices(
        kind="set", value=["ts", "uniform", "gaussian", "beta", "mixed"]
    )
    source_beta_alpha_choices: Choices = Choices(kind="range", value=[0.5, 5.0])
    source_beta_beta_choices: Choices = Choices(kind="range", value=[0.5, 5.0])
    propagation_mode_choices: Choices = Choices(kind="set", value=["type_eager", "type_lazy"])
    cat_label_permute_prob_choices: Choices = Choices(kind="range", value=[0.3, 1.0])
    cat_label_reverse_prob_choices: Choices = Choices(kind="range", value=[0.2, 0.8])


@dataclass(frozen=True)
class DAGParams:
    """Parameters controlling DAG structure generation."""

    ba_sink_edge_dropout: float = 0.4
    ba_flip_leaf_edge_prob: float = 0.0
    ba_max_in_degree: int = 4
    ba_m: int = 2
    er_p_choices: Choices = Choices(kind="range", value=[0.3, 0.8])
    ws_rewire_p_choices: Choices = Choices(kind="range", value=[0.1, 0.3])
    layered_depth_choices: Choices = Choices(kind="range", value=[2, 8])
    layered_edge_dropout_p: float = 0.1
    edge_weight_dist_choices: Choices = Choices(
        kind="set", value=["gaussian", "lognormal", "cauchy", "uniform"]
    )


@dataclass(frozen=True)
class Config:
    """
    Top-level configuration for synthetic database generation.

    Args:
        database_params: Parameters for database structure and tables.
        scm_params: Parameters for Structural Causal Models.
        dag_params: Parameters for DAG structure.
        val_split: Fraction of rows for validation split.
        test_split: Fraction of rows for test split.
        cache_dir: Optional directory for caching generated databases.
        schema_file: Optional SQL schema file for schema-based generation.
    """

    database_params: DatabaseParams = field(default_factory=DatabaseParams)
    scm_params: SCMParams = field(default_factory=SCMParams)
    dag_params: DAGParams = field(default_factory=DAGParams)
    val_split: float = 0.8
    test_split: float = 0.9
    cache_dir: str | None = None
    schema_file: str | None = None
