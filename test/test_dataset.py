import numpy as np
import pytest

from plurel.config import Choices, Config, DatabaseParams
from plurel.dataset import COLUMN_TRANSFORM_REGISTRY, SyntheticDataset

_TRANSFORM_INPUTS = {
    "standard_normal": np.random.default_rng(0).standard_normal(1000),
    "all_zeros": np.zeros(100),
    "large_positive": np.full(100, 1e6),
    "large_negative": np.full(100, -1e6),
    "mixed_sign": np.linspace(-1e4, 1e4, 1000),
}


@pytest.mark.parametrize("seed", list(range(100)))
def test_dataset(seed):
    config = Config(
        database_params=DatabaseParams(
            num_tables_choices=Choices(kind="range", value=[1, 5]),
            num_rows_entity_table_choices=Choices(kind="range", value=[40, 80]),
            num_rows_activity_table_choices=Choices(kind="range", value=[100, 200]),
        )
    )
    dataset = SyntheticDataset(seed=seed, config=config)
    db = dataset.make_db()
    assert db is not None


@pytest.mark.parametrize("seed", list(range(20)))
def test_dataset_with_sql_file(seed, schema_sql):
    config = Config(
        database_params=DatabaseParams(
            num_rows_entity_table_choices=Choices(kind="range", value=[40, 80]),
            num_rows_activity_table_choices=Choices(kind="range", value=[100, 200]),
        ),
        schema_file=schema_sql,
    )
    dataset = SyntheticDataset(seed=seed, config=config)
    db = dataset.make_db()
    assert db is not None


@pytest.mark.parametrize("transform_name", list(COLUMN_TRANSFORM_REGISTRY.keys()))
@pytest.mark.parametrize("input_name", list(_TRANSFORM_INPUTS.keys()))
def test_column_transform_finite(transform_name, input_name):
    x = _TRANSFORM_INPUTS[input_name]
    result = COLUMN_TRANSFORM_REGISTRY[transform_name](x)
    assert np.all(np.isfinite(result)), (
        f"transform '{transform_name}' on '{input_name}' produced non-finite values"
    )
