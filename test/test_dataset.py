import pytest
from pathlib import Path

from plurel.dataset import SyntheticDataset
from plurel.config import Config, DatabaseParams, Choices


@pytest.mark.parametrize("seed", list(range(10)))
def test_dataset(seed):
    config = Config(
        database_params=DatabaseParams(
            num_tables_choices=Choices(kind="range", value=[3, 5]),
            num_rows_entity_table_choices=Choices(kind="range", value=[40, 80]),
            num_rows_activity_table_choices=Choices(kind="range", value=[100, 200]),
        )
    )
    dataset = SyntheticDataset(seed=seed, config=config)
    db = dataset.make_db()
    assert db is not None


@pytest.mark.parametrize("seed", list(range(10)))
def test_dataset_with_sql_file(seed, schema_sql):
    config = Config(
        database_params=DatabaseParams(
            num_tables_choices=Choices(kind="range", value=[3, 5]),
            num_rows_entity_table_choices=Choices(kind="range", value=[40, 80]),
            num_rows_activity_table_choices=Choices(kind="range", value=[100, 200]),
        ),
        schema_file=schema_sql,
    )
    dataset = SyntheticDataset(seed=seed, config=config)
    db = dataset.make_db()
    assert db is not None
