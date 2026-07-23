"""Preprocess a relbench-format dataset dir into rustler's on-disk format.

rustler's preprocessor is self-describing: it reads a ``manifest.yaml`` next to
``db/<table>.parquet`` as the sole source of relational metadata (primary keys,
foreign keys, time columns). ``write_manifest`` produces that manifest from a
relbench ``Database`` object; ``preprocess_db`` runs the Rust preprocessor.
"""

import os
from pathlib import Path

import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings

maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))

import rustler


def write_manifest(db, db_name: str, dataset_dir, description: str = "") -> Path:
    """Write a relbench-3.0.0 ``manifest.yaml`` for a relbench ``Database``.

    ``dataset_dir`` is the dataset root containing ``db/<table>.parquet``.
    """
    import yaml

    dataset_dir = Path(dataset_dir).expanduser()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    for table_name, table in db.table_dict.items():
        tables[table_name] = {
            "pkey": table.pkey_col,
            "time_col": table.time_col,
            "fkeys": dict(table.fkey_col_to_pkey_table),
        }

    manifest = {
        "name": db_name,
        "manifest_version": 1,
        "description": description,
        "tables": tables,
    }

    manifest_path = dataset_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=True, default_flow_style=False)
    return manifest_path


def write_task_manifest(
    task_dir, entity_table: str, entity_col: str, target_col: str, task_type: str, time_col: str
) -> Path:
    """Write a ``manifest.yaml`` for a task dir with train/val/test parquets."""
    import yaml

    task_dir = Path(task_dir).expanduser()
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "entity_table": entity_table,
        "entity_col": entity_col,
        "target_col": target_col,
        "task_type": task_type,
        "time_col": time_col,
    }
    manifest_path = task_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=True, default_flow_style=False)
    return manifest_path


def preprocess_db(dataset_dir, out_dir, source: str | None = None, skip_tasks: bool = False):
    """Run the Rust preprocessor: ``<dataset_dir>`` -> ``<out_dir>/<db_name>/``.

    ``dataset_dir`` must contain ``manifest.yaml`` and ``db/*.parquet``
    (+ optional ``tasks/<task>/{train,val,test}.parquet`` with their own
    manifests).
    """
    dataset_dir = str(Path(dataset_dir).expanduser())
    out_dir = str(Path(out_dir).expanduser())
    os.makedirs(out_dir, exist_ok=True)
    rustler.preprocess(dataset_dir, out_dir, source=source, skip_tasks=skip_tasks)
