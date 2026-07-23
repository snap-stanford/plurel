"""Preprocess a relbench-3.0.0 dataset dir into rustler's on-disk format.

A relbench-3.0.0 dataset dir is self-describing: a ``manifest.yaml`` next to
``db/<table>.parquet`` is the sole source of relational metadata (primary keys,
foreign keys, time columns). ``write_manifest`` / ``write_task_manifest``
produce those manifests via ``relbench.manifest``; ``preprocess_db`` runs the
Rust preprocessor.
"""

import os
from pathlib import Path

import maturin_import_hook
from maturin_import_hook.settings import MaturinSettings

maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))

import rustler


def write_manifest(
    db,
    db_name: str,
    dataset_dir,
    val_timestamp,
    test_timestamp,
    description: str | None = None,
) -> Path:
    """Write a relbench-3.0.0 ``manifest.yaml`` for a relbench ``Database``.

    ``dataset_dir`` is the dataset root containing ``db/<table>.parquet``.
    """
    from relbench.manifest import DatasetManifest, TableSpec

    dataset_dir = Path(dataset_dir).expanduser()
    manifest = DatasetManifest(
        name=db_name,
        val_timestamp=str(val_timestamp),
        test_timestamp=str(test_timestamp),
        description=description,
        tables={
            table_name: TableSpec(
                pkey=table.pkey_col,
                time_col=table.time_col,
                fkeys=dict(table.fkey_col_to_pkey_table),
            )
            for table_name, table in db.table_dict.items()
        },
    )
    manifest_path = dataset_dir / "manifest.yaml"
    manifest.save(manifest_path)
    return manifest_path


def write_task_manifest(
    task_dir,
    name: str,
    entity_table: str,
    entity_col: str,
    target_col: str,
    task_type: str,
    time_col: str,
) -> Path:
    """Write a ``manifest.yaml`` for a task dir with train/val/test parquets.

    ``kind="external"``: the split parquets are served as-is.
    """
    from relbench.manifest import TaskManifest

    task_dir = Path(task_dir).expanduser()
    manifest = TaskManifest(
        name=name,
        kind="external",
        task_type=task_type,
        entity_table=entity_table,
        entity_col=entity_col,
        target_col=target_col,
        time_col=time_col,
    )
    manifest_path = task_dir / "manifest.yaml"
    manifest.save(manifest_path)
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
