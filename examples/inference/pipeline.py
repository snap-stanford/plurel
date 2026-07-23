"""Shared helpers for the inference walkthrough (steps 1-3).

All the reusable logic — converting a SQL database to RelBench format, defining the
task, preprocessing, running the model, and scoring — lives here so the numbered
step scripts stay short. You normally don't edit this file; edit ``config.py``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from relbench.base import Database, Table

# Architecture of the released 12-block checkpoint.
MODEL_DEFAULTS = dict(
    embedding_model="all-MiniLM-L12-v2",
    d_text=384,
    num_blocks=12,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    ctx_len=1024,
    max_bfs_width=256,
)


def task_kind(task_cfg: dict) -> str:
    """'clf' for binary_classification, 'reg' for regression."""
    t = task_cfg["task_type"]
    if t == "binary_classification":
        return "clf"
    if t == "regression":
        return "reg"
    raise ValueError(f"task_type must be 'binary_classification' or 'regression', got {t!r}")


# --------------------------------------------------------------------------- #
# Disk layout (the Rust sampler reads ~/scratch/relbench, writes ~/scratch/pre)
# --------------------------------------------------------------------------- #
def dataset_dir(db_name: str) -> Path:
    """<relbench cache>/<db_name>, ensuring the ~/scratch layout exists."""
    scratch = Path(os.environ["HOME"]) / "scratch"
    (scratch / "pre").mkdir(parents=True, exist_ok=True)
    link = scratch / "relbench"
    if not link.exists():
        import pooch

        cache = Path(pooch.os_cache("relbench"))
        cache.mkdir(parents=True, exist_ok=True)
        link.symlink_to(cache)
    return link / db_name


# --------------------------------------------------------------------------- #
# Step 1: SQL database -> RelBench Database
# --------------------------------------------------------------------------- #
def read_tables(uri: str, table_names) -> dict[str, pd.DataFrame]:
    """Read the named tables into DataFrames.

    A ``.duckdb`` / ``.db`` file path is read natively; anything else is treated
    as a SQLAlchemy URI (Postgres: install ``psycopg2-binary``, MySQL: ``pymysql``).
    """
    if uri.endswith((".duckdb", ".db")):
        import duckdb

        if not Path(uri).exists():
            raise FileNotFoundError(f"DuckDB file not found: {uri}")
        con = duckdb.connect(uri, read_only=True)
        try:
            return {t: con.execute(f'SELECT * FROM "{t}"').df() for t in table_names}
        finally:
            con.close()

    from sqlalchemy import create_engine

    engine = create_engine(uri)
    try:
        return {t: pd.read_sql(f'SELECT * FROM "{t}"', engine) for t in table_names}
    finally:
        engine.dispose()


def build_database(tables_cfg: dict, raw: dict[str, pd.DataFrame]) -> Database:
    """Attach primary-key / foreign-key / time metadata to the raw tables."""
    table_dict = {}
    for name, cfg in tables_cfg.items():
        df = raw[name]
        time_col = cfg.get("time_col")
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
        table_dict[name] = Table(
            df=df,
            fkey_col_to_pkey_table=dict(cfg.get("fkeys") or {}),
            pkey_col=cfg.get("pkey"),
            time_col=time_col,
        )
    return Database(table_dict)


def save_database(db_name: str, db: Database) -> Path:
    out = dataset_dir(db_name) / "db"
    db.save(out)
    # the Rust preprocessor reads relational metadata from manifest.yaml
    from rt.preprocess import write_manifest

    write_manifest(db, db_name, dataset_dir(db_name))
    return out


# --------------------------------------------------------------------------- #
# Step 2: labeled rows -> RelBench task tables
# --------------------------------------------------------------------------- #
def build_task_tables(task_cfg: dict) -> dict[str, Table]:
    """One RelBench Table per split, from the labeled parquet/csv files.

    Each task table's entity column is a foreign key into the entity table.
    """
    task_kind(task_cfg)  # validate task_type early
    time_col = task_cfg["time_col"]
    splits = task_cfg["splits"]
    if "test" not in splits:
        raise ValueError("task 'splits' must include a 'test' entry (the labeled rows to score).")
    tables = {}
    for split, path in splits.items():
        p = Path(path)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        df[time_col] = pd.to_datetime(df[time_col])
        tables[split] = Table(
            df=df,
            fkey_col_to_pkey_table={task_cfg["entity_col"]: task_cfg["entity_table"]},
            pkey_col=None,
            time_col=time_col,
        )
    return tables


def save_task_tables(
    db_name: str, task_name: str, split_tables: dict[str, Table], task_cfg: dict | None = None
) -> Path:
    out = dataset_dir(db_name) / "tasks" / task_name
    for split, table in split_tables.items():
        table.save(out / f"{split}.parquet")
    if task_cfg is not None:
        from rt.preprocess import write_task_manifest

        write_task_manifest(
            out,
            entity_table=task_cfg["entity_table"],
            entity_col=task_cfg["entity_col"],
            target_col=task_cfg["target_col"],
            task_type=task_cfg["task_type"],
            time_col=task_cfg["time_col"],
        )
    return out


# --------------------------------------------------------------------------- #
# Step 3: preprocess -> infer -> evaluate
# --------------------------------------------------------------------------- #
def resolve_checkpoint(local_path, hf_repo, hf_filename) -> tuple[Path, dict]:
    """Return (checkpoint path, model config), downloading from the HF Hub if needed.

    The repo's config.json carries the architecture the checkpoint was trained with;
    MODEL_DEFAULTS is the fallback for local checkpoints without one.
    """
    if local_path:
        p = Path(local_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p, dict(MODEL_DEFAULTS)
    from huggingface_hub import hf_hub_download

    print(f"   downloading checkpoint {hf_repo}/{hf_filename} from the HF Hub...")
    ckpt = Path(hf_hub_download(repo_id=hf_repo, filename=hf_filename, repo_type="model"))
    cfg = dict(MODEL_DEFAULTS)
    cfg.update(
        json.loads(Path(hf_hub_download(repo_id=hf_repo, filename="config.json")).read_text())
    )
    return ckpt, cfg


def ensure_preprocessed(db_name: str, embedding_model: str) -> None:
    """Run the Rust preprocessor + text embedding unless already cached."""
    pre_root = Path(os.environ["HOME"]) / "scratch" / "pre"
    pre = pre_root / db_name
    emb = pre / f"text_emb_{embedding_model}.bin"
    if (pre / "table_info.json").exists() and emb.exists():
        print(f"   using cached preprocessed data at {pre}")
        return
    if not (pre / "table_info.json").exists():
        print(f"   preprocessing '{db_name}' with the Rust sampler...")
        from rt.preprocess import preprocess_db

        preprocess_db(dataset_dir(db_name), pre_root)
    if not emb.exists():
        print(f"   embedding text columns for '{db_name}'...")
        from rt.embed import main as embed_main

        embed_main(db_name, pre_dir=str(pre_root), embedding_model=embedding_model)


@torch.inference_mode()
def run_inference(config, ckpt: Path, device: str, batch_size: int, num_workers: int, cfg=None):
    """Run the checkpoint over the task's test split.

    Returns (entity_node_idx, timestamp, prediction) arrays — one entry per test
    row, keyed by the entity node and time so evaluate() can join them back.
    """
    from rt.data import RelationalDataset

    # rt.model compiles flex_attention at import, which hangs on CPU. RT_EAGER=1
    # (set for --device cpu) no-ops torch.compile around that import.
    if os.environ.get("RT_EAGER") == "1":
        _real = torch.compile
        torch.compile = lambda fn=None, *a, **k: fn if fn is not None else (lambda f: f)
        from rt.model import RelationalTransformer

        torch.compile = _real
    else:
        from rt.model import RelationalTransformer

    cfg = cfg or dict(MODEL_DEFAULTS)
    net = RelationalTransformer(
        num_blocks=cfg["num_blocks"],
        d_model=cfg["d_model"],
        d_text=cfg["d_text"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    )
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net = net.to(device).to(torch.bfloat16).eval()

    task = config.TASK
    ds = RelationalDataset(
        tasks=[(config.DB_NAME, task["name"], task["target_col"], "test", [])],
        batch_size=batch_size,
        rank=0,
        world_size=1,
        ctx_len=cfg["ctx_len"],
        max_bfs_width=cfg["max_bfs_width"],
        embedding_model=cfg["embedding_model"],
        d_text=cfg["d_text"],
        seed=0,
        pre_dir=str(Path(os.environ["HOME"]) / "scratch" / "pre"),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers, pin_memory=(device == "cuda"), in_order=True
    )

    kind = task_kind(task)
    n_batches = len(loader)
    t0 = time.time()
    ents, tss, vals = [], [], []
    for bi, batch in enumerate(loader):
        # phantom slots in the final batch carry no target cells, so the
        # target-indexed gathers below skip them automatically
        batch.pop("batch_mask")
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        _, yhat = net(batch)
        if bi == 0 or (bi + 1) % 50 == 0 or bi + 1 == n_batches:
            print(f"   batch {bi + 1}/{n_batches} ({time.time() - t0:.0f}s)", flush=True)
        is_t = batch["is_targets"].bool()
        if kind == "clf":
            v = torch.sigmoid(yhat["boolean"][is_t].float()).flatten()
        else:
            v = yhat["number"][is_t].float().flatten()
        # the target node's first foreign->primary neighbour is the entity node
        ents.append(batch["f2p_nbr_idxs"][:, :, 0][is_t].flatten().cpu().numpy())
        tss.append(batch["timestamps"][is_t].flatten().cpu().numpy())
        vals.append(v.cpu().numpy())

    return (
        np.concatenate(ents).astype(np.int64),
        np.concatenate(tss).astype(np.int64),
        np.concatenate(vals).astype(np.float64),
    )


def evaluate(config, entity_node_idx, timestamp, pred) -> float:
    """Join predictions back to the labeled test rows and report the metric.

    Predictions are joined by (entity id, timestamp) — this is order-independent
    and robust to how the preprocessing shuffled the rows.
    """
    from sklearn.metrics import mean_absolute_error, roc_auc_score

    task = config.TASK
    kind = task_kind(task)
    test_df = pd.read_parquet(dataset_dir(config.DB_NAME) / "tasks" / task["name"] / "test.parquet")

    # entity node idx -> entity id (undo the node offset from preprocessing)
    info_path = Path(os.environ["HOME"]) / "scratch" / "pre" / config.DB_NAME / "table_info.json"
    offset = json.loads(info_path.read_text())[f"{task['entity_table']}:Db"]["node_idx_offset"]
    entity_id = entity_node_idx - offset

    # regression predictions are z-score normalized; undo with train target stats
    if kind == "reg" and "train" in task["splits"]:
        train_df = pd.read_parquet(
            dataset_dir(config.DB_NAME) / "tasks" / task["name"] / "train.parquet"
        )
        t = train_df[task["target_col"]].astype(float)
        pred = pred * (t.std(ddof=1) or 1.0) + t.mean()

    key_to_pred = {(int(e), int(t)): p for e, t, p in zip(entity_id, timestamp, pred)}
    test_ts = (pd.to_datetime(test_df[task["time_col"]]).astype("int64") // 10**9).to_numpy()
    test_ent = test_df[task["entity_col"]].astype(np.int64).to_numpy()
    full = np.array([key_to_pred.get((int(e), int(t)), np.nan) for e, t in zip(test_ent, test_ts)])
    covered = ~np.isnan(full)
    if not covered.all():
        print(f"   [warn] {int((~covered).sum())}/{len(full)} test rows had no prediction")

    y = test_df[task["target_col"]].astype(float).to_numpy()[covered]
    if kind == "clf":
        metric, name = roc_auc_score((y > 0).astype(int), full[covered]), "AUROC"
    else:
        metric, name = mean_absolute_error(y, full[covered]), "MAE"
    print(
        f"\n[result] {config.DB_NAME}/{task['name']}   {name} = {metric:.4f}   (n={int(covered.sum())})"
    )
    return metric
