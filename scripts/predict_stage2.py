"""Stage 2 (relbench-hf env): join stage-1 predictions to the relbench-hf leaderboard
test-table keys via (entity_id, timestamp) and write a leaderboard-format prediction CSV.

stage1 emits, per test prediction:
  entity_node_idx = the entity (e.g. driver) node's absolute index,
  timestamp       = unix seconds,
  pred.
We recover entity_id = entity_node_idx - <entity_table>:Db node_idx_offset (from
~/scratch/pre/<db>/table_info.json), then join (entity_id, timestamp) to the relbench-hf
core test table's (entity_col, time_col) to place each prediction on the right row.

This is robust to the preprocessing's parquet physical row order (the old node_idx-offset
positional mapping was wrong for some tasks, e.g. rel-f1/driver-dnf).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from relbench.base import TaskType
from relbench.load import load_task
from relbench.hf import CORE_REPO
from relbench.leaderboard import write_prediction_table

PRE = Path(os.environ.get("RT_PRE", Path(os.environ["HOME"]) / "scratch" / "pre"))
# relbench-1.x task parquets (train split target stats -> regression denormalization).
RELBENCH_CACHE = Path(os.environ.get("RELBENCH_CACHE",
                                     Path(os.environ["HOME"]) / ".cache" / "relbench"))


def _entity_offset(db, entity_table):
    info = json.loads((PRE / db / "table_info.json").read_text())
    key = f"{entity_table}:Db"
    if key not in info:
        raise KeyError(f"{key} not in {db}/table_info.json (keys: {list(info)[:8]}...)")
    return info[key]["node_idx_offset"]


def _train_target_stats(db, table, target_col):
    """Train-split target mean/std (ddof=1) — the normalization the preprocessing applied."""
    pq = RELBENCH_CACHE / db / "tasks" / table / "train.parquet"
    s = pd.read_parquet(pq)[target_col].astype(float)
    std = s.std(ddof=1)
    return float(s.mean()), float(std if std and std == std else 1.0)


def main(db, table, pred_npz, out_dir):
    task = load_task(f"{CORE_REPO}/{db}", table)
    ec, tc = task.entity_col, task.time_col
    entity_table = task.entity_table
    masked = task.get_table("test", mask_input_cols=True).df.reset_index(drop=True)
    n = len(masked)

    off = _entity_offset(db, entity_table)
    z = np.load(pred_npz, allow_pickle=True)
    entity_id = (z["entity_node_idx"].astype(np.int64) - off)
    ts = z["timestamp"].astype(np.int64)
    pvals = z["pred"].astype(np.float64)

    # Regression predictions are in normalized (z-score) space — denormalize to original scale
    # using the train-split target stats (the same normalization rustler/src/pre.rs applied).
    if task.task_type == TaskType.REGRESSION:
        mean, std = _train_target_stats(db, table, task.target_col)
        pvals = pvals * std + mean

    key_to_pred = {(int(e), int(t)): p for e, t, p in zip(entity_id, ts, pvals)}

    core_ts = (pd.to_datetime(masked[tc]).astype("int64") // 10**9).to_numpy()
    core_ent = masked[ec].astype(np.int64).to_numpy()
    full = np.array([key_to_pred.get((int(e), int(t)), np.nan)
                     for e, t in zip(core_ent, core_ts)], dtype=np.float64)

    covered = ~np.isnan(full)
    if not covered.all():
        print(f"[stage2 WARN] {int((~covered).sum())}/{n} core test rows unmatched "
              f"(entity_table={entity_table}, off={off})")

    out_csv = Path(out_dir) / f"{db}__{table}.csv"
    write_prediction_table(task, full, out_csv)
    print(f"[stage2 OK] {db}/{table}: matched {int(covered.sum())}/{n} -> {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    main(a.db, a.table, a.pred, a.out_dir)
