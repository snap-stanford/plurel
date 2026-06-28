"""Stage 1 (model env): run a trained RT / PluRel checkpoint on a task's TEST split and
emit raw predictions keyed by the ENTITY KEY (entity node idx + timestamp). NO relbench dep.

Output: .npz with [entity_node_idx, timestamp, pred]
  - binary_classification: pred = probability in [0,1]  (sigmoid applied)
  - regression:            pred = numeric value
  - entity_node_idx = f2p_nbr_idxs[is_targets][:, 0]  (the seed task-node's parent = the entity
    node, e.g. the driver). Stage2 subtracts the entity table's node_idx_offset to get the
    relbench entity id, then joins on (entity_id, timestamp).
  - timestamp = the `timestamps` batch field at the target position (unix seconds).

WHY NOT row_index = node_idx - offset: the Rust preprocessing's node order is the parquet's
PHYSICAL row order, which does NOT always equal a fresh pd.read_parquet() order (verified:
rel-f1/driver-top3 matched, but rel-f1/driver-dnf did NOT -> AUC 0.50 vs true 0.63). The
(entity, timestamp) key is order-independent and matches the relbench test table exactly
(verified driver-dnf: label-match 1.000 via this key).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# rt/model.py does `flex_attention = torch.compile(flex_attention)` at import. On large
# tasks the block-mask shapes recompile repeatedly and the compile can take >10 min and
# never reach a batch. With RT_EAGER=1 we no-op torch.compile so flex_attention runs eager
# (slower per batch but starts immediately and never recompiles) — set for big test sets.
if os.environ.get("RT_EAGER") == "1":
    _real_compile = torch.compile
    torch.compile = lambda fn=None, *a, **k: (fn if fn is not None else (lambda f: f))
else:
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256

from rt.data import RelationalDataset
from rt.model import RelationalTransformer

if os.environ.get("RT_EAGER") == "1":
    torch.compile = _real_compile  # restore after rt.model captured the no-op

REG_TABLES = {
    "item-sales", "user-ltv", "item-ltv", "post-votes", "site-success",
    "study-adverse", "user-attendance", "driver-position", "ad-ctr",
}


def _node_idx_offset(db, table, split):
    pre = Path(os.environ["HOME"]) / "scratch" / "pre" / db / "table_info.json"
    info = json.loads(pre.read_text())
    key = f"{table}:Db" if f"{table}:Db" in info else f"{table}:{split.capitalize()}"
    return info[key]["node_idx_offset"]


@torch.inference_mode()
def main(db, table, target, ckpt, out, *, split="test", batch_size=32, num_workers=8,
         max_bfs_width=256, embedding_model="all-MiniLM-L12-v2", d_text=384, seq_len=1024,
         num_blocks=12, d_model=256, num_heads=8, d_ff=1024, device="cuda"):
    task_type = "reg" if table in REG_TABLES else "clf"

    net = RelationalTransformer(num_blocks=num_blocks, d_model=d_model, d_text=d_text,
                                num_heads=num_heads, d_ff=d_ff)
    net.load_state_dict(torch.load(Path(ckpt).expanduser(), map_location="cpu"))
    net = net.to(device).to(torch.bfloat16).eval()

    # RT uses seq_len=; PluRel renamed it to ctx_len=. Support both.
    _ds_common = dict(
        tasks=[(db, table, target, split, [])], batch_size=batch_size,
        rank=0, world_size=1, max_bfs_width=max_bfs_width, embedding_model=embedding_model,
        d_text=d_text, seed=0)
    try:
        ds = RelationalDataset(seq_len=seq_len, **_ds_common)
    except TypeError:
        ds = RelationalDataset(ctx_len=seq_len, **_ds_common)
    ds.sampler.shuffle_py(0)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers,
                                         pin_memory=True, in_order=True)

    n_batches = len(loader)
    t0 = time.time()
    ents, tss, vals = [], [], []
    for bi, batch in enumerate(loader):
        tbs = batch.pop("true_batch_size")
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        batch["masks"][tbs:, :] = False
        batch["is_targets"][tbs:, :] = False
        batch["is_padding"][tbs:, :] = True
        _, yhat = net(batch)
        if bi == 0 or (bi + 1) % 50 == 0 or bi + 1 == n_batches:
            el = time.time() - t0
            print(f"  [{db}/{table}] batch {bi+1}/{n_batches} ({el:.0f}s, {el/(bi+1):.2f}s/batch)",
                  flush=True)
        is_t = batch["is_targets"].bool()
        if task_type == "clf":
            v = torch.sigmoid(yhat["boolean"][is_t].float()).flatten()
        else:
            v = yhat["number"][is_t].float().flatten()
        # entity = the seed task-node's first foreign->primary neighbour (the entity node).
        ent = batch["f2p_nbr_idxs"][:, :, 0][is_t].flatten()
        ts = batch["timestamps"][is_t].flatten()
        assert v.numel() == tbs and ent.numel() == tbs and ts.numel() == tbs
        ents.append(ent.cpu().numpy())
        tss.append(ts.cpu().numpy())
        vals.append(v.cpu().numpy())

    entity_node_idx = np.concatenate(ents).astype(np.int64)
    timestamp = np.concatenate(tss).astype(np.int64)
    pred = np.concatenate(vals).astype(np.float64)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, entity_node_idx=entity_node_idx, timestamp=timestamp, pred=pred,
             task_type=np.array(task_type))
    print(f"[stage1 OK] {db}/{table} ({task_type}): {len(pred)} preds -> {out}")
    print(f"  pred min={pred.min():.4g} max={pred.max():.4g} mean={pred.mean():.4g}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for a in ["db", "table", "target", "ckpt", "out"]:
        p.add_argument(f"--{a}", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    main(a.db, a.table, a.target, a.ckpt, a.out,
         batch_size=a.batch_size, num_workers=a.num_workers, device=a.device)
