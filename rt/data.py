import json
import math
import os
from functools import cache
from pathlib import Path

import maturin_import_hook
import ml_dtypes  # noqa: F401
import numpy as np
import torch
from maturin_import_hook.settings import MaturinSettings
from torch.utils.data import Dataset

maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))


from rustler import Sampler

# Default preprocessed-data sources on the Hugging Face Hub. A `pre_dir` may be
# a local path or a Hub repo spec `org/repo[/subdir]`; only the files needed
# for the requested databases are downloaded and cached.
RELBENCH_PRE = "stanford-star/relbench-preprocessed"
PLUREL_PRE = "stanford-star/plurel-preprocessed"

MAX_F2P_NBRS = 5  # see fly.rs

# Per-dataset files needed by the sampler (plus text_emb_<model>.bin).
CORE_FILES = (
    "meta.json",
    "nodes.rkyv",
    "offsets.rkyv",
    "p2f_adj.rkyv",
    "table_info.json",
    "column_index.json",
)


def resolve_pre_dir(pre_dir: str, db_names, embedding_model: str) -> str:
    """Return a local root directory containing ``<db>/`` subfolders.

    If ``pre_dir`` is an existing local path it is returned as-is. Otherwise it
    is treated as a Hub ``org/repo[/subdir]`` dataset spec and only the files
    needed for ``db_names`` (+ the chosen ``embedding_model``) are downloaded
    into the HF cache.
    """
    p = Path(pre_dir).expanduser()
    if p.exists():
        return str(p)

    from huggingface_hub import snapshot_download

    parts = str(pre_dir).strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"{pre_dir!r} is neither an existing local path nor a Hub 'org/name[/subdir]' spec."
        )
    repo_id, subdir = "/".join(parts[:2]), "/".join(parts[2:])
    prefix = f"{subdir}/" if subdir else ""
    patterns = []
    for db in dict.fromkeys(db_names):  # dedup, preserve order
        patterns += [f"{prefix}{db}/{f}" for f in CORE_FILES]
        patterns.append(f"{prefix}{db}/text_emb_{embedding_model}.bin")

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
    )
    return str(Path(local) / subdir) if subdir else str(local)


@cache
def _load_column_index(db_name: str, pre_dir: str) -> dict:
    column_index_path = os.path.join(pre_dir, db_name, "column_index.json")
    with open(column_index_path) as f:
        return json.load(f)


def get_column_index(column_name: str, table_name: str, db_name: str, pre_dir: str) -> int:
    """
    Get the index of a column in the text embeddings for a given dataset.
    """
    column_index = _load_column_index(db_name, pre_dir)
    target = f"{column_name} of {table_name}"

    if target not in column_index:
        raise ValueError(f'Column "{target}" not found in {pre_dir}/{db_name}/column_index.json.')

    return column_index[target]


class RelationalDataset(Dataset):
    def __init__(
        self,
        tasks,
        batch_size,
        rank,
        world_size,
        ctx_len,
        max_bfs_width,
        embedding_model,
        d_text,
        seed,
        pre_dir=RELBENCH_PRE,
        train=False,
        num_walks=10_000,
        walk_length=20,
        mask_prob_max=0.0,
        items_per_task=-1,
    ):
        # `pre_dir` may be a local path or a HuggingFace Hub repo spec.
        pre_dir = resolve_pre_dir(pre_dir, [t[0] for t in tasks], embedding_model)

        dataset_tuples = []
        target_column_indices = []
        drop_column_indices = []

        for db_name, table_name, target_column, split, columns_to_drop in tasks:
            if split == "train":
                split = "Train"
            elif split == "val":
                split = "Val"
            elif split == "test":
                split = "Test"

            table_info_path = f"{pre_dir}/{db_name}/table_info.json"
            with open(table_info_path) as f:
                table_info = json.load(f)

            table_info_key = (
                f"{table_name}:Db" if f"{table_name}:Db" in table_info else f"{table_name}:{split}"
            )
            info = table_info[table_info_key]
            node_idx_offset = info["node_idx_offset"]
            num_nodes = info["num_nodes"]

            target_idx = get_column_index(target_column, table_name, db_name, pre_dir)
            target_column_indices.append(target_idx)

            drop_indices = [
                get_column_index(col, table_name, db_name, pre_dir) for col in columns_to_drop
            ]
            drop_column_indices.append(drop_indices)

            dataset_tuples.append((db_name, table_name, node_idx_offset, num_nodes))

        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        self.sampler = Sampler(
            dataset_tuples=dataset_tuples,
            global_rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_ctx_sizes=[ctx_len],
            bfs_widths=[max_bfs_width],
            num_walks=num_walks,
            walk_length=walk_length,
            prefer_latest=[True],
            mask_prob_max=mask_prob_max,
            embedding_model=embedding_model,
            pre_dir=pre_dir,
            d_text=d_text,
            shuffle_seed=seed,
            context_seed=seed,
            target_columns=target_column_indices,
            columns_to_drop=drop_column_indices,
            items_per_task=items_per_task,
            quiet=False,
            ignore_data_errors=False,
            num_prev_skipped=0,
            skip_text_cols=False,
            mmap_populate=False,
            balance_labels=[False],
            timeout_per_item=10.0,
            ablate_schema_semantics=False,
            vector_db_path=None,
            train_only_fallback=False,
        )

        self.batch_size = batch_size
        self.world_size = world_size
        self.ctx_len = ctx_len
        self.d_text = d_text
        self.train = train
        self._worker_configured = False

    def __len__(self):
        return math.ceil(self.sampler.num_items / (self.batch_size * self.world_size))

    def __getitem__(self, batch_idx):
        if self.train:
            # Train mode: rustler samples items randomly and auto-advances its
            # step; shard the step stream across dataloader workers.
            if not self._worker_configured:
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    self.sampler.set_step_py(worker_info.id)
                    self.sampler.set_stride_py(worker_info.num_workers)
                self._worker_configured = True
            tup = self.sampler.batch_py(None, self.batch_size, self.ctx_len)
        else:
            # Eval mode: deterministic pass over items, sharded across ranks.
            # Overshooting slots in the last batch come back with
            # batch_mask[i] = False.
            tup = self.sampler.batch_py(batch_idx, self.batch_size, self.ctx_len)
        return self._process_batch(tup)

    def _process_batch(self, tup):
        out = dict(tup)
        seq_len = out.pop("seq_len")

        for k, v in out.items():
            if k in [
                "number_values",
                "datetime_values",
                "text_values",
                "col_name_values",
                "boolean_values",
            ]:
                out[k] = torch.from_numpy(v.view(np.float16)).view(torch.bfloat16)
            else:
                out[k] = torch.from_numpy(v)

        out["node_idxs"] = out["node_idxs"].view(-1, seq_len)
        out["sem_types"] = out["sem_types"].view(-1, seq_len)
        out["is_targets"] = out["is_targets"].view(-1, seq_len)
        out["is_task_nodes"] = out["is_task_nodes"].view(-1, seq_len)
        out["is_padding"] = out["is_padding"].view(-1, seq_len)
        out["table_name_idxs"] = out["table_name_idxs"].view(-1, seq_len)
        out["col_name_idxs"] = out["col_name_idxs"].view(-1, seq_len)
        out["class_value_idxs"] = out["class_value_idxs"].view(-1, seq_len)
        out["timestamps"] = out["timestamps"].view(-1, seq_len)
        out["seed_node_idxs"] = out["seed_node_idxs"].view(-1, seq_len)
        out["bfs_depths"] = out["bfs_depths"].view(-1, seq_len)

        out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, seq_len, MAX_F2P_NBRS)
        out["number_values"] = out["number_values"].view(-1, seq_len, 1)
        out["datetime_values"] = out["datetime_values"].view(-1, seq_len, 1)
        out["boolean_values"] = out["boolean_values"].view(-1, seq_len, 1).bfloat16()
        out["text_values"] = out["text_values"].view(-1, seq_len, self.d_text)
        out["col_name_values"] = out["col_name_values"].view(-1, seq_len, self.d_text)

        return out
