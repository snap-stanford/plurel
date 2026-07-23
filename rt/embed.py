import os

import numpy as np
import orjson
import strictfire
import torch
from ml_dtypes import bfloat16
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, batch_size, embedding_model, device_type):
        self.model = SentenceTransformer(
            f"sentence-transformers/{embedding_model}",
            device=device_type,
            model_kwargs={
                "dtype": torch.bfloat16 if device_type == "cuda" else torch.float32,
            },
        )
        self.model = torch.compile(self.model)
        self.batch_size = batch_size

    def __call__(self, text_list, device=None):
        return self.model.encode(
            text_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=device,
        )


def main(
    dataset_name,
    pre_dir="~/scratch/pre",
    device=None,
    batch_size=8192,
    embedding_model="all-MiniLM-L12-v2",
):
    pre_dir = os.path.expanduser(pre_dir)
    if device is None:
        # Get list of all available CUDA devices
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            device = [f"cuda:{i}" for i in range(num_devices)]
            print(f"Found {num_devices} CUDA device(s): {device}")
        else:
            device = "cpu"
            print("No CUDA devices available, using CPU")

    if isinstance(device, list):
        device_type = torch.device(device[0]).type
    else:
        device_type = torch.device(device).type

    text_path = f"{pre_dir}/{dataset_name}/text.json"
    with open(text_path) as f:
        raw = f.read()
    text_list = orjson.loads(raw)

    text_embedder = TextEmbedder(
        batch_size,
        embedding_model=embedding_model,
        device_type=device_type,
    )
    emb_list = text_embedder(text_list, device=device)

    emb_path = f"{pre_dir}/{dataset_name}/text_emb_{embedding_model}.bin"
    emb = np.stack(emb_list).astype(bfloat16)
    emb.tofile(emb_path)

    # record the embedding in meta.json so consumers can discover it
    meta_path = f"{pre_dir}/{dataset_name}/meta.json"
    if os.path.exists(meta_path):
        import json

        with open(meta_path) as f:
            meta = json.load(f)
        meta.setdefault("text_embeddings", {})[embedding_model] = {
            "file": f"text_emb_{embedding_model}.bin",
            "d_text": int(emb.shape[1]),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    strictfire.StrictFire(main)
