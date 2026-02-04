<div align="center">
  <h1>PluRel</h1>
  <p>
Synthetic Data unlocks Scaling Laws for Relational Foundation Models
</p>
<img src="assets/scaling_law.png" alt="Scaling Law Plot"/>
</div>
<br>

## Framework Design

<img src="assets/plurel.png" alt="PluRel Logo"/>


## Setup

Setup the environment with [pixi](https://pixi.sh/latest/installation/)

```bash
# setup pixi environment
$ pixi install

# Compile and install the rust sampler
$ cd rustler && pixi run maturin develop --uv --release && cd ..

# Run pytest as a sanity check
$ pixi run pytest
```


## Synthesize Relational Data from Scratch

- The `SyntheticDataset` class can be used to create [relbench](https://github.com/snap-stanford/relbench) compatible dataset objects.
- It only requires a `seed` and a `Config` object that contains `database`, `scm` and `dag` level params for sampling. See example below.

```py
from plurel import SyntheticDataset, Config

# create relbench compatible dataset
dataset = SyntheticDataset(seed=0, config=Config())

# create database which can be cached via relbench APIs
db = dataset.make_db()
```

### Scalable Generation

We also provide a multiprocessing-based script to generate databases in parallel.

```bash
$ pixi run python scripts/synthetic_gen.py \
    --seed_offset 0 \
    --num_dbs 1000 \
    --num_proc 16
```

> [!NOTE]
> Checkout notebooks in `examples/` for synthesizing from SQL schemas


## Download Preprocessed Data

The preprocessed synthetic data is available on the Hugging Face Hub at [kvignesh1420/plurel](https://huggingface.co/datasets/kvignesh1420/plurel/tree/main).

1. Install the HuggingFace CLI (if not present)
```bash
pixi add huggingface_hub
```

2. Create the destination
```bash
mkdir -p ~/scratch/pre
```

3. Download the repository contents into ~/scratch/pre
```bash
pixi run hf download kvignesh1420/plurel \
    --repo-type dataset \
    --local-dir ~/scratch/pre
```

## Download Synthetic Pretrained Checkpoints

The synthetic pretrained model checkpoints are hosted on the Hugging Face Hub at [kvignesh1420/relational-transformer-plurel](https://huggingface.co/kvignesh1420/relational-transformer-plurel/tree/main).

```bash
$ mkdir -p ~/scratch/rt_hf_ckpts

$ pixi run hf download kvignesh1420/relational-transformer-plurel \
    --repo-type model \
    --local-dir ~/scratch/rt_hf_ckpts
```

The downloaded checkpoints will be listed as:

```bash
$ ls ~/scratch/rt_hf_ckpts

# model pretrained on a dataset of size 4B tokens curated from 1024 synthetic RDBs
synthetic-pretrain_rdb_1024_size_4b.pt

# model pretrained on a dataset of size 16B tokens curated from 512 synthetic RDBs
synthetic-pretrain_rdb_512_size_16b.pt
```

## Pretraining Experiments

- Baseline (real-world) pretraining on relbench datasets with a randomly initialized relational-transformer (RT) model.

```bash
$ pixi run torchrun --standalone --nproc_per_node=1 scripts/baseline_pretrain.py
```

- Synthetic pretraining on varying number of databases and dataset sizes with a randomly initialized RT model.

```bash
$ pixi run torchrun --standalone --nproc_per_node=1 scripts/synthetic_pretrain.py
```

- Continued pretraining on relbench datasets using the synthetic pretrained models. For faster experimentation, the downloaded models from huggingface (stored in `~/scratch/rt_hf_ckpts`) can be passed to the `load_ckpt_path` argument in the training script.

```bash
$ pixi run torchrun --standalone --nproc_per_node=1 scripts/cntd_pretrain.py
```