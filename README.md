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
$ pixi install
$ pixi run pip install -e .
```

Run pytest as a sanity check

```bash
$ pixi run pytest
```


### Synthesize relational data from scratch

- The `SyntheticDataset` class can be used to create [relbench](https://github.com/snap-stanford/relbench) compatible dataset objects.
- It only requires a `seed` and a `Config` object that contains `database`, `scm` and `dag` level params for sampling. See example below.

```py
from plurel import SyntheticDataset, Config

# create relbench compatible dataset
dataset = SyntheticDataset(seed=0, config=Config())

# create database which can be cached via relbench APIs
db = dataset.make_db()
```

#### Scalable generation

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

1. Install the CLI
```bash
pip install -U huggingface_hub
```

2. Create the destination
```bash
mkdir -p ~/scratch/pre
```

3. Download the repository contents into ~/scratch/pre
```bash
huggingface-cli download kvignesh1420/plurel \
  --repo-type dataset \
  --local-dir ~/scratch/pre \
  --local-dir-use-symlinks False
```

## Download Synthetic Pretrained Checkpoints

The synthetic pretrained model checkpoints are hosted on the Hugging Face Hub at [kvignesh1420/relational-transformer-plurel](https://huggingface.co/kvignesh1420/relational-transformer-plurel/tree/main). The downloaded models can be passed to the `load_ckpt_path` argument of the training scripts.

1. Install the Hugging Face CLI (if not already installed):
```bash
pip install -U huggingface_hub
```

2. Download checkpoints:
```bash
mkdir -p ~/scratch/rt_ckpts
huggingface-cli download kvignesh1420/relational-transformer-plurel \
  --repo-type model \
  --local-dir ~/scratch/rt_ckpts \
  --local-dir-use-symlinks False
```