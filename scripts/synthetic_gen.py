import argparse
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from plurel.config import Choices, Config, DAGParams, DatabaseParams, SCMParams
from plurel.dataset import SyntheticDataset
from plurel.utils import set_random_seed
from rt.tasks import DB_PREFIX

PRESETS = ("main", "v1.0.0")


def build_config(preset: str, cache_dir: Path) -> Config:
    """Build the generation Config.

    "main": current main defaults.

    "v1.0.0": reconstructs the *workload* of the v1.0.0 release on top of main's
    code — v1.0.0-equivalent database structure/complexity, with all of main's
    code-level speedups active. Each Choices added since v1.0.0 is pinned back to
    its v1.0.0 value:

      - scm_layout_choices    : the 5 layouts v1.0.0 sampled (no WattsStrogatz /
                                RandomCauchy, which main added)
      - activation_choices    : the original 6 (relu/elu/selu/silu/softsign/tanh)
      - mlp_num_layers        : fixed 2 (v1.0.0 MLP default)
      - mlp_weight_density    : 1.0 (v1.0.0 had no Bernoulli sparsity mask)
      - source_gen_type       : "ts" only (v1.0.0 had no uniform/gaussian/beta/mixed)
      - ts_ar_rho / value_scale: 0 / 1 (v1.0.0 TS had no AR noise, min/max in [0,1])
      - propagation_agg       : "sum" only (v1.0.0 aggregated parents by summation)
      - propagation_mode      : "type_eager" (v1.0.0 was type-aware throughout)
      - col_transform         : "identity" (v1.0.0 applied no post-hoc col transforms)
      - edge_weight_dist      : "gaussian" (v1.0.0 always used np.random.randn weights)

    Structural ranges (num tables/rows/cols, scm_col_node_perc, bi-HSBM, init fns)
    are unchanged between versions, so they keep their defaults. Node noise is the
    one v1.0.0 detail that can't be replicated exactly (v1.0.0 used 5 fixed Beta
    objects; main samples alpha/beta) — left at main defaults since it is a
    values-only difference.
    """
    if preset == "main":
        return Config(cache_dir=cache_dir)
    if preset == "v1.0.0":
        return Config(
            database_params=DatabaseParams(
                col_transform_choices=Choices(kind="set", value=["identity"]),
            ),
            scm_params=SCMParams(
                scm_layout_choices=Choices(
                    kind="set",
                    value=[
                        "ErdosRenyi",
                        "BarabasiAlbert",
                        "RandomTree",
                        "ReverseRandomTree",
                        "Layered",
                    ],
                ),
                activation_choices=Choices(
                    kind="set",
                    value=[F.relu, F.elu, F.selu, F.silu, F.softsign, F.tanh],
                ),
                mlp_num_layers_choices=Choices(kind="range", value=[2, 2]),
                mlp_weight_density_choices=Choices(kind="range", value=[1.0, 1.0]),
                source_gen_type_choices=Choices(kind="set", value=["ts"]),
                ts_ar_rho_choices=Choices(kind="range", value=[0.0, 0.0]),
                ts_value_scale_choices=Choices(kind="set", value=[1]),
                propagation_agg_choices=Choices(kind="set", value=["sum"]),
                propagation_mode_choices=Choices(kind="set", value=["type_eager"]),
            ),
            dag_params=DAGParams(
                edge_weight_dist_choices=Choices(kind="set", value=["gaussian"]),
            ),
            cache_dir=cache_dir,
        )
    raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")


def generate_plurel_db(
    seed: int,
    preprocess: bool = False,
    preset: str = "main",
    db_prefix: str = DB_PREFIX,
):
    torch.set_num_threads(1)
    set_random_seed(0)
    db_name = f"{db_prefix}-{seed}"
    print(f"Creating dataset: {db_name}")

    rustler_dir = Path("rustler").resolve()

    dataset = SyntheticDataset(
        seed=seed,
        config=build_config(
            preset=preset, cache_dir=Path(f"~/.cache/relbench/{db_name}").expanduser()
        ),
    )

    # generate and cache db in relbench format
    dataset.get_db()

    if preprocess:
        # pre-process
        subprocess.run(
            ["pixi", "run", "cargo", "run", "--release", "--", "pre", db_name],
            cwd=rustler_dir,
            check=True,
        )
        # embed text
        subprocess.run(
            ["pixi", "run", "python", "-m", "rt.embed", db_name],
            cwd=rustler_dir,
            check=True,
        )


def main(
    seed_offset: int,
    num_dbs: int,
    num_proc: int,
    preprocess: bool = False,
    preset: str = "main",
    db_prefix: str = DB_PREFIX,
):
    seeds = [idx + seed_offset for idx in range(num_dbs)]
    worker = partial(generate_plurel_db, preprocess=preprocess, preset=preset, db_prefix=db_prefix)

    with Pool(processes=num_proc) as p:
        list(
            tqdm(
                p.imap_unordered(worker, seeds),
                total=len(seeds),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset.")

    parser.add_argument(
        "--seed_offset",
        type=int,
        required=True,
        help="Seed offset for database generation. DBs will be named <db_prefix>-<seed>.",
    )

    parser.add_argument(
        "--num_dbs",
        type=int,
        required=True,
        help="Number of databases to generate.",
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use (default: number of CPU cores).",
    )

    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run preprocessing and embedding steps. Use --preprocess to enable.",
    )

    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="main",
        help=(
            "Config workload to generate with. 'main' uses current defaults; "
            "'v1.0.0' reconstructs the v1.0.0 release workload on main's "
            "optimized code (default: main)."
        ),
    )

    parser.add_argument(
        "--db_prefix",
        type=str,
        default=DB_PREFIX,
        help=f"DB name prefix; DBs are named <db_prefix>-<seed> (default: {DB_PREFIX}).",
    )

    args = parser.parse_args()

    main(
        seed_offset=args.seed_offset,
        num_dbs=args.num_dbs,
        num_proc=args.num_proc,
        preprocess=args.preprocess,
        preset=args.preset,
        db_prefix=args.db_prefix,
    )
