import argparse
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from plurel.config import Config
from plurel.dataset import SyntheticDataset
from plurel.utils import set_random_seed


def generate_rel_synthetic_db(seed: int, preprocess: bool = False):
    set_random_seed(0)
    db_name = f"rel-synthetic-{seed}"
    print(f"Creating dataset: {db_name}")

    rustler_dir = Path("rustler").resolve()

    dataset = SyntheticDataset(
        seed=seed,
        config=Config(cache_dir=Path(f"~/.cache/relbench/{db_name}").expanduser()),
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


def main(seed_offset: int, num_dbs: int, num_proc: int, preprocess: bool = False):
    seeds = [idx + seed_offset for idx in range(num_dbs)]
    worker = partial(generate_rel_synthetic_db, preprocess=preprocess)

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
        help="Seed offset for database generation. DBs will be named rel-synthetic-<seed>.",
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

    args = parser.parse_args()

    main(
        seed_offset=args.seed_offset,
        num_dbs=args.num_dbs,
        num_proc=args.num_proc,
        preprocess=args.preprocess,
    )
