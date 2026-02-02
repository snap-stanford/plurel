from pathlib import Path
import argparse

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from plurel.dataset import SyntheticDataset
from plurel.config import Config
from plurel.utils import set_random_seed


def generate_rel_synthetic_db(seed: int):
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


def main(seed_offset: int, num_dbs: int, num_proc: int):
    seeds = [idx + seed_offset for idx in range(num_dbs)]

    with Pool(processes=num_proc) as p:
        list(
            tqdm(
                p.imap_unordered(generate_rel_synthetic_db, seeds),
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

    args = parser.parse_args()

    main(
        seed_offset=args.seed_offset,
        num_dbs=args.num_dbs,
        num_proc=args.num_proc,
    )
