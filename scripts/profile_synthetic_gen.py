"""Profile latency and peak memory of synthetic database generation.

Generates one database per seed (default: 10 seeds) with a fixed table count
and reports per-seed wall time and peak resident-set-size growth, plus
mean/std/min/max across seeds.

Example
-------
$ pixi run python scripts/profile_synthetic_gen.py --num_tables 8 --num_seeds 10
$ pixi run python scripts/profile_synthetic_gen.py --num_tables 20 --num_warmup 2
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import threading
import time

# Silence inner tqdm progress bars before plurel imports them.
os.environ.setdefault("TQDM_DISABLE", "1")

import psutil
import torch

from plurel import Choices, Config, DatabaseParams, SCMParams
from plurel.dataset import SyntheticDataset


def _fixed(value):
    """Wrap a single value as a degenerate `Choices` so sampling is deterministic."""
    return Choices(kind="set", value=[value])


class RSSPoller:
    """Background thread that samples RSS at a fixed interval and records the peak."""

    def __init__(self, proc: psutil.Process, interval_s: float = 0.02):
        self._proc = proc
        self._interval = interval_s
        self._peak_bytes = proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
            except psutil.NoSuchProcess:
                return
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._stop.wait(self._interval)

    def __enter__(self) -> RSSPoller:
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes


def measure_one(
    seed: int,
    num_tables: int,
    propagate_batch_size: int,
) -> tuple[float, int]:
    """Run one generation and return (wall_time_s, peak_rss_delta_bytes).

    Row counts use the default `DatabaseParams` ranges so per-seed variance
    reflects realistic table-size sampling.
    """
    config = Config(
        database_params=DatabaseParams(num_tables_choices=_fixed(num_tables)),
        scm_params=SCMParams(propagate_batch_size=propagate_batch_size),
    )
    dataset = SyntheticDataset(seed=seed, config=config)

    gc.collect()
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    with RSSPoller(proc) as poller:
        t0 = time.perf_counter()
        dataset.make_db()
        elapsed = time.perf_counter() - t0
    return elapsed, poller.peak_bytes - rss_before


def _summary_stats(xs: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.mean(xs)
    std = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mean, std, min(xs), max(xs)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num_tables",
        type=int,
        default=8,
        help="Fixed number of tables per database (default: 8).",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="Number of seeds to profile (default: 10).",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=1,
        help=(
            "Throwaway iterations before measurement to absorb one-time "
            "torch/numpy init and allocator warmup (default: 1). Warmup uses "
            "seed=seed_offset; measured seeds still start at seed_offset."
        ),
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Seed offset; runs seeds [seed_offset, seed_offset + num_seeds).",
    )
    parser.add_argument(
        "--propagate_batch_size",
        type=int,
        default=4096,
        help="Rows per chunk in batched propagation (default: 4096).",
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=1,
        help="torch.set_num_threads value; 1 gives consistent timing (default: 1).",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.torch_num_threads)

    print("config")
    print(f"  num_tables             {args.num_tables}")
    print(f"  num_seeds              {args.num_seeds}")
    print(f"  num_warmup             {args.num_warmup}")
    print(f"  seed_offset            {args.seed_offset}")
    print(f"  propagate_batch_size   {args.propagate_batch_size}")
    print(f"  torch_num_threads      {args.torch_num_threads}")
    print()

    for w in range(args.num_warmup):
        print(f"warmup [{w + 1}/{args.num_warmup}]", flush=True)
        measure_one(
            seed=args.seed_offset,
            num_tables=args.num_tables,
            propagate_batch_size=args.propagate_batch_size,
        )
    if args.num_warmup:
        print()

    print(f"  {'seed':>4}   {'time (s)':>9}   {'peak Δ RSS (MB)':>17}")
    print(f"  {'-' * 4}   {'-' * 9}   {'-' * 17}")

    times: list[float] = []
    peak_deltas_mb: list[float] = []
    for offset in range(args.num_seeds):
        seed = args.seed_offset + offset
        elapsed, peak_delta_bytes = measure_one(
            seed=seed,
            num_tables=args.num_tables,
            propagate_batch_size=args.propagate_batch_size,
        )
        peak_delta_mb = peak_delta_bytes / (1024 * 1024)
        times.append(elapsed)
        peak_deltas_mb.append(peak_delta_mb)
        print(f"  {seed:>4d}   {elapsed:>9.3f}   {peak_delta_mb:>17.1f}")

    t_mean, t_std, t_min, t_max = _summary_stats(times)
    m_mean, m_std, m_min, m_max = _summary_stats(peak_deltas_mb)
    print()
    print("summary")
    print(f"  time         {t_mean:>7.3f} ± {t_std:.3f} s    [{t_min:.3f}, {t_max:.3f}]")
    print(f"  peak Δ RSS   {m_mean:>7.1f} ± {m_std:.1f} MB   [{m_min:.1f}, {m_max:.1f}]")


if __name__ == "__main__":
    main()
