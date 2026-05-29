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
import multiprocessing
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


def build_config(num_tables: int, propagate_batch_size: int) -> Config:
    """Build the main-branch Config to profile, fixing the table count and batch
    size while leaving all other parameters at their defaults."""
    return Config(
        database_params=DatabaseParams(num_tables_choices=_fixed(num_tables)),
        scm_params=SCMParams(propagate_batch_size=propagate_batch_size),
    )


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
    absolute: bool = False,
) -> tuple[float, int]:
    """Run one generation and return (wall_time_s, peak_rss_bytes).

    Row counts use the default `DatabaseParams` ranges so per-seed variance
    reflects realistic table-size sampling.

    With ``absolute=False`` the memory value is peak RSS *growth* during the
    call (``peak - rss_before``), which excludes the interpreter/torch baseline
    but is only meaningful in a fresh process (the allocator high-water mark
    carries across calls in a shared process). With ``absolute=True`` it is the
    absolute peak RSS of the process — only use this when each call runs in its
    own subprocess (see ``--isolate``).
    """
    config = build_config(num_tables=num_tables, propagate_batch_size=propagate_batch_size)
    dataset = SyntheticDataset(seed=seed, config=config)

    gc.collect()
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    with RSSPoller(proc) as poller:
        t0 = time.perf_counter()
        dataset.make_db()
        elapsed = time.perf_counter() - t0
    peak = poller.peak_bytes if absolute else poller.peak_bytes - rss_before
    return elapsed, peak


def _isolated_worker(payload: tuple) -> tuple[float, int]:
    """Top-level (picklable) worker for spawn-based per-seed isolation.

    Each invocation runs in a fresh process (Pool with maxtasksperchild=1), so
    the reported absolute peak RSS = baseline + this generation's peak, with no
    carry-over from other seeds.
    """
    seed, num_tables, propagate_batch_size, torch_num_threads = payload
    torch.set_num_threads(torch_num_threads)
    return measure_one(
        seed=seed,
        num_tables=num_tables,
        propagate_batch_size=propagate_batch_size,
        absolute=True,
    )


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
    parser.add_argument(
        "--isolate",
        action="store_true",
        help=(
            "Run each seed in a fresh spawned subprocess and report ABSOLUTE "
            "peak RSS (baseline + generation) instead of per-call RSS growth. "
            "Required for meaningful absolute-memory numbers, since the allocator "
            "high-water mark otherwise carries across seeds in a shared process. "
            "Warmup is skipped in this mode (fresh processes share no state)."
        ),
    )
    args = parser.parse_args()

    torch.set_num_threads(args.torch_num_threads)

    mem_label = "peak RSS (GB)" if args.isolate else "peak Δ RSS (MB)"

    print("config")
    print(f"  num_tables             {args.num_tables}")
    print(f"  num_seeds              {args.num_seeds}")
    print(f"  num_warmup             {0 if args.isolate else args.num_warmup}")
    print(f"  seed_offset            {args.seed_offset}")
    print(f"  propagate_batch_size   {args.propagate_batch_size}")
    print(f"  torch_num_threads      {args.torch_num_threads}")
    print(f"  isolate                {args.isolate}")
    print(f"  memory metric          {'absolute peak RSS' if args.isolate else 'peak RSS growth'}")
    print()

    times: list[float] = []
    peaks: list[float] = []  # MB when delta-mode, GB when isolate-mode

    if args.isolate:
        ctx = multiprocessing.get_context("spawn")
        payloads = [
            (
                args.seed_offset + offset,
                args.num_tables,
                args.propagate_batch_size,
                args.torch_num_threads,
            )
            for offset in range(args.num_seeds)
        ]
        print(f"  {'seed':>4}   {'time (s)':>9}   {mem_label:>17}")
        print(f"  {'-' * 4}   {'-' * 9}   {'-' * 17}")
        # processes=1 keeps runs serial (clean timing); maxtasksperchild=1
        # gives each seed a brand-new process (clean absolute peak RSS).
        with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
            for payload, (elapsed, peak_bytes) in zip(
                payloads, pool.imap(_isolated_worker, payloads)
            ):
                seed = payload[0]
                peak_gb = peak_bytes / (1024**3)
                times.append(elapsed)
                peaks.append(peak_gb)
                print(f"  {seed:>4d}   {elapsed:>9.3f}   {peak_gb:>17.3f}")
    else:
        for w in range(args.num_warmup):
            print(f"warmup [{w + 1}/{args.num_warmup}]", flush=True)
            measure_one(
                seed=args.seed_offset,
                num_tables=args.num_tables,
                propagate_batch_size=args.propagate_batch_size,
            )
        if args.num_warmup:
            print()

        print(f"  {'seed':>4}   {'time (s)':>9}   {mem_label:>17}")
        print(f"  {'-' * 4}   {'-' * 9}   {'-' * 17}")
        for offset in range(args.num_seeds):
            seed = args.seed_offset + offset
            elapsed, peak_delta_bytes = measure_one(
                seed=seed,
                num_tables=args.num_tables,
                propagate_batch_size=args.propagate_batch_size,
            )
            peak_delta_mb = peak_delta_bytes / (1024 * 1024)
            times.append(elapsed)
            peaks.append(peak_delta_mb)
            print(f"  {seed:>4d}   {elapsed:>9.3f}   {peak_delta_mb:>17.1f}")

    t_mean, t_std, t_min, t_max = _summary_stats(times)
    m_mean, m_std, m_min, m_max = _summary_stats(peaks)
    mem_unit = "GB" if args.isolate else "MB"
    mem_fmt = ".3f" if args.isolate else ".1f"
    print()
    print("summary")
    print(f"  time         {t_mean:>7.3f} ± {t_std:.3f} s    [{t_min:.3f}, {t_max:.3f}]")
    print(
        f"  {mem_label.split(' (')[0]:<12} {m_mean:>7{mem_fmt}} ± {m_std:{mem_fmt}} {mem_unit}   "
        f"[{m_min:{mem_fmt}}, {m_max:{mem_fmt}}]"
    )


if __name__ == "__main__":
    main()
