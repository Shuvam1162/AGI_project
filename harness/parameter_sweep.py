"""
parameter_sweep.py
------------------
Orchestrates the full experimental parameter sweep described in the project plan.

Experiment 1 (RQ1 – Cycle Budget):
    5 tasks × 8 cycle levels × 10 seeds = 400 runs
    Memory fixed at Nm = 10,000 (full budget)

Experiment 2 (RQ2 – Memory Capacity):
    5 tasks × 8 memory levels × 10 seeds = 400 runs
    Cycles fixed at Nc = 50,000 (full budget)

Experiment 3 (RQ3 – Critical Budget Search):
    Binary search over Nc for each task to find N* (±10% precision)

Results are saved as CSV files in data/results/.
"""

import os
import csv
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# ── Parameter grids from Section 6.2 of the project plan ─────────────────────
CYCLE_LEVELS = [10, 50, 100, 250, 500, 1_000, 5_000, 50_000]
MEMORY_LEVELS = [25, 50, 100, 250, 500, 1_000, 5_000, 10_000]
N_SEEDS = 10
TASKS = [
    "task1_deductive",
    "task2_inductive",
    "task3_revision",
    "task4_abductive",
    "task5_multihop",
]

# τ threshold for "meaningful belief revision" (Section 6.3)
TAU = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _run_single(args: Tuple) -> Dict[str, Any]:
    """Worker function executed in a subprocess."""
    task, nc, nm, seed, vary = args
    # Import here so each worker process gets its own copy
    from harness.nars_simulator import NARSSimulator
    sim = NARSSimulator(vary=vary)
    f, c = sim.run(task, max_cycles=nc, concept_bag_size=nm, seed=seed)
    gt_f, gt_c = sim.ground_truth(task)
    return {
        "task":    task,
        "nc":      nc,
        "nm":      nm,
        "seed":    seed,
        "f":       round(f, 6),
        "c":       round(c, 6),
        "f_star":  round(gt_f, 6),
        "c_star":  round(gt_c, 6),
    }


def _save_csv(rows: List[Dict], path: Path):
    """Write result rows to a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved {len(rows)} rows → {path}")


def run_experiment1(
    results_dir: str = "data/results",
    max_workers: int = 4,
) -> Path:
    """
    Experiment 1: Vary inference-cycle budget.
    Returns path to the saved CSV.
    """
    log.info("=== Experiment 1: Cycle Budget Sweep ===")
    jobs = [
        (task, nc, 10_000, seed, "cycles")
        for task in TASKS
        for nc   in CYCLE_LEVELS
        for seed in range(N_SEEDS)
    ]
    log.info(f"Total jobs: {len(jobs)}")
    rows = _execute_jobs(jobs, max_workers)
    out = Path(results_dir) / "exp1_cycle_sweep.csv"
    _save_csv(rows, out)
    return out


def run_experiment2(
    results_dir: str = "data/results",
    max_workers: int = 4,
) -> Path:
    """
    Experiment 2: Vary concept-memory size.
    Returns path to the saved CSV.
    """
    log.info("=== Experiment 2: Memory Capacity Sweep ===")
    jobs = [
        (task, 50_000, nm, seed, "memory")
        for task in TASKS
        for nm   in MEMORY_LEVELS
        for seed in range(N_SEEDS)
    ]
    log.info(f"Total jobs: {len(jobs)}")
    rows = _execute_jobs(jobs, max_workers)
    out = Path(results_dir) / "exp2_memory_sweep.csv"
    _save_csv(rows, out)
    return out


def run_experiment3(
    results_dir: str = "data/results",
) -> Path:
    """
    Experiment 3: Binary search for critical cycle budget N* per task.
    Uses a single seed (seed=0) with 5 repetitions at each probe point.
    Returns path to the saved CSV.
    """
    log.info("=== Experiment 3: Critical Budget Search ===")
    from harness.nars_simulator import NARSSimulator
    sim = NARSSimulator(vary="cycles")

    rows = []
    for task in TASKS:
        lo, hi = 10, 50_000
        while hi - lo > lo * 0.10:   # stop at ±10% precision
            mid = int(np.sqrt(lo * hi))  # geometric midpoint for log search
            cs = [sim.run(task, max_cycles=mid, concept_bag_size=10_000, seed=s)[1]
                  for s in range(5)]
            mean_c = np.mean(cs)
            if mean_c >= TAU:
                hi = mid
            else:
                lo = mid
        n_star = int(np.sqrt(lo * hi))
        log.info(f"  {task}: N* ≈ {n_star} (c̄ probe = {mean_c:.3f})")
        rows.append({"task": task, "n_star_nc": n_star, "tau": TAU})

    out = Path(results_dir) / "exp3_critical_budget.csv"
    _save_csv(rows, out)
    return out


def _execute_jobs(
    jobs: List[Tuple],
    max_workers: int,
) -> List[Dict]:
    """Run jobs in parallel using ProcessPoolExecutor."""
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_single, job): job for job in jobs}
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                log.info(f"  {done}/{len(jobs)} runs complete ({elapsed:.1f}s)")
    log.info(f"  All {len(rows)} runs complete in {time.time()-t0:.1f}s")
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NARS parameter sweep")
    parser.add_argument("--exp", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.exp in ("1", "all"):
        run_experiment1(max_workers=args.workers)
    if args.exp in ("2", "all"):
        run_experiment2(max_workers=args.workers)
    if args.exp in ("3", "all"):
        run_experiment3()
