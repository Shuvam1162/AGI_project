"""
validation_sweep.py
-------------------
Runs the same parameter sweeps as parameter_sweep.py but using the faithful
NAL inference engine (nal_engine.py) instead of the calibrated simulator.

Experiments
-----------
  Exp-V1 : Cycle budget sweep   — 5 tasks × 8 Nc levels × 10 seeds = 400 runs
  Exp-V2 : Memory cap sweep     — 5 tasks × 8 Nm levels × 10 seeds = 400 runs
  Exp-V3 : Binary search for N* — per task, geometric bisection

Output
------
  data/validation_nc.csv    columns: task, nc, seed, f, c
  data/validation_nm.csv    columns: task, nm, seed, f, c
  data/validation_nstar.csv columns: task, nstar_nc, nstar_nm
"""

from __future__ import annotations
import os
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from harness.opennars_runner import run_task

# ── Sweep grid (mirrors parameter_sweep.py) ───────────────────────────────────
TASKS   = ["task1_deductive", "task2_inductive", "task3_revision",
           "task4_abductive", "task5_multihop"]
NC_GRID = [10, 50, 100, 250, 500, 1_000, 5_000, 50_000]
NM_GRID = [25, 50, 100, 250, 500, 1_000, 5_000, 10_000]
SEEDS   = list(range(10))

CONFIDENCE_THRESHOLD: float = 0.5   # τ for N* search
N_WORKERS: int = 4
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Fixed Nm for Exp-V1; fixed Nc for Exp-V2
FIXED_NM_FOR_NC_SWEEP = 10_000
FIXED_NC_FOR_NM_SWEEP = 50_000


# ── Worker functions (must be picklable for multiprocessing) ──────────────────

def _nc_worker(args):
    task, nc, seed = args
    result = run_task(task, nc=nc, nm=FIXED_NM_FOR_NC_SWEEP, seed=seed)
    f, c   = result if result else (0.5, 0.01)
    return task, nc, seed, f, c


def _nm_worker(args):
    task, nm, seed = args
    result = run_task(task, nc=FIXED_NC_FOR_NM_SWEEP, nm=nm, seed=seed)
    f, c   = result if result else (0.5, 0.01)
    return task, nm, seed, f, c


# ── Experiment V1: Cycle Budget Sweep ────────────────────────────────────────

def sweep_nc(workers: int = N_WORKERS) -> List[dict]:
    jobs = [(task, nc, seed) for task in TASKS for nc in NC_GRID for seed in SEEDS]
    rows = []
    print(f"  Exp-V1 (cycle sweep): {len(jobs)} runs …")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_nc_worker, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            task, nc, seed, f, c = fut.result()
            rows.append({"task": task, "nc": nc, "seed": seed, "f": f, "c": c})
            if i % 50 == 0:
                print(f"    {i}/{len(jobs)} done ({time.time()-t0:.1f}s)")
    print(f"  Exp-V1 done in {time.time()-t0:.1f}s")
    return rows


def sweep_nm(workers: int = N_WORKERS) -> List[dict]:
    jobs = [(task, nm, seed) for task in TASKS for nm in NM_GRID for seed in SEEDS]
    rows = []
    print(f"  Exp-V2 (memory sweep): {len(jobs)} runs …")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_nm_worker, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            task, nm, seed, f, c = fut.result()
            rows.append({"task": task, "nm": nm, "seed": seed, "f": f, "c": c})
            if i % 50 == 0:
                print(f"    {i}/{len(jobs)} done ({time.time()-t0:.1f}s)")
    print(f"  Exp-V2 done in {time.time()-t0:.1f}s")
    return rows


# ── Experiment V3: Binary search for N* ──────────────────────────────────────

def _avg_confidence(task: str, budget: int, mode: str, n_trials: int = 5) -> float:
    """Average confidence over n_trials seeds for a given budget."""
    total = 0.0
    for seed in range(n_trials):
        if mode == "nc":
            result = run_task(task, nc=budget, nm=FIXED_NM_FOR_NC_SWEEP, seed=seed)
        else:
            result = run_task(task, nc=FIXED_NC_FOR_NM_SWEEP, nm=budget, seed=seed)
        _, c = result if result else (0.5, 0.01)
        total += c
    return total / n_trials


def find_nstar(task: str, mode: str = "nc") -> int:
    """
    Geometric binary search for the minimum budget at which mean c ≥ τ.
    Search range: [lo, hi] = [10, 50000] for Nc, [25, 10000] for Nm.
    """
    lo = 10  if mode == "nc" else 25
    hi = 50_000 if mode == "nc" else 10_000

    # Check if even the full budget achieves threshold
    if _avg_confidence(task, hi, mode) < CONFIDENCE_THRESHOLD:
        return hi  # never reaches threshold

    # Binary search in log space
    lo_log = math.log(lo)
    hi_log = math.log(hi)
    for _ in range(14):          # ≤14 iterations → ±10% precision
        mid_log = (lo_log + hi_log) / 2.0
        mid = int(math.exp(mid_log))
        avg_c = _avg_confidence(task, mid, mode)
        if avg_c >= CONFIDENCE_THRESHOLD:
            hi_log = mid_log
        else:
            lo_log = mid_log
        if (hi_log - lo_log) < 0.1:   # ~10% precision
            break

    return int(math.exp(hi_log))


def sweep_nstar() -> List[dict]:
    rows = []
    print("  Exp-V3 (binary search N*)…")
    for task in TASKS:
        ns_nc = find_nstar(task, mode="nc")
        ns_nm = find_nstar(task, mode="nm")
        rows.append({"task": task, "nstar_nc": ns_nc, "nstar_nm": ns_nm})
        print(f"    {task}: N*_Nc={ns_nc:,}  N*_Nm={ns_nm:,}")
    return rows


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {path}")


# ── Main entry ────────────────────────────────────────────────────────────────

def run_validation_sweep(workers: int = N_WORKERS) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n── Validation sweep (faithful NAL engine) ──────────────────────────")

    nc_rows    = sweep_nc(workers)
    nm_rows    = sweep_nm(workers)
    nstar_rows = sweep_nstar()

    _write_csv(os.path.join(DATA_DIR, "validation_nc.csv"),    nc_rows)
    _write_csv(os.path.join(DATA_DIR, "validation_nm.csv"),    nm_rows)
    _write_csv(os.path.join(DATA_DIR, "validation_nstar.csv"), nstar_rows)

    print("── Validation sweep complete ────────────────────────────────────────\n")


if __name__ == "__main__":
    run_validation_sweep()
