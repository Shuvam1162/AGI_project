"""
run_sweep_fast.py
-----------------
Runs the validation sweep sequentially (no multiprocessing) and saves
incremental CSVs. Use this when multiprocessing causes issues.

Usage: python run_sweep_fast.py
"""
import os, sys, csv, math, time
sys.path.insert(0, os.path.dirname(__file__))

from harness.opennars_runner import run_task

TASKS   = ["task1_deductive","task2_inductive","task3_revision",
           "task4_abductive","task5_multihop"]
NC_GRID = [10, 50, 100, 250, 500, 1_000, 5_000, 50_000]
NM_GRID = [25, 50, 100, 250, 500, 1_000, 5_000, 10_000]
SEEDS   = list(range(10))
FIXED_NM = 10_000
FIXED_NC = 50_000
CONFIDENCE_THRESHOLD = 0.5

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  → Saved {path}")


def sweep_nc():
    path = os.path.join(DATA_DIR, "validation_nc.csv")
    rows = []
    total = len(TASKS) * len(NC_GRID) * len(SEEDS)
    print(f"Exp-V1 (Nc sweep): {total} runs …")
    t0 = time.time()
    i = 0
    for task in TASKS:
        for nc in NC_GRID:
            for seed in SEEDS:
                f, c = run_task(task, nc=nc, nm=FIXED_NM, seed=seed)
                rows.append({"task": task, "nc": nc, "seed": seed, "f": f, "c": c})
                i += 1
                if i % 40 == 0:
                    print(f"  {i}/{total}  ({time.time()-t0:.1f}s)")
    write_csv(path, rows)
    print(f"Exp-V1 done in {time.time()-t0:.1f}s")
    return rows


def sweep_nm():
    path = os.path.join(DATA_DIR, "validation_nm.csv")
    rows = []
    total = len(TASKS) * len(NM_GRID) * len(SEEDS)
    print(f"\nExp-V2 (Nm sweep): {total} runs …")
    t0 = time.time()
    i = 0
    for task in TASKS:
        for nm in NM_GRID:
            for seed in SEEDS:
                f, c = run_task(task, nc=FIXED_NC, nm=nm, seed=seed)
                rows.append({"task": task, "nm": nm, "seed": seed, "f": f, "c": c})
                i += 1
                if i % 40 == 0:
                    print(f"  {i}/{total}  ({time.time()-t0:.1f}s)")
    write_csv(path, rows)
    print(f"Exp-V2 done in {time.time()-t0:.1f}s")
    return rows


def avg_c(task, budget, mode, n=5):
    total = 0.0
    for seed in range(n):
        if mode == "nc":
            _, c = run_task(task, nc=budget, nm=FIXED_NM, seed=seed)
        else:
            _, c = run_task(task, nc=FIXED_NC, nm=budget, seed=seed)
        total += c
    return total / n


def find_nstar(task, mode="nc"):
    lo = 10 if mode == "nc" else 25
    hi = 50_000 if mode == "nc" else 10_000
    if avg_c(task, hi, mode) < CONFIDENCE_THRESHOLD:
        return hi
    lo_l, hi_l = math.log(lo), math.log(hi)
    for _ in range(14):
        mid_l = (lo_l + hi_l) / 2
        mid = int(math.exp(mid_l))
        if avg_c(task, mid, mode) >= CONFIDENCE_THRESHOLD:
            hi_l = mid_l
        else:
            lo_l = mid_l
        if hi_l - lo_l < 0.1:
            break
    return int(math.exp(hi_l))


def sweep_nstar():
    path = os.path.join(DATA_DIR, "validation_nstar.csv")
    rows = []
    print("\nExp-V3 (N* binary search) …")
    for task in TASKS:
        ns_nc = find_nstar(task, "nc")
        ns_nm = find_nstar(task, "nm")
        rows.append({"task": task, "nstar_nc": ns_nc, "nstar_nm": ns_nm})
        print(f"  {task}: N*_Nc={ns_nc:,}  N*_Nm={ns_nm:,}")
    write_csv(path, rows)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("  NAL Engine Validation Sweep")
    print("=" * 60)
    sweep_nc()
    sweep_nm()
    sweep_nstar()
    print(f"\nTotal: {time.time()-t0:.1f}s")
    print("=" * 60)
