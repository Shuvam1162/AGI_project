"""
main.py
-------
End-to-end pipeline for the NARS Resource Constraint Study.

Usage:
    python main.py                    # run full pipeline
    python main.py --skip-sweep       # use existing CSV results
    python main.py --workers 8        # parallelism
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = "data/results"
FIGURES_DIR = "figures"


def main(skip_sweep: bool = False, workers: int = 4):
    # ─────────────────────────────────────────────
    # PHASE 1: Run experiments (or load from disk)
    # ─────────────────────────────────────────────
    exp1_csv = Path(RESULTS_DIR) / "exp1_cycle_sweep.csv"
    exp2_csv = Path(RESULTS_DIR) / "exp2_memory_sweep.csv"
    exp3_csv = Path(RESULTS_DIR) / "exp3_critical_budget.csv"

    if not skip_sweep or not exp1_csv.exists():
        log.info("Running parameter sweep experiments …")
        from harness.parameter_sweep import (
            run_experiment1, run_experiment2, run_experiment3,
        )
        run_experiment1(RESULTS_DIR, max_workers=workers)
        run_experiment2(RESULTS_DIR, max_workers=workers)
        run_experiment3(RESULTS_DIR)
    else:
        log.info("Loading existing experiment results …")

    df1 = pd.read_csv(exp1_csv)
    df2 = pd.read_csv(exp2_csv)
    df3 = pd.read_csv(exp3_csv)

    # ─────────────────────────────────────────────
    # PHASE 2: Compute metrics
    # ─────────────────────────────────────────────
    log.info("Computing metrics …")
    from analysis.metrics import (
        compute_metrics, aggregate_metrics,
        classify_degradation_profile, find_critical_budget,
    )

    df1 = compute_metrics(df1)
    df2 = compute_metrics(df2)

    agg1 = aggregate_metrics(df1, "nc")
    agg2 = aggregate_metrics(df2, "nm")

    # Save aggregated results
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    agg1.to_csv(Path(RESULTS_DIR) / "agg1_cycle.csv", index=False)
    agg2.to_csv(Path(RESULTS_DIR) / "agg2_memory.csv", index=False)

    # Classify degradation profiles
    tasks = df1.task.unique().tolist()
    profiles_cycles  = {t: classify_degradation_profile(agg1, "nc", t) for t in tasks}
    profiles_memory  = {t: classify_degradation_profile(agg2, "nm", t) for t in tasks}

    log.info("Degradation profiles (cycles):")
    for t, p in profiles_cycles.items():
        log.info(f"  {t}: {p}")
    log.info("Degradation profiles (memory):")
    for t, p in profiles_memory.items():
        log.info(f"  {t}: {p}")

    # Find critical budgets from aggregated data
    n_star_nc = {}
    n_star_nm = {}
    for t in tasks:
        n_star_nc[t], _ = find_critical_budget(agg1, "nc", t)
        n_star_nm[t], _ = find_critical_budget(agg2, "nm", t)

    log.info("Critical budgets (from aggregated data):")
    for t in tasks:
        log.info(f"  {t}: N*_c={n_star_nc[t]}, N*_m={n_star_nm[t]}")

    # Also read binary-search N* from Exp3
    exp3_nstar = dict(zip(df3.task, df3.n_star_nc))
    log.info("Binary-search critical budgets (Exp3):")
    for t, v in exp3_nstar.items():
        log.info(f"  {t}: {v}")

    # ─────────────────────────────────────────────
    # PHASE 3: Statistical tests
    # ─────────────────────────────────────────────
    log.info("Running Wilcoxon significance tests …")
    from analysis.statistics import (
        wilcoxon_adjacent_levels, compare_cycle_vs_memory, summarise_significance,
    )

    wilcox1 = wilcoxon_adjacent_levels(df1, "nc", "delta")
    wilcox2 = wilcoxon_adjacent_levels(df2, "nm", "delta")
    cmp_df  = compare_cycle_vs_memory(df1, df2, "delta")

    wilcox1.to_csv(Path(RESULTS_DIR) / "stats_wilcoxon_cycles.csv", index=False)
    wilcox2.to_csv(Path(RESULTS_DIR) / "stats_wilcoxon_memory.csv", index=False)
    cmp_df.to_csv(Path(RESULTS_DIR)  / "stats_cycle_vs_memory.csv", index=False)

    sig_summary = summarise_significance(wilcox1)
    log.info("Significance summary (cycle sweep):")
    for t, s in sig_summary.items():
        log.info(f"  {t}: {s['significant']}/{s['total_comparisons']} pairs significant")

    # ─────────────────────────────────────────────
    # PHASE 4: Generate figures
    # ─────────────────────────────────────────────
    log.info("Generating figures …")
    from analysis.visualize import generate_all_figures

    fig_paths = generate_all_figures(
        agg1=agg1,
        agg2=agg2,
        profiles_cycles=profiles_cycles,
        profiles_memory=profiles_memory,
        n_star_nc=n_star_nc,
        n_star_nm=n_star_nm,
        out_dir=FIGURES_DIR,
    )

    # ─────────────────────────────────────────────
    # Return everything for report generation
    # ─────────────────────────────────────────────
    return {
        "df1": df1, "df2": df2, "df3": df3,
        "agg1": agg1, "agg2": agg2,
        "profiles_cycles": profiles_cycles,
        "profiles_memory": profiles_memory,
        "n_star_nc": n_star_nc,
        "n_star_nm": n_star_nm,
        "exp3_nstar": exp3_nstar,
        "wilcox1": wilcox1,
        "wilcox2": wilcox2,
        "cmp_df":  cmp_df,
        "fig_paths": fig_paths,
        "sig_summary": sig_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(skip_sweep=args.skip_sweep, workers=args.workers)
