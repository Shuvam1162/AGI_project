"""
statistics.py
-------------
Statistical significance testing for NARS experiment results.

Uses the paired Wilcoxon signed-rank test (α = 0.05) as specified in Section
6.4 of the project plan. Chosen for robustness to non-normal distributions,
which arise due to stochastic priority sampling in the NARS concept bag.

Tests performed:
  - Between adjacent resource levels: does increasing Nc/Nm significantly
    improve δ (or ρ)?
  - Between cycle-budget and memory-budget degradation: are the patterns
    qualitatively different (RQ2)?
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


ALPHA = 0.05


def wilcoxon_adjacent_levels(
    df: pd.DataFrame,
    budget_col: str,
    metric: str = "delta",
    task: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run paired Wilcoxon tests between adjacent budget levels for all tasks
    (or a single task if specified).

    For each adjacent pair (level_i, level_{i+1}), we test:
      H0: metric at level_i == metric at level_{i+1}
      H1: metric at level_i != metric at level_{i+1}  (two-sided)

    Seeds serve as paired observations.

    Returns a DataFrame with columns:
      task, level_lo, level_hi, statistic, p_value, significant
    """
    tasks = [task] if task else df.task.unique().tolist()
    levels = sorted(df[budget_col].unique())
    records = []

    for t in tasks:
        sub = df[df.task == t]
        for i in range(len(levels) - 1):
            lo, hi = levels[i], levels[i + 1]
            x = sub[sub[budget_col] == lo][metric].values
            y = sub[sub[budget_col] == hi][metric].values

            # Align lengths (should both be N_SEEDS, but guard anyway)
            n = min(len(x), len(y))
            if n < 3:
                continue
            x, y = x[:n], y[:n]

            # Skip if all differences are zero
            if np.all(x - y == 0):
                stat, p = np.nan, 1.0
            else:
                try:
                    stat, p = stats.wilcoxon(x, y, alternative="two-sided")
                except ValueError:
                    stat, p = np.nan, 1.0

            records.append({
                "task":        t,
                "level_lo":    lo,
                "level_hi":    hi,
                "statistic":   round(float(stat), 4) if not np.isnan(stat) else np.nan,
                "p_value":     round(float(p), 6),
                "significant": p < ALPHA,
            })

    return pd.DataFrame(records)


def compare_cycle_vs_memory(
    df_cycles: pd.DataFrame,
    df_memory: pd.DataFrame,
    metric: str = "delta",
) -> pd.DataFrame:
    """
    For each task, compare the distributions of <metric> across all budget
    levels between the cycle-sweep and memory-sweep experiments.

    Uses Mann-Whitney U test (non-paired, since the two distributions have
    different budget axes).

    Returns a DataFrame with: task, statistic, p_value, significant, interpretation
    """
    records = []
    for task in df_cycles.task.unique():
        x = df_cycles[df_cycles.task == task][metric].values
        y = df_memory[df_memory.task == task][metric].values
        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        records.append({
            "task":        task,
            "statistic":   round(float(stat), 2),
            "p_value":     round(float(p), 6),
            "significant": p < ALPHA,
            "interpretation": (
                "Cycle and memory degradation are DISTINCT" if p < ALPHA
                else "No significant difference between cycle and memory degradation"
            ),
        })
    return pd.DataFrame(records)


def summarise_significance(wilcoxon_df: pd.DataFrame) -> Dict[str, int]:
    """
    Count the number of significant vs. non-significant adjacent-level
    comparisons per task.
    """
    summary = {}
    for task, grp in wilcoxon_df.groupby("task"):
        summary[task] = {
            "total_comparisons":  len(grp),
            "significant":        int(grp.significant.sum()),
            "non_significant":    int((~grp.significant).sum()),
        }
    return summary


if __name__ == "__main__":
    # Smoke test with random data
    rng = np.random.default_rng(42)
    records = []
    for task in ["task1", "task2"]:
        for nc in [10, 100, 1000]:
            for seed in range(10):
                records.append({
                    "task": task, "nc": nc, "seed": seed,
                    "delta": rng.uniform(0, 1),
                    "rho": rng.uniform(0, 1),
                })
    df = pd.DataFrame(records)
    result = wilcoxon_adjacent_levels(df, "nc", "delta")
    print(result.to_string(index=False))
