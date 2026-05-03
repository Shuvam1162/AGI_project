"""
metrics.py
----------
Compute per-run and per-group metrics defined in the project plan (Section 6.3):

  δ  (truth-value error)     – Euclidean distance from ground-truth tuple
  ρ  (confidence retention)  – fraction of ground-truth confidence recovered
  N* (critical budget)       – smallest Nc/Nm such that c ≥ τ = 0.5
  Degradation profile        – smooth vs. cliff-edge classification
"""

import numpy as np
import pandas as pd
from typing import Tuple

TAU = 0.5  # confidence threshold for meaningful belief revision


# ── Core metric functions ─────────────────────────────────────────────────────

def truth_value_error(f: float, c: float, f_star: float, c_star: float) -> float:
    """δ = Euclidean distance from ground-truth truth-value tuple."""
    return float(np.sqrt((f - f_star) ** 2 + (c - c_star) ** 2))


def confidence_retention(c: float, c_star: float) -> float:
    """ρ = c / c* — fraction of ground-truth confidence recovered."""
    if c_star == 0:
        return 0.0
    return float(np.clip(c / c_star, 0.0, 2.0))   # cap at 2 to handle outliers


# ── Batch metric computation ──────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add δ and ρ columns to a raw experiment results DataFrame.

    Expected columns: task, nc (or nm), seed, f, c, f_star, c_star
    """
    df = df.copy()
    df["delta"] = df.apply(
        lambda r: truth_value_error(r.f, r.c, r.f_star, r.c_star), axis=1
    )
    df["rho"] = df.apply(
        lambda r: confidence_retention(r.c, r.c_star), axis=1
    )
    return df


def aggregate_metrics(df: pd.DataFrame, budget_col: str) -> pd.DataFrame:
    """
    Aggregate over seeds: compute mean ± std of δ, ρ, c per (task, budget).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: task, <budget_col>, delta, rho, c
    budget_col : str
        Either "nc" or "nm"

    Returns
    -------
    DataFrame with columns:
        task, <budget_col>, delta_mean, delta_std, rho_mean, rho_std,
        c_mean, c_std, f_mean
    """
    agg = (
        df.groupby(["task", budget_col])
        .agg(
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            rho_mean=("rho", "mean"),
            rho_std=("rho", "std"),
            c_mean=("c", "mean"),
            c_std=("c", "std"),
            f_mean=("f", "mean"),
        )
        .reset_index()
    )
    return agg


def classify_degradation_profile(
    agg: pd.DataFrame,
    budget_col: str,
    task: str,
    delta_jump_threshold: float = 0.15,
) -> str:
    """
    Classify a task's degradation profile as 'smooth' or 'cliff'.

    Logic:
      - Compute δ differences between successive budget levels.
      - If the largest single step exceeds delta_jump_threshold while
        preceding steps are small, classify as 'cliff'.
      - Otherwise 'smooth'.
    """
    sub = agg[agg.task == task].sort_values(budget_col)
    deltas = sub.delta_mean.values
    if len(deltas) < 3:
        return "unknown"

    diffs = np.diff(deltas)            # negative = improving with more budget
    # Normalise: look at absolute diffs as fraction of total delta range
    delta_range = deltas.max() - deltas.min()
    if delta_range < 1e-6:
        return "smooth"

    norm_diffs = np.abs(diffs) / delta_range

    # Cliff: one large jump (>threshold) while most others are small
    sorted_diffs = np.sort(norm_diffs)[::-1]
    if sorted_diffs[0] > delta_jump_threshold and (
        len(sorted_diffs) < 2 or sorted_diffs[1] < sorted_diffs[0] * 0.4
    ):
        return "cliff"
    return "smooth"


def find_critical_budget(
    agg: pd.DataFrame,
    budget_col: str,
    task: str,
    tau: float = TAU,
) -> Tuple[int, float]:
    """
    Find N* from aggregated data: smallest budget level at which c_mean ≥ τ.

    Returns (N*, c_mean at N*) or (None, None) if never reached.
    """
    sub = agg[agg.task == task].sort_values(budget_col)
    above = sub[sub.c_mean >= tau]
    if above.empty:
        return None, None
    row = above.iloc[0]
    return int(row[budget_col]), float(row.c_mean)
