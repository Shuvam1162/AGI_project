"""
validation_compare.py
---------------------
Cross-validates the calibrated simulator results against the faithful
NAL inference engine output.

Generates
---------
  figures/figV1_delta_compare_nc.png  — δ vs Nc: Simulator vs Engine (line chart)
  figures/figV2_rho_compare_nc.png    — ρ vs Nc: Simulator vs Engine
  figures/figV3_profile_compare.png   — Degradation profile side-by-side
  figures/figV4_nstar_compare.png     — N* bar chart: Simulator vs Engine
  data/validation_summary.csv         — Per-task agreement metrics

Metrics
-------
  δ  = √[(f−f*)² + (c−c*)²]   (truth-value error)
  ρ  = c / c*                  (confidence retention)
  r  = Pearson correlation of δ(sim) vs δ(engine) across budget levels
  profile agreement: bool — does engine agree on SMOOTH vs CLIFF?
"""

from __future__ import annotations
import os
import math
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(_ROOT, "data")
FIG_DIR  = os.path.join(_ROOT, "figures")

# ── Ground truth (same as nars_simulator.py) ──────────────────────────────────
GROUND_TRUTH: Dict[str, Tuple[float, float]] = {
    "task1_deductive": (1.00, 0.73),
    "task2_inductive": (0.82, 0.61),
    "task3_revision":  (0.52, 0.63),
    "task4_abductive": (0.72, 0.58),
    "task5_multihop":  (0.68, 0.54),
}

TASK_LABELS = {
    "task1_deductive": "T1: Deductive",
    "task2_inductive": "T2: Inductive",
    "task3_revision":  "T3: Revision",
    "task4_abductive": "T4: Abductive",
    "task5_multihop":  "T5: Multi-hop",
}

TASK_COLORS = {
    "task1_deductive": "#0077B6",
    "task2_inductive": "#6C63FF",
    "task3_revision":  "#F4A261",
    "task4_abductive": "#E74C3C",
    "task5_multihop":  "#9B59B6",
}

SIM_NSTAR = {
    "task1_deductive": {"nc": 374,   "nm": 374},
    "task2_inductive": {"nc": 952,   "nm": 952},
    "task3_revision":  {"nc": 1623,  "nm": 1623},
    "task4_abductive": {"nc": 400,   "nm": 400},
    "task5_multihop":  {"nc": 952,   "nm": 952},
}

SMOOTH_TASKS = {"task1_deductive", "task2_inductive", "task3_revision"}
CLIFF_TASKS  = {"task4_abductive", "task5_multihop"}


# ── Helper functions ──────────────────────────────────────────────────────────

def delta(f: float, c: float, task: str) -> float:
    f_star, c_star = GROUND_TRUTH[task]
    return math.sqrt((f - f_star) ** 2 + (c - c_star) ** 2)


def rho(c: float, task: str) -> float:
    _, c_star = GROUND_TRUTH[task]
    return c / c_star if c_star > 0 else 0.0


def classify_profile(rho_values: List[float]) -> str:
    """Smooth = monotonically non-decreasing ρ. Cliff = large non-monotone jump."""
    diffs = [rho_values[i + 1] - rho_values[i] for i in range(len(rho_values) - 1)]
    n_decreasing = sum(1 for d in diffs if d < -0.05)
    # Also check for cliff pattern: near-zero ρ followed by sudden jump
    max_jump = max(diffs) if diffs else 0
    cliff_pattern = any(r < 0.15 for r in rho_values[:3]) and max_jump > 0.3
    if n_decreasing >= 2 or cliff_pattern:
        return "CLIFF"
    return "SMOOTH"


def _load_agg(csv_path: str, budget_col: str) -> pd.DataFrame:
    """Load CSV and aggregate (mean f, mean c) per task × budget."""
    df = pd.read_csv(csv_path)
    df["delta"] = df.apply(lambda r: delta(r["f"], r["c"], r["task"]), axis=1)
    df["rho"]   = df.apply(lambda r: rho(r["c"], r["task"]), axis=1)
    agg = df.groupby(["task", budget_col])[["f", "c", "delta", "rho"]].mean().reset_index()
    return agg


# ── Figure helpers ────────────────────────────────────────────────────────────

STYLE = dict(linewidth=1.8, marker="o", markersize=5)


def _fig_setup(title: str, nrows: int = 1, ncols: int = 5, figsize=(16, 3.5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    return fig, axes.flatten()


# ── Figure V1: δ comparison (Nc sweep) ───────────────────────────────────────

def fig_delta_compare_nc(sim_agg: pd.DataFrame, eng_agg: pd.DataFrame) -> None:
    fig, axes = _fig_setup("figV1: Truth-value Error δ vs Cycle Budget — Simulator vs Engine")
    for ax, task in zip(axes, GROUND_TRUTH.keys()):
        color = TASK_COLORS[task]
        s = sim_agg[sim_agg.task == task].sort_values("nc")
        e = eng_agg[eng_agg.task == task].sort_values("nc")
        ax.plot(s["nc"], s["delta"], **STYLE, color=color,     label="Simulator", linestyle="-")
        ax.plot(e["nc"], e["delta"], **STYLE, color=color,     label="Engine",    linestyle="--", alpha=0.75)
        ax.set_xscale("log")
        ax.set_title(TASK_LABELS[task], fontsize=9, fontweight="bold")
        ax.set_xlabel("Nc (log scale)", fontsize=7)
        ax.set_ylabel("δ" if ax == axes[0] else "", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "figV1_delta_compare_nc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure V2: ρ comparison (Nc sweep) ───────────────────────────────────────

def fig_rho_compare_nc(sim_agg: pd.DataFrame, eng_agg: pd.DataFrame) -> None:
    fig, axes = _fig_setup("figV2: Confidence Retention ρ vs Cycle Budget — Simulator vs Engine")
    for ax, task in zip(axes, GROUND_TRUTH.keys()):
        color = TASK_COLORS[task]
        s = sim_agg[sim_agg.task == task].sort_values("nc")
        e = eng_agg[eng_agg.task == task].sort_values("nc")
        ax.plot(s["nc"], s["rho"], **STYLE, color=color, label="Simulator", linestyle="-")
        ax.plot(e["nc"], e["rho"], **STYLE, color=color, label="Engine",    linestyle="--", alpha=0.75)
        ax.axhline(y=1.0, color="grey", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.3)
        ax.set_title(TASK_LABELS[task], fontsize=9, fontweight="bold")
        ax.set_xlabel("Nc (log scale)", fontsize=7)
        ax.set_ylabel("ρ = c/c*" if ax == axes[0] else "", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "figV2_rho_compare_nc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure V3: Profile comparison ─────────────────────────────────────────────

def fig_profile_compare(sim_agg: pd.DataFrame, eng_agg: pd.DataFrame) -> None:
    tasks = list(GROUND_TRUTH.keys())
    fig, axes = plt.subplots(2, 5, figsize=(16, 5), sharex=False)
    fig.suptitle("figV3: Degradation Profiles — Simulator (top) vs Engine (bottom)",
                 fontsize=12, fontweight="bold")

    for col, task in enumerate(tasks):
        color = TASK_COLORS[task]
        for row, (df, label) in enumerate([(sim_agg, "Simulator"), (eng_agg, "Engine")]):
            ax = axes[row][col]
            d = df[df.task == task].sort_values("nc")
            ax.plot(d["nc"], d["rho"], color=color, marker="o", linewidth=2, markersize=5)
            ax.axhline(y=0.5 / GROUND_TRUTH[task][1], color="red",
                       linestyle="--", linewidth=0.8, label="τ/c*")
            ax.axhline(y=1.0, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_ylim(-0.05, 1.35)
            if row == 0:
                ax.set_title(TASK_LABELS[task], fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{label}\nρ = c/c*", fontsize=8)
            ax.set_xlabel("Nc" if row == 1 else "", fontsize=7)
            ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "figV3_profile_compare.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure V4: N* bar chart comparison ───────────────────────────────────────

def fig_nstar_compare(eng_nstar: pd.DataFrame) -> None:
    tasks = list(GROUND_TRUTH.keys())
    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    sim_vals = [SIM_NSTAR[t]["nc"] for t in tasks]
    eng_vals = [int(eng_nstar[eng_nstar.task == t]["nstar_nc"].values[0])
                if t in eng_nstar.task.values else 0
                for t in tasks]

    ax.bar(x - width / 2, sim_vals, width, label="Simulator N*", color="#0077B6", alpha=0.85)
    ax.bar(x + width / 2, eng_vals, width, label="Engine N*",    color="#48CAE4", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=9)
    ax.set_ylabel("Critical Budget N* (log scale)", fontsize=10)
    ax.set_title("figV4: N* Comparison — Simulator vs Faithful NAL Engine",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "figV4_nstar_compare.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Summary statistics & CSV ──────────────────────────────────────────────────

def compute_summary(
    sim_nc: pd.DataFrame,
    eng_nc: pd.DataFrame,
    eng_nstar: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for task in GROUND_TRUTH.keys():
        s = sim_nc[sim_nc.task == task].sort_values("nc")
        e = eng_nc[eng_nc.task == task].sort_values("nc")

        # Pearson r on δ columns (aligned by nc)
        s_d = s["delta"].values
        e_d = e["delta"].values
        n   = min(len(s_d), len(e_d))
        r, p = pearsonr(s_d[:n], e_d[:n]) if n >= 3 else (float("nan"), 1.0)

        # Profile classification
        sim_profile = "SMOOTH" if task in SMOOTH_TASKS else "CLIFF"
        eng_rho_vals = e["rho"].tolist()
        eng_profile  = classify_profile(eng_rho_vals)
        agrees = (sim_profile == eng_profile)

        # N* from engine
        nstar_row = eng_nstar[eng_nstar.task == task]
        eng_ns = int(nstar_row["nstar_nc"].values[0]) if len(nstar_row) else -1
        sim_ns = SIM_NSTAR[task]["nc"]

        rows.append({
            "task":           task,
            "sim_profile":    sim_profile,
            "engine_profile": eng_profile,
            "profile_agrees": agrees,
            "pearson_r":      round(r, 3),
            "pearson_p":      round(p, 4),
            "sim_nstar_nc":   sim_ns,
            "engine_nstar_nc":eng_ns,
            "nstar_ratio":    round(eng_ns / sim_ns, 2) if sim_ns > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_validation_compare() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    sim_nc_path = os.path.join(DATA_DIR, "results", "exp1_cycle_sweep.csv")
    eng_nc_path = os.path.join(DATA_DIR, "validation_nc.csv")
    eng_ns_path = os.path.join(DATA_DIR, "validation_nstar.csv")

    missing = [p for p in [sim_nc_path, eng_nc_path, eng_ns_path] if not os.path.exists(p)]
    if missing:
        print(f"  ✗  Missing data files: {missing}")
        print("     Run validate.py first.")
        return

    sim_nc  = _load_agg(sim_nc_path, "nc")
    eng_nc  = _load_agg(eng_nc_path, "nc")
    eng_ns  = pd.read_csv(eng_ns_path)

    print("\n── Validation comparison ────────────────────────────────────────────")

    # ── Generate figures ──────────────────────────────────────────────────────
    fig_delta_compare_nc(sim_nc, eng_nc)
    fig_rho_compare_nc(sim_nc, eng_nc)
    fig_profile_compare(sim_nc, eng_nc)
    fig_nstar_compare(eng_ns)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = compute_summary(sim_nc, eng_nc, eng_ns)
    summary_path = os.path.join(DATA_DIR, "validation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n  Saved → {summary_path}")

    # ── Print to console ──────────────────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │            Simulator ↔ Engine Validation Summary                │")
    print("  ├──────────────────┬──────────┬──────────┬────────┬───────────────┤")
    print("  │ Task             │ Sim Prof │ Eng Prof │ Agrees │ Pearson r     │")
    print("  ├──────────────────┼──────────┼──────────┼────────┼───────────────┤")
    for _, row in summary.iterrows():
        tag  = TASK_LABELS[row["task"]]
        tick = "✓" if row["profile_agrees"] else "✗"
        r    = f"{row['pearson_r']:.3f}" if not math.isnan(row["pearson_r"]) else "N/A"
        print(f"  │ {tag:<16} │ {row['sim_profile']:<8} │ {row['engine_profile']:<8} │ {tick:<6} │ r = {r:<9} │")
    print("  └──────────────────┴──────────┴──────────┴────────┴───────────────┘")

    n_agree = summary["profile_agrees"].sum()
    print(f"\n  Profile agreement: {n_agree}/5 tasks")
    mean_r = summary["pearson_r"].dropna().mean()
    print(f"  Mean Pearson r(δ): {mean_r:.3f}")
    print("\n── Validation comparison complete ───────────────────────────────────\n")
