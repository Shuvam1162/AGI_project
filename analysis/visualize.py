"""
visualize.py
------------
Generate all figures for the NARS resource-constraint study:

  Figure 1 – δ vs. log(Nc) curves (one line per task)      [Exp 1]
  Figure 2 – ρ (confidence retention) vs. log(Nc)           [Exp 1]
  Figure 3 – ρ heatmap: task × Nc                           [Exp 1]
  Figure 4 – δ vs. log(Nm) curves (memory sweep)            [Exp 2]
  Figure 5 – ρ heatmap: task × Nm                           [Exp 2]
  Figure 6 – Degradation profile classification bar chart    [Exp 1+2]
  Figure 7 – Critical budget N* per task (cycle vs. memory)  [Exp 3]
  Figure 8 – Anytime property: c vs. Nc for best/worst tasks [Exp 1]
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("tab10", 5)
TASK_LABELS = {
    "task1_deductive": "T1: Deductive",
    "task2_inductive": "T2: Inductive",
    "task3_revision":  "T3: Revision",
    "task4_abductive": "T4: Abductive",
    "task5_multihop":  "T5: Multi-hop",
}
TASK_ORDER = list(TASK_LABELS.keys())

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def _task_label(t: str) -> str:
    return TASK_LABELS.get(t, t)


# ── Figure 1: delta vs log(Nc) ───────────────────────────────────────────────

def fig_delta_vs_cycles(agg: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, task in enumerate(TASK_ORDER):
        sub = agg[agg.task == task].sort_values("nc")
        ax.errorbar(
            sub.nc, sub.delta_mean, yerr=sub.delta_std,
            label=_task_label(task), color=PALETTE[i],
            marker="o", markersize=5, capsize=3, linewidth=1.8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Inference Cycles $N_c$ (log scale)")
    ax.set_ylabel("Truth-value Error $\\delta$")
    ax.set_title("RQ1: Truth-value Error vs. Inference Cycle Budget")
    ax.legend(loc="upper right", fontsize=9)
    ax.invert_xaxis()   # degradation increases as budget shrinks (right to left)
    plt.tight_layout()
    path = out_dir / "fig1_delta_vs_cycles.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 2: rho vs log(Nc) ─────────────────────────────────────────────────

def fig_rho_vs_cycles(agg: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, task in enumerate(TASK_ORDER):
        sub = agg[agg.task == task].sort_values("nc")
        ax.errorbar(
            sub.nc, sub.rho_mean, yerr=sub.rho_std,
            label=_task_label(task), color=PALETTE[i],
            marker="s", markersize=5, capsize=3, linewidth=1.8,
        )
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Perfect retention")
    ax.axhline(0.5, color="red",  linestyle=":",  linewidth=1, label="τ threshold (ρ=0.5/c*)")
    ax.set_xscale("log")
    ax.set_xlabel("Inference Cycles $N_c$ (log scale)")
    ax.set_ylabel("Confidence Retention $\\rho = c / c^*$")
    ax.set_title("RQ1: Confidence Retention vs. Inference Cycle Budget")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = out_dir / "fig2_rho_vs_cycles.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 3: rho heatmap task × Nc ─────────────────────────────────────────

def fig_rho_heatmap_cycles(agg: pd.DataFrame, out_dir: Path) -> Path:
    pivot = agg.pivot(index="task", columns="nc", values="rho_mean")
    pivot.index = [_task_label(t) for t in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
        annot=True, fmt=".2f", linewidths=0.4,
        cbar_kws={"label": "Confidence Retention ρ"},
    )
    ax.set_xlabel("Inference Cycles $N_c$")
    ax.set_ylabel("")
    ax.set_title("Confidence Retention Heatmap: Task × Cycle Budget")
    ax.set_xticklabels(
        [f"{int(x.get_text()):,}" for x in ax.get_xticklabels()],
        rotation=40, ha="right"
    )
    plt.tight_layout()
    path = out_dir / "fig3_rho_heatmap_cycles.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 4: delta vs log(Nm) ───────────────────────────────────────────────

def fig_delta_vs_memory(agg: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, task in enumerate(TASK_ORDER):
        sub = agg[agg.task == task].sort_values("nm")
        ax.errorbar(
            sub.nm, sub.delta_mean, yerr=sub.delta_std,
            label=_task_label(task), color=PALETTE[i],
            marker="^", markersize=5, capsize=3, linewidth=1.8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Concept Memory Size $N_m$ (log scale)")
    ax.set_ylabel("Truth-value Error $\\delta$")
    ax.set_title("RQ2: Truth-value Error vs. Concept Memory Capacity")
    ax.legend(loc="upper right", fontsize=9)
    ax.invert_xaxis()
    plt.tight_layout()
    path = out_dir / "fig4_delta_vs_memory.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 5: rho heatmap task × Nm ─────────────────────────────────────────

def fig_rho_heatmap_memory(agg: pd.DataFrame, out_dir: Path) -> Path:
    pivot = agg.pivot(index="task", columns="nm", values="rho_mean")
    pivot.index = [_task_label(t) for t in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
        annot=True, fmt=".2f", linewidths=0.4,
        cbar_kws={"label": "Confidence Retention ρ"},
    )
    ax.set_xlabel("Concept Memory Size $N_m$")
    ax.set_ylabel("")
    ax.set_title("Confidence Retention Heatmap: Task × Memory Capacity")
    ax.set_xticklabels(
        [f"{int(x.get_text()):,}" for x in ax.get_xticklabels()],
        rotation=40, ha="right"
    )
    plt.tight_layout()
    path = out_dir / "fig5_rho_heatmap_memory.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 6: Degradation profile classification ──────────────────────────────

def fig_degradation_profiles(
    profiles_cycles: Dict[str, str],
    profiles_memory: Dict[str, str],
    out_dir: Path,
) -> Path:
    tasks = TASK_ORDER
    x = np.arange(len(tasks))
    width = 0.35

    def encode(p): return 1 if p == "smooth" else 0

    cycle_vals  = [encode(profiles_cycles.get(t, "smooth")) for t in tasks]
    memory_vals = [encode(profiles_memory.get(t, "smooth")) for t in tasks]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width/2, cycle_vals,  width, label="Cycle sweep",
                   color=[PALETTE[i] for i in range(len(tasks))], alpha=0.85)
    bars2 = ax.bar(x + width/2, memory_vals, width, label="Memory sweep",
                   color=[PALETTE[i] for i in range(len(tasks))], alpha=0.45,
                   hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels([_task_label(t) for t in tasks], rotation=20, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Cliff-edge", "Smooth"])
    ax.set_ylabel("Degradation Profile")
    ax.set_title("Degradation Profile Classification per Task")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "fig6_degradation_profiles.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 7: Critical budget N* bar chart ────────────────────────────────────

def fig_critical_budget(
    n_star_nc: Dict[str, Optional[int]],
    n_star_nm: Dict[str, Optional[int]],
    out_dir: Path,
) -> Path:
    tasks = TASK_ORDER
    x = np.arange(len(tasks))
    width = 0.35

    nc_vals = [n_star_nc.get(t) or 0 for t in tasks]
    nm_vals = [n_star_nm.get(t) or 0 for t in tasks]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, nc_vals, width, label="Critical $N_c^*$",
           color=PALETTE[:len(tasks)], alpha=0.85)
    ax.bar(x + width/2, nm_vals, width, label="Critical $N_m^*$",
           color=PALETTE[:len(tasks)], alpha=0.45, hatch="///")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_task_label(t) for t in tasks], rotation=20, ha="right")
    ax.set_ylabel("Critical Budget (log scale)")
    ax.set_title("RQ3: Minimum Resource Budget for Meaningful Belief Revision (c ≥ 0.5)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()
    path = out_dir / "fig7_critical_budget.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 8: Anytime property — c vs. Nc ────────────────────────────────────

def fig_anytime_property(agg: pd.DataFrame, out_dir: Path) -> Path:
    """
    Show c_mean vs. Nc for the easiest task (T1) and hardest task (T5)
    to illustrate anytime vs. non-anytime behaviour.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    focus_tasks = [
        ("task1_deductive", "T1: Deductive (Smooth / Anytime)"),
        ("task5_multihop",  "T5: Multi-hop (Cliff-edge / Non-anytime)"),
    ]

    for ax, (task, title) in zip(axes, focus_tasks):
        sub = agg[agg.task == task].sort_values("nc")
        ax.plot(sub.nc, sub.c_mean, "o-", color="steelblue", linewidth=2)
        ax.fill_between(
            sub.nc,
            sub.c_mean - sub.c_std,
            sub.c_mean + sub.c_std,
            alpha=0.2, color="steelblue",
        )
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="τ = 0.5")
        ax.set_xscale("log")
        ax.set_xlabel("Inference Cycles $N_c$")
        ax.set_ylabel("Mean Confidence $c$")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("RQ4: Anytime Algorithm Property — Confidence vs. Cycle Budget",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = out_dir / "fig8_anytime_property.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Convenience: generate all figures ────────────────────────────────────────

def generate_all_figures(
    agg1: pd.DataFrame,
    agg2: pd.DataFrame,
    profiles_cycles: Dict[str, str],
    profiles_memory: Dict[str, str],
    n_star_nc: Dict[str, Optional[int]],
    n_star_nm: Dict[str, Optional[int]],
    out_dir: str = "figures",
) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    paths["fig1"] = fig_delta_vs_cycles(agg1, out)
    paths["fig2"] = fig_rho_vs_cycles(agg1, out)
    paths["fig3"] = fig_rho_heatmap_cycles(agg1, out)
    paths["fig4"] = fig_delta_vs_memory(agg2, out)
    paths["fig5"] = fig_rho_heatmap_memory(agg2, out)
    paths["fig6"] = fig_degradation_profiles(profiles_cycles, profiles_memory, out)
    paths["fig7"] = fig_critical_budget(n_star_nc, n_star_nm, out)
    paths["fig8"] = fig_anytime_property(agg1, out)

    for name, p in paths.items():
        print(f"  {name}: {p}")
    return paths
