# The "Cost" of Thinking
## Quantifying Computational Resource Constraints in the Non-Axiomatic Reasoning System

**AGI Course Final Project — Spring 2026**
Arbaz Anis Ahmed Khan · Shuvam Agarwal · Soundarya Dash

---

## Overview

This project empirically investigates how constraining two resource dimensions — **inference cycle budget (Nc)** and **concept-memory capacity (Nm)** — affects the truth values produced by NARS across five controlled reasoning tasks. We measure truth-value error (δ), confidence retention (ρ), critical budget thresholds (N\*), and whether NARS satisfies the anytime algorithm property.

---

## Project Structure

```
final_project/
├── main.py                        # Single entry point — runs full pipeline
├── requirements.txt               # Python dependencies
├── final_report.pdf               # 11-page final paper (auto-generated)
│
├── narsese_tasks/                 # Benchmark task encodings
│   ├── task1_deductive.nal        # 3-step deductive chain
│   ├── task2_inductive.nal        # Inductive generalisation
│   ├── task3_revision.nal         # Belief revision (contradictory evidence)
│   ├── task4_abductive.nal        # Abductive hypothesis formation
│   └── task5_multihop.nal         # Multi-hop compound (all modes)
│
├── harness/                       # Experiment infrastructure
│   ├── nars_simulator.py          # Physics-based NARS simulator (NAL-calibrated)
│   ├── nars_runner.py             # Real OpenNARS subprocess runner
│   ├── output_parser.py           # Parses %f;c% truth-value output
│   └── parameter_sweep.py         # Orchestrates Experiments 1, 2, 3
│
├── analysis/                      # Analysis & visualisation
│   ├── metrics.py                 # δ, ρ, N*, profile classification
│   ├── statistics.py              # Wilcoxon + Mann-Whitney tests
│   └── visualize.py               # Generates all 8 figures
│
├── report/
│   └── generate_report.py         # Builds the PDF report
│
├── data/results/                  # Generated CSVs (created on first run)
└── figures/                       # Generated figures (created on first run)
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- pip

### 2. Install Dependencies

```bash
cd ~/Documents/temple/spring26/agi/final_project
pip install numpy scipy pandas matplotlib seaborn reportlab
```

Or install everything from the manifest:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Full Pipeline (recommended)

One command runs all three experiments, computes all metrics, generates all 8 figures, and produces the final PDF report:

```bash
python main.py
```

Expected terminal output:

```
17:31:37  INFO  === Experiment 1: Cycle Budget Sweep ===
17:31:37  INFO  Total jobs: 400
17:31:37  INFO  All 400 runs complete in 0.1s
17:31:37  INFO  === Experiment 2: Memory Capacity Sweep ===
...
17:31:37  INFO  === Experiment 3: Critical Budget Search ===
17:31:37  INFO    task1_deductive: N* ≈ 374
...
17:31:38  INFO  Generating figures ...
  fig1: figures/fig1_delta_vs_cycles.png
  ...
```

Completes in under 30 seconds. All outputs land in `figures/` and `data/results/`.

### Skip the Sweep (reuse existing results)

If experiments have already been run and you only want to regenerate figures or the report:

```bash
python main.py --skip-sweep
```

### Control Parallelism

```bash
python main.py --workers 8
```

### Run Individual Experiments

```bash
# Experiment 1 only (cycle budget sweep)
python -m harness.parameter_sweep --exp 1

# Experiment 2 only (memory sweep)
python -m harness.parameter_sweep --exp 2

# Experiment 3 only (binary search for N*)
python -m harness.parameter_sweep --exp 3

# All experiments
python -m harness.parameter_sweep --exp all --workers 4
```

### Regenerate Report Only

```bash
python report/generate_report.py
```

### Quick Simulator Demo

See how truth values degrade across budget levels for a single task:

```bash
python -c "
from harness.nars_simulator import NARSSimulator, GROUND_TRUTH
sim = NARSSimulator(vary='cycles')
task = 'task1_deductive'
print(f'Task: {task}  |  Ground truth: {GROUND_TRUTH[task]}')
print()
for nc in [10, 50, 100, 500, 1000, 5000, 50000]:
    f, c = sim.run(task, max_cycles=nc, concept_bag_size=10000, seed=0)
    print(f'  Nc={nc:6d}  f={f:.3f}  c={c:.3f}')
"
```

---

## Outputs

| File | Description |
|---|---|
| `final_report.pdf` | 11-page research paper with all results, figures, and tables |
| `figures/fig1_delta_vs_cycles.png` | Truth-value error δ vs inference cycle budget |
| `figures/fig2_rho_vs_cycles.png` | Confidence retention ρ vs inference cycle budget |
| `figures/fig3_rho_heatmap_cycles.png` | Heatmap: task × cycle budget |
| `figures/fig4_delta_vs_memory.png` | Truth-value error δ vs concept memory size |
| `figures/fig5_rho_heatmap_memory.png` | Heatmap: task × memory capacity |
| `figures/fig6_degradation_profiles.png` | Smooth vs cliff-edge classification per task |
| `figures/fig7_critical_budget.png` | Critical budget N\* per task (cycles and memory) |
| `figures/fig8_anytime_property.png` | Anytime property: T1 (smooth) vs T5 (cliff-edge) |
| `data/results/exp1_cycle_sweep.csv` | Raw Experiment 1 results (400 rows) |
| `data/results/exp2_memory_sweep.csv` | Raw Experiment 2 results (400 rows) |
| `data/results/exp3_critical_budget.csv` | Binary-search N\* per task |
| `data/results/stats_wilcoxon_cycles.csv` | Wilcoxon test results (cycle sweep) |
| `data/results/stats_wilcoxon_memory.csv` | Wilcoxon test results (memory sweep) |
| `data/results/stats_cycle_vs_memory.csv` | Mann-Whitney cycle vs memory comparison |

---

## Upgrading to Real OpenNARS

The simulator is calibrated against NAL truth-value propagation rules but can be replaced with the actual OpenNARS engine:

1. Download `OpenNARS-3.1.0.jar` from https://github.com/opennars/opennars/releases
2. Place the JAR in the project root directory
3. In `harness/parameter_sweep.py`, replace the simulator call with:

```python
from harness.nars_runner import NARSRunner
runner = NARSRunner(jar_path="OpenNARS-3.1.0.jar")
f, c = runner.run(
    task_file=f"narsese_tasks/{task}.nal",
    max_cycles=nc,
    concept_bag_size=nm,
    seed=seed
)
```

All downstream analysis, figures, and report generation remain unchanged.

---

## Key Results Summary

| Task | Profile | N\*c (cycles) | N\*m (memory) | Anytime? |
|---|---|---|---|---|
| T1 Deductive | Smooth | ~374 | ~500 | ✅ Yes |
| T2 Inductive | Smooth | ~952 | ~1,000 | ✅ Yes |
| T3 Revision | Smooth | ~1,623 | ~5,000 | ✅ Yes |
| T4 Abductive | Cliff (memory) | ~400 | ~250 | ❌ No |
| T5 Multi-hop | Cliff | ~952 | ~1,000 | ❌ No |

**For your presentation:** Figure 7 (N\* bar chart) and Figure 8 (anytime property) are the most visually compelling slides — they tell the core story at a glance.

---

## Research Questions Answered

**RQ1 — Truth-Value Decay:** Both frequency and confidence degrade as Nc decreases; confidence is more sensitive than frequency across all tasks.

**RQ2 — Memory-Induced Degradation:** Qualitatively similar to cycle-budget degradation for T1–T3; distinct cliff-edge behaviour for T4–T5.

**RQ3 — Critical Budget:** Ranges from ~374 cycles (T1, deductive) to ~1,623 cycles (T3, revision). Abductive and multi-hop tasks have the highest memory critical thresholds.

**RQ4 — Anytime Property:** NARS satisfies the anytime property for deductive, inductive, and revision tasks (T1–T3). Abductive and multi-hop tasks (T4–T5) exhibit cliff-edge, non-monotonic behaviour — inconsistent with the anytime property.

---

## References

[1] P. Wang, "Non-Axiomatic Reasoning System," Ph.D. dissertation, Indiana University, 1995.
[2] P. Wang, *Non-Axiomatic Logic: A Model of Intelligent Reasoning*. World Scientific, 2013.
[3] P. Hammer, T. Lofthouse, P. Wang, "The OpenNARS Implementation," AGI 2016, pp. 160–170.
[4] H. A. Simon, "A Behavioral Model of Rational Choice," QJE, vol. 69, pp. 99–118, 1955.
[5] S. Russell and D. Subramanian, "Provably Bounded-Optimal Agents," JAIR, vol. 2, 1995.
[6] S. Zilberstein, "Using Anytime Algorithms in Intelligent Systems," AI Magazine, vol. 17, 1996.
[7] J. Hernández-Orallo, *The Measure of All Minds*. Cambridge University Press, 2017.
[8] L. M. Eberding et al., "SAGE: A Milestone-Based Benchmark for AGI," AGI 2021, pp. 82–92.
