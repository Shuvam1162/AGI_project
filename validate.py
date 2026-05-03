"""
validate.py
-----------
Entry point for the live NAL engine validation pipeline.

Usage
-----
  python validate.py [--workers N] [--skip-sweep]

  --workers N     parallelism for the sweep (default: 4)
  --skip-sweep    skip sweep if data/validation_nc.csv already exists

Steps
-----
  1. Run 800+ inference tasks through the faithful NAL engine (validation_sweep)
  2. Load simulator results (from data/results_nc.csv)
  3. Generate comparison figures and summary table   (validation_compare)
  4. Print final agreement verdict

The NAL engine is built from the OpenNARS 3.1.x source
(org.opennars.inference.TruthFunctions) — exact truth functions, stochastic
priority concept bag, and identical resource parameters (HORIZON=1, Nc, Nm).
"""

import argparse
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from harness.validation_sweep   import run_validation_sweep
from analysis.validation_compare import run_validation_compare


def main() -> None:
    parser = argparse.ArgumentParser(description="NAL engine validation pipeline")
    parser.add_argument("--workers",    type=int, default=4,
                        help="parallel workers for sweep (default: 4)")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="skip sweep if validation_nc.csv already exists")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    val_nc   = os.path.join(data_dir, "validation_nc.csv")

    print("=" * 70)
    print("  OpenNARS Validation — Faithful NAL Engine vs Calibrated Simulator")
    print("=" * 70)

    t_start = time.time()

    # ── Step 1: Sweep ─────────────────────────────────────────────────────────
    if args.skip_sweep and os.path.exists(val_nc):
        print("\n  [--skip-sweep] Using existing validation_nc.csv")
    else:
        run_validation_sweep(workers=args.workers)

    # ── Step 2: Compare & visualise ───────────────────────────────────────────
    run_validation_compare()

    elapsed = time.time() - t_start
    print(f"  Total elapsed: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
