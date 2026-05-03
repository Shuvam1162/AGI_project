"""
opennars_runner.py
------------------
Runs a Narsese task through the faithful NAL engine with specified resource
limits, returning the engine's answer for the primary query in the .nal file.

The primary query is always the FIRST query found in the .nal file.
If the engine cannot derive an answer, returns (0.5, 0.01) — NARS default
for an unanswered question (maximum uncertainty, near-zero confidence).
"""

from __future__ import annotations
import os
from typing import Optional, Tuple

from harness.nal_engine     import NALEngine
from harness.narsese_parser import parse_file

# ── Paths ─────────────────────────────────────────────────────────────────────
_TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "narsese_tasks")

# Answer returned when engine produces no derivation for the query
_NO_ANSWER: Tuple[float, float] = (0.5, 0.01)


def run_task(
    task_name: str,
    nc:   int,
    nm:   int,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Load *task_name*.nal, run the NAL engine with budget (nc, nm, seed),
    and return (f, c) for the primary query.

    Parameters
    ----------
    task_name : one of task1_deductive … task5_multihop
    nc        : inference cycle budget (Nc)
    nm        : concept memory capacity (Nm)
    seed      : RNG seed (for reproducibility)

    Returns
    -------
    (f, c) — truth value of the primary query answer, or (0.5, 0.01) if none.
    """
    path = os.path.join(_TASK_DIR, f"{task_name}.nal")
    beliefs, queries = parse_file(path)

    if not queries:
        return _NO_ANSWER

    engine = NALEngine(nc=nc, nm=nm, seed=seed)

    # Feed all beliefs into the engine
    for b in beliefs:
        engine.input_belief(b.subj, b.pred, b.f, b.c)

    # Register all queries; primary = first in file
    primary = (queries[0].subj, queries[0].pred)
    registered: set = set()
    for q in queries:
        key = (q.subj, q.pred)
        if key not in registered:
            engine.input_query(q.subj, q.pred)
            registered.add(key)

    results = engine.run()

    answer = results.get(primary)
    return answer if answer is not None else _NO_ANSWER
