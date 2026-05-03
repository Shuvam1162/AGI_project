"""
output_parser.py
----------------
Parse OpenNARS CLI output to extract truth values <f, c> from derived beliefs.

OpenNARS outputs lines like:
  Answer: <A --> D>. %1.00;0.73%
  OUT: <bird --> canFly>. %0.80;0.45%

This module parses those lines and returns (frequency, confidence) tuples.
"""

import re
from typing import Optional, Tuple, List

# Regex for NARS truth-value output:  %f;c%  where f,c in [0,1]
_TRUTH_RE = re.compile(r'%([01]?\.\d+);([01]?\.\d+)%')

# Regex for full answer line
_ANSWER_RE = re.compile(
    r'(?:Answer|OUT|answer):\s*(.+?)\.\s*%([01]?\.\d+);([01]?\.\d+)%'
)


def parse_truth_value(line: str) -> Optional[Tuple[float, float]]:
    """
    Extract (frequency, confidence) from a single NARS output line.
    Returns None if no truth value found.
    """
    m = _TRUTH_RE.search(line)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_output_block(output: str) -> List[Tuple[str, float, float]]:
    """
    Parse an entire NARS output block and return a list of
    (statement, frequency, confidence) triples for all answer lines.
    """
    results = []
    for line in output.splitlines():
        m = _ANSWER_RE.search(line)
        if m:
            statement = m.group(1).strip()
            f = float(m.group(2))
            c = float(m.group(3))
            results.append((statement, f, c))
    return results


def extract_best_answer(output: str) -> Optional[Tuple[float, float]]:
    """
    From a NARS output block, return the (f, c) of the last Answer line,
    which is typically the most-revised belief for the queried statement.
    Returns (0.5, 0.0) if no answer found (null / no revision).
    """
    results = parse_output_block(output)
    if results:
        _, f, c = results[-1]
        return f, c
    # NARS produced no answer for the query → treat as null belief
    return 0.5, 0.0


if __name__ == "__main__":
    sample = """
    OUT: <A --> B>. %1.00;0.90%
    Answer: <A --> D>. %1.00;0.73%
    """
    print(parse_output_block(sample))
    print(extract_best_answer(sample))
