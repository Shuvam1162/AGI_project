"""
narsese_parser.py
-----------------
Minimal Narsese (.nal) parser for the five benchmark tasks.

Handles the subset of NAL-1/2 syntax used in our task files:
  <A --> B>. %{f c}          →  Belief(subj, pred, f, c)
  <A --> B>. %f;c%           →  Belief(subj, pred, f, c)   (alternate format)
  <A --> B>.                 →  Belief(subj, pred, 1.0, 0.9) (defaults)
  <A --> B>?                 →  Query(subj, pred)
  % ...                      →  comment, ignored
  // ...                     →  comment, ignored
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ParsedBelief:
    subj: str
    pred: str
    f: float
    c: float


@dataclass
class ParsedQuery:
    subj: str
    pred: str


# Regex patterns
_RE_STMT   = re.compile(r"<\s*(\S+)\s+-->\s+(\S+)\s*>")
_RE_TRUTH1 = re.compile(r"%\{?\s*([\d.]+)\s+[\s,;]?\s*([\d.]+)\s*\}?%?")  # %{f c}%
_RE_TRUTH2 = re.compile(r"%\s*([\d.]+)\s*[;,]\s*([\d.]+)\s*%")             # %f;c%
_RE_TRUTH3 = re.compile(r"%\s*([\d.]+)\s*%")                                # %f%  (confidence omitted)


def parse_file(path: str) -> Tuple[List[ParsedBelief], List[ParsedQuery]]:
    """Parse a .nal file and return (beliefs, queries)."""
    beliefs: List[ParsedBelief] = []
    queries: List[ParsedQuery]  = []

    with open(path, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blank lines and comments
            if not line or line.startswith("%") or line.startswith("//"):
                continue

            # Strip inline comments
            if "//" in line:
                line = line[:line.index("//")].strip()
            if line.startswith("%") and not re.search(r"<.+-->", line):
                continue

            stmt_match = _RE_STMT.search(line)
            if stmt_match is None:
                continue

            subj = stmt_match.group(1)
            pred = stmt_match.group(2)

            # Is it a query?
            after_stmt = line[stmt_match.end():]
            if "?" in after_stmt or line.rstrip().endswith("?"):
                queries.append(ParsedQuery(subj, pred))
                continue

            # Extract truth value
            f, c = _extract_truth(after_stmt)
            beliefs.append(ParsedBelief(subj, pred, f, c))

    return beliefs, queries


def _extract_truth(text: str) -> Tuple[float, float]:
    """Parse truth value from the string following a statement."""
    # Pattern %{f c}%  or  %{f, c}
    m = _RE_TRUTH1.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Pattern %f;c%
    m = _RE_TRUTH2.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Pattern %f%  (confidence defaults to 0.9)
    m = _RE_TRUTH3.search(text)
    if m:
        return float(m.group(1)), 0.9

    # No truth value — use OpenNARS defaults (f=1.0, c=0.9)
    return 1.0, 0.9
