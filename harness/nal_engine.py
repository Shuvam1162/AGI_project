"""
nal_engine.py
-------------
Faithful Python reimplementation of the OpenNARS 3.1.x NAL inference engine.

Truth functions ported EXACTLY from:
  org.opennars.inference.TruthFunctions.java  (MIT Licence, OpenNARS authors)
  org.opennars.inference.UtilityFunctions.java

Architecture mirrors OpenNARS 3.1.x:
  - Priority concept bag  (capacity = Nm)
  - Per-concept belief table (revised in-place)
  - Each inference cycle: pick concept → match task × belief → derive
  - Resource pressure: Nc caps cycles; Nm caps how many concepts survive eviction

Supported rules (NAL-1 / NAL-2 inheritance):
  Deduction   <M-->P>, <S-->M>  |-  <S-->P>
  Induction   <M-->P>, <M-->S>  |-  <S-->P>   (= abduction(v2,v1))
  Abduction   <P-->M>, <S-->M>  |-  <S-->P>
  Revision    <S-->P>, <S-->P>  |-  <S-->P>   (same key, different evidence)

Design choices
--------------
HORIZON = 1.0  (OpenNARS default; Parameters.java: public float HORIZON = 1)
Priority decay per cycle = 0.995 (mild forgetting)
Eviction = stochastic among lowest-quintile priorities (matches bag sampling)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ── OpenNARS constants ────────────────────────────────────────────────────────
HORIZON: float = 1.0          # Parameters.HORIZON = 1
MIN_CONFIDENCE: float = 1e-6  # ignore beliefs with effectively zero confidence
PRIORITY_DECAY: float = 0.995  # per-cycle priority decay (mimics forgetting)


# ═══════════════════════════════════════════════════════════════════════════════
# Truth functions — ported verbatim from TruthFunctions.java / UtilityFunctions
# ═══════════════════════════════════════════════════════════════════════════════

def w2c(w: float) -> float:
    """Evidence weight → confidence.  w2c(w) = w / (w + HORIZON)"""
    return w / (w + HORIZON)


def c2w(c: float) -> float:
    """Confidence → evidence weight.  c2w(c) = HORIZON * c / (1 - c)"""
    if c >= 1.0 - 1e-9:
        return 1e9
    return HORIZON * c / (1.0 - c)


def _and(*args: float) -> float:
    """Logical AND: product of all arguments."""
    r = 1.0
    for a in args:
        r *= a
    return r


def truth_deduction(f1: float, c1: float, f2: float, c2: float) -> Tuple[float, float]:
    """
    <M-->P>, <S-->M> |- <S-->P>
    f = f1 * f2
    c = and(c1, c2, f)  =  c1 * c2 * f
    """
    f = f1 * f2
    c = _and(c1, c2, f)
    return f, c


def truth_abduction(f1: float, c1: float, f2: float, c2: float) -> Tuple[float, float]:
    """
    <P-->M>, <S-->M> |- <S-->P>   (given v1=<P-->M>, v2=<S-->M>)
    w = and(f2, c1, c2)
    c = w2c(w)
    f = f1
    """
    w = _and(f2, c1, c2)
    c = w2c(w)
    return f1, c


def truth_induction(f1: float, c1: float, f2: float, c2: float) -> Tuple[float, float]:
    """
    <M-->P>, <M-->S> |- <S-->P>   (induction = abduction with swapped args)
    From source: induction(v1, v2) = abduction(v2, v1)
    """
    return truth_abduction(f2, c2, f1, c1)


def truth_revision(f1: float, c1: float, f2: float, c2: float) -> Tuple[float, float]:
    """
    {<S-->P>, <S-->P>} |- <S-->P>
    Weighted average by evidence weight.
    w1 = c2w(c1), w2 = c2w(c2), w = w1+w2
    f  = (w1*f1 + w2*f2) / w
    c  = w2c(w)
    """
    w1 = c2w(c1)
    w2 = c2w(c2)
    w  = w1 + w2
    f  = (w1 * f1 + w2 * f2) / w
    c  = w2c(w)
    return f, c


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Belief:
    subj: str
    pred: str
    f: float
    c: float
    priority: float = 0.8

    def key(self) -> Tuple[str, str]:
        return (self.subj, self.pred)

    def quality(self) -> float:
        """Used to initialise concept priority (mirrors OpenNARS BudgetFunctions)."""
        return math.sqrt(self.f * self.c)


class BeliefBase:
    """
    Stores the best belief for each (subj, pred) key.
    Capacity = Nm: evicts lowest-priority belief when full.
    Revision is applied automatically when the same key is added again.
    """

    def __init__(self, nm: int, rng: random.Random) -> None:
        self.nm  = nm
        self.rng = rng
        self._beliefs: Dict[Tuple[str, str], Belief] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, b: Belief) -> Optional[Belief]:
        """Insert or revise.  Returns the resulting belief."""
        key = b.key()
        if key in self._beliefs:
            old = self._beliefs[key]
            # Evidence independence: only revise when the new belief
            # carries genuinely different evidence (different truth tuple).
            # Prevents the confidence spiral from repeated identical derivations
            # (mirrors OpenNARS evidence-base non-overlap requirement).
            if abs(b.f - old.f) < 0.01 and abs(b.c - old.c) < 0.01:
                return old  # same evidence, skip revision
            f, c  = truth_revision(old.f, old.c, b.f, b.c)
            # Also cap confidence at 0.99 to prevent runaway accumulation
            c = min(c, 0.99)
            new_p = max(old.priority, b.priority)
            revised = Belief(b.subj, b.pred, f, c, new_p)
            self._beliefs[key] = revised
            return revised
        else:
            self._beliefs[key] = b
            if len(self._beliefs) > self.nm:
                self._evict()
            return b

    def find(self, subj: str, pred: str) -> Optional[Belief]:
        return self._beliefs.get((subj, pred))

    def predicates_for(self, subj: str) -> List[Belief]:
        """All beliefs where *subj* is the subject: <subj --> P>."""
        return [b for b in self._beliefs.values() if b.subj == subj]

    def subjects_for(self, pred: str) -> List[Belief]:
        """All beliefs where *pred* is the predicate: <M --> pred>."""
        return [b for b in self._beliefs.values() if b.pred == pred]

    def all(self) -> List[Belief]:
        return list(self._beliefs.values())

    def __len__(self) -> int:
        return len(self._beliefs)

    # ── internal ──────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Stochastically evict from the lowest-priority quintile."""
        if not self._beliefs:
            return
        items = sorted(self._beliefs.values(), key=lambda b: b.priority)
        # candidates = bottom 20% or at least 1 item
        n_cand = max(1, len(items) // 5)
        victim = self.rng.choice(items[:n_cand])
        del self._beliefs[victim.key()]


class ConceptBag:
    """
    Priority bag over term strings.  Mirrors OpenNARS Bag<Concept>.
    Selection is stochastic, weighted by priority (softmax-like).
    Eviction removes the lowest-priority concept when over capacity.
    """

    def __init__(self, nm: int, rng: random.Random) -> None:
        self.nm  = nm
        self.rng = rng
        self._p: Dict[str, float] = {}   # term → priority

    def add(self, term: str, priority: float) -> None:
        if term in self._p:
            # Forgetting function (blends old and new, OpenNARS style)
            self._p[term] = 0.8 * max(self._p[term], priority) + 0.2 * priority
        else:
            self._p[term] = priority
        while len(self._p) > self.nm:
            self._evict()

    def select(self) -> Optional[str]:
        """Stochastic priority-weighted selection."""
        if not self._p:
            return None
        terms = list(self._p.keys())
        weights = [self._p[t] + 1e-6 for t in terms]
        total = sum(weights)
        r = self.rng.uniform(0, total)
        cumsum = 0.0
        for t, w in zip(terms, weights):
            cumsum += w
            if r <= cumsum:
                return t
        return terms[-1]

    def update(self, term: str, delta: float = 0.05) -> None:
        """Boost priority after a concept is fired and produces a derivation."""
        if term in self._p:
            self._p[term] = min(1.0, self._p[term] + delta)

    def decay_all(self) -> None:
        """Mild priority decay each cycle (attention forgetting)."""
        for t in self._p:
            self._p[t] *= PRIORITY_DECAY

    def _evict(self) -> None:
        if not self._p:
            return
        victim = min(self._p, key=lambda t: self._p[t])
        del self._p[victim]


# ═══════════════════════════════════════════════════════════════════════════════
# Main inference engine
# ═══════════════════════════════════════════════════════════════════════════════

class NALEngine:
    """
    Bounded NAL-1/2 inference engine.

    Parameters
    ----------
    nc : int   — inference cycle budget  (Nc)
    nm : int   — concept memory capacity (Nm, applied to both bags)
    seed : int — RNG seed for reproducibility
    """

    def __init__(self, nc: int, nm: int, seed: int = 0) -> None:
        self.nc  = nc
        self.nm  = nm
        self.rng = random.Random(seed)
        self.belief_base  = BeliefBase(nm, self.rng)
        self.concept_bag  = ConceptBag(nm, self.rng)
        self._queries: List[Tuple[str, str]] = []

    # ── public API ────────────────────────────────────────────────────────────

    def input_belief(self, subj: str, pred: str, f: float, c: float) -> None:
        """Feed a Narsese belief: <subj --> pred>. %f;c%"""
        priority = Belief(subj, pred, f, c).quality() * 0.9 + 0.1
        b = Belief(subj, pred, f, c, priority)
        self.belief_base.add(b)
        self.concept_bag.add(subj, priority)
        self.concept_bag.add(pred, priority)

    def input_query(self, subj: str, pred: str) -> None:
        """Register a query: <subj --> pred>?"""
        self._queries.append((subj, pred))
        # Boost attention on query concepts
        for term in (subj, pred):
            self.concept_bag.add(term, 0.9)

    def run(self) -> Dict[Tuple[str, str], Optional[Tuple[float, float]]]:
        """
        Run Nc inference cycles.
        Returns final (f, c) for each registered query, or None if unanswered.
        """
        for _ in range(self.nc):
            concept = self.concept_bag.select()
            if concept is None:
                break
            self._fire(concept)
            self.concept_bag.decay_all()

        return {
            q: (b.f, b.c) if (b := self.belief_base.find(*q)) else None
            for q in self._queries
        }

    # ── inference rules ───────────────────────────────────────────────────────

    def _fire(self, concept: str) -> None:
        """
        Fire *concept*: apply all applicable NAL-1 rules.

        Roles of *concept*:
          M in deduction : <S-->M> × <M-->P>  →  <S-->P>
          M in induction : <M-->P1> × <M-->P2> → <P2-->P1>
          M in abduction : <M1-->P> × <M2-->P> → <M2-->M1>  (P=concept)
        """
        as_pred  = self.belief_base.subjects_for(concept)   # <S --> concept>
        as_subj  = self.belief_base.predicates_for(concept) # <concept --> P>

        produced = False

        # ── Rule 1: Deduction  <S-->M>, <M-->P>  |-  <S-->P> ────────────────
        for s_m in as_pred:
            for m_p in as_subj:
                if s_m.subj == m_p.pred:   # trivial loop guard
                    continue
                f, c = truth_deduction(s_m.f, s_m.c, m_p.f, m_p.c)
                if c > MIN_CONFIDENCE:
                    p = _and(s_m.priority, m_p.priority) * 0.9 + 0.05
                    self._derive(s_m.subj, m_p.pred, f, c, p)
                    produced = True

        # ── Rule 2: Induction  <M-->P1>, <M-->P2>  |-  <P2-->P1> ────────────
        for i, b1 in enumerate(as_subj):
            for b2 in as_subj[i + 1:]:
                if b1.pred == b2.pred:
                    continue
                f, c = truth_induction(b1.f, b1.c, b2.f, b2.c)
                if c > MIN_CONFIDENCE:
                    p = _and(b1.priority, b2.priority) * 0.7
                    self._derive(b2.pred, b1.pred, f, c, p)
                    produced = True
                # Symmetric
                f2, c2 = truth_induction(b2.f, b2.c, b1.f, b1.c)
                if c2 > MIN_CONFIDENCE:
                    self._derive(b1.pred, b2.pred, f2, c2, p)

        # ── Rule 3: Abduction  <M1-->P>, <M2-->P>  |-  <M2-->M1> ────────────
        #           (here P = concept, M1/M2 are subjects in as_pred)
        for i, b1 in enumerate(as_pred):
            for b2 in as_pred[i + 1:]:
                if b1.subj == b2.subj:
                    continue
                f, c = truth_abduction(b1.f, b1.c, b2.f, b2.c)
                if c > MIN_CONFIDENCE:
                    p = _and(b1.priority, b2.priority) * 0.7
                    self._derive(b2.subj, b1.subj, f, c, p)
                    produced = True
                f2, c2 = truth_abduction(b2.f, b2.c, b1.f, b1.c)
                if c2 > MIN_CONFIDENCE:
                    self._derive(b1.subj, b2.subj, f2, c2, p)

        if produced:
            self.concept_bag.update(concept, 0.06)

    def _derive(self, subj: str, pred: str, f: float, c: float, p: float) -> None:
        """Add a derived belief; update concept bag accordingly."""
        b = Belief(subj, pred, f, c, p)
        result = self.belief_base.add(b)
        if result is not None:
            self.concept_bag.add(subj, p)
            self.concept_bag.add(pred, p)
