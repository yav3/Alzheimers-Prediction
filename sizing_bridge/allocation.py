# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/allocation.py at bba1add.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Spending a finite shot budget where it changes the answer.

The claim this package is built around is that a governed layer decides where
the next expensive evaluation goes. Everything before this module decides how
*large* a problem is; this one decides how a fixed measurement budget is spent
once the problem is fixed. They are different levers and both are needed: a
right-sized job measured uniformly still wastes most of its shots.

The estimator is a weighted sum of group expectations, E = Σ_g c_g ⟨P_g⟩, and
the variance of the estimate under an allocation {S_g} is

    Var = Σ_g c_g² σ_g² / S_g          with  Σ_g S_g = S.

Minimising that under the budget constraint is a Lagrange problem with the
classical answer

    S_g ∝ |c_g| σ_g                     (Neyman allocation)

and the resulting standard error is (Σ_g |c_g| σ_g) / √S, against the uniform
allocation's √(G · Σ_g c_g² σ_g²) / √S. The ratio between them is the honest
statement of what the allocation buys, and `AllocationReport` reports it rather
than asserting a speedup.

Two properties are enforced because both are places this goes quietly wrong.
Every group with a non-negligible coefficient receives at least ``min_shots``,
so the allocation cannot silently drop a term whose variance was
under-estimated from a small pilot. And the integer rounding is done by largest
remainder, so the shots handed out sum to exactly the budget — a loop that
rounds each group independently is off by a few thousand shots on a real
Hamiltonian and nobody notices.

Variances are the caller's. Supply them from a pilot run, from a previous
campaign, or from a bound; where they come from decides how good the allocation
is, and this module reports which source it was given rather than pretending
the number is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "AllocationReport",
    "MeasurementGroup",
    "allocate_shots",
    "uniform_allocation",
]

VarianceSource = Literal["pilot", "prior_campaign", "bound", "assumed"]


@dataclass(frozen=True)
class MeasurementGroup:
    """One commuting group of Hamiltonian terms, measurable in a single basis.

    ``coefficient`` is the group's weight in the estimator; ``variance`` is the
    per-shot variance of its measured expectation. For a Pauli expectation
    bounded in [-1, 1] the variance is at most 1, which is the default bound
    when nothing better is known.
    """

    label: str
    coefficient: float
    variance: float = 1.0
    source: VarianceSource = "assumed"

    def __post_init__(self) -> None:
        if self.variance < 0:
            raise ValueError(f"{self.label}: variance must be non-negative")
        if not math.isfinite(self.coefficient) or not math.isfinite(self.variance):
            raise ValueError(f"{self.label}: coefficient and variance must be finite")

    @property
    def weight(self) -> float:
        """|c| σ — the quantity optimal allocation is proportional to."""
        return abs(self.coefficient) * math.sqrt(self.variance)


@dataclass
class AllocationReport:
    """How the budget was spent, and what that bought against measuring evenly."""

    shots: dict[str, int]
    total_shots: int
    standard_error: float
    uniform_standard_error: float
    min_shots: int
    groups_at_floor: list[str] = field(default_factory=list)
    variance_sources: dict[str, int] = field(default_factory=dict)

    @property
    def variance_reduction(self) -> float:
        """How many times smaller the variance is than measuring uniformly.

        Variance, not standard error — the shot count needed for a target
        precision scales with variance, so this is the figure that translates
        into machine time.
        """
        if self.standard_error <= 0:
            return 1.0
        return (self.uniform_standard_error / self.standard_error) ** 2

    @property
    def shots_saved_for_equal_precision(self) -> int:
        """Shots a uniform allocation would need to match this precision."""
        return int(round(self.total_shots * self.variance_reduction)) - self.total_shots

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_shots": self.total_shots,
            "standard_error": self.standard_error,
            "uniform_standard_error": self.uniform_standard_error,
            "variance_reduction": round(self.variance_reduction, 6),
            "shots_saved_for_equal_precision": self.shots_saved_for_equal_precision,
            "min_shots": self.min_shots,
            "groups_at_floor": list(self.groups_at_floor),
            "variance_sources": dict(self.variance_sources),
            "shots": dict(self.shots),
        }


def _standard_error(groups: list[MeasurementGroup], shots: dict[str, int]) -> float:
    """√(Σ c² σ² / S) for a given allocation; a starved group is infinite."""
    total = 0.0
    for g in groups:
        s = shots.get(g.label, 0)
        if s <= 0:
            if abs(g.coefficient) > 0 and g.variance > 0:
                return math.inf
            continue
        total += (g.coefficient**2) * g.variance / s
    return math.sqrt(total)


def uniform_allocation(groups: list[MeasurementGroup], budget: int) -> dict[str, int]:
    """Split the budget evenly — the baseline the optimal one is measured against."""
    if not groups:
        return {}
    base, remainder = divmod(budget, len(groups))
    shots = {g.label: base for g in groups}
    for g in groups[:remainder]:
        shots[g.label] += 1
    return shots


def allocate_shots(
    groups: list[MeasurementGroup],
    budget: int,
    *,
    min_shots: int = 1,
) -> AllocationReport:
    """Allocate ``budget`` shots across ``groups`` to minimise estimator variance.

    Returns the allocation with the standard error it achieves and the error a
    uniform split of the same budget would have achieved, so the caller can see
    what the decision was worth instead of being told.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if min_shots < 0:
        raise ValueError("min_shots must be non-negative")
    if not groups:
        raise ValueError("no measurement groups supplied")

    labels = [g.label for g in groups]
    if len(set(labels)) != len(labels):
        raise ValueError("measurement group labels must be unique")

    if min_shots * len(groups) > budget:
        raise ValueError(
            f"budget {budget} cannot give {min_shots} shots to each of "
            f"{len(groups)} groups; raise the budget or lower min_shots"
        )

    weights = np.array([g.weight for g in groups], dtype=float)
    total_weight = float(weights.sum())

    if total_weight <= 0:
        # Every group is a constant or has zero variance: nothing to optimise,
        # and spreading evenly is the only defensible answer.
        shots = uniform_allocation(groups, budget)
    else:
        floor_total = min_shots * len(groups)
        free = budget - floor_total
        exact = weights / total_weight * free
        base = np.floor(exact).astype(int)
        # Largest-remainder rounding, so the parts sum to the whole. Rounding
        # each group independently leaves the budget unspent.
        remainder = free - int(base.sum())
        order = np.argsort(-(exact - base))
        for i in range(remainder):
            base[order[i % len(groups)]] += 1
        shots = {g.label: int(base[i]) + min_shots for i, g in enumerate(groups)}

    at_floor = [g.label for g in groups if shots[g.label] <= min_shots and g.weight > 0]
    sources: dict[str, int] = {}
    for g in groups:
        sources[g.source] = sources.get(g.source, 0) + 1

    return AllocationReport(
        shots=shots,
        total_shots=int(sum(shots.values())),
        standard_error=_standard_error(groups, shots),
        uniform_standard_error=_standard_error(groups, uniform_allocation(groups, budget)),
        min_shots=min_shots,
        groups_at_floor=at_floor,
        variance_sources=sources,
    )
