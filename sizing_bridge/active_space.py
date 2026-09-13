# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/active_space.py at bba1add.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Active-space selection for quantum chemistry, with a bound and a refusal.

The lever in a hybrid chemistry workload is the active space. Choosing
(n_e, n_o) fixes the qubit count, the two-electron integral tensor, the
trial-state dimension and every hour of classical post-processing that
follows — and it is chosen once, by a person, before any circuit exists.

Selection here works from **natural orbital occupation numbers**, the cheap
classical precursor an MP2 or CISD calculation already produces. Orbitals
whose occupation sits away from 0 or 2 are the correlated ones; those pinned
near 2 are inactive core and those near 0 are unoccupied. That much is the
natural-orbital occupation window, which is established prior art and not
ours — see `docs/` and the module tests for the standing citation to AVAS,
occupation-window methods and the DMRG orbital-entropy work.

What this module adds is the part that makes a truncation defensible rather
than merely smaller:

* **A measured bound.** Deviation from integer occupancy, ``min(n, 2 - n)``,
  is zero for a closed or empty orbital and maximal at half filling. Summed
  over all orbitals it is the total correlation mass; summed over the chosen
  window it is what the active space retains. The ratio is reported with every
  selection, so the caller knows what the truncation cost rather than trusting
  that it cost little.

* **A refusal.** When no window inside the caller's orbital budget retains
  enough of that mass, the answer is that the molecule does not support
  reduction at this tolerance. A selector that always returns a smaller number
  is not a guard rail, and for an active space a wrong answer is much more
  expensive than an expensive one.

* **Chemical validity, enforced.** A space is rejected when the electron count
  cannot fill the orbitals, when it fills them exactly (a closed shell is a
  single determinant and carries no correlation at all), or when the electron
  count is odd — closed-shell CASCI determinant counting does not apply there.
  These are easy errors to make on paper: 26 electrons in 13 orbitals looks
  like a reduction and is in fact one determinant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any, Literal

import numpy as np

__all__ = [
    "ranked_spaces",
    "ActiveSpace",
    "SelectionReport",
    "correlation_mass",
    "determinant_count",
    "qubit_count",
    "select_active_space",
]

Verdict = Literal["selected", "refused"]

#: An orbital this close to 0 or 2 contributes no usable correlation.
_INTEGER_TOL = 1e-9


@dataclass(frozen=True)
class ActiveSpace:
    """A chosen (n_e, n_o) space and what it costs to run."""

    n_electrons: int
    n_orbitals: int
    orbital_indices: tuple[int, ...]

    @property
    def qubits(self) -> int:
        """Spin-orbitals, which is the qubit count under a standard mapping."""
        return qubit_count(self.n_orbitals)

    @property
    def n_determinants(self) -> int:
        return determinant_count(self.n_electrons, self.n_orbitals)

    @property
    def label(self) -> str:
        return f"({self.n_electrons}e, {self.n_orbitals}o)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_electrons": self.n_electrons,
            "n_orbitals": self.n_orbitals,
            "label": self.label,
            "qubits": self.qubits,
            "n_determinants": self.n_determinants,
            "orbital_indices": list(self.orbital_indices),
        }


@dataclass
class SelectionReport:
    """The outcome, the evidence for it, and what it is worth."""

    verdict: Verdict
    space: ActiveSpace | None
    reference: ActiveSpace
    correlation_retained: float
    correlation_tolerance: float
    reason: str
    candidates_evaluated: int = 0
    rejected_for_validity: tuple[str, ...] = field(default_factory=tuple)

    @property
    def selected(self) -> bool:
        return self.verdict == "selected"

    @property
    def reach_multiplier(self) -> float:
        """How much further a fixed qubit budget reaches at this size."""
        if self.space is None:
            return 1.0
        return self.reference.qubits / self.space.qubits

    def cost_multiplier(self, exponent: float) -> float:
        """Post-processing cost relative to the reference, at the caller's exponent.

        The exponent belongs to the caller's workload — for a published
        QC-AFQMC post-processing step it is that paper's scaling. This is
        arithmetic on their number, not a claim about it.
        """
        if exponent <= 0:
            raise ValueError("exponent must be positive")
        if self.space is None:
            return 1.0
        return (self.space.qubits / self.reference.qubits) ** exponent

    def speedup(self, exponent: float) -> float:
        multiplier = self.cost_multiplier(exponent)
        return 1.0 / multiplier if multiplier > 0 else float("inf")

    def determinant_reduction(self) -> float:
        """How many times smaller the CASCI determinant space became."""
        if self.space is None:
            return 1.0
        return self.reference.n_determinants / max(self.space.n_determinants, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "space": self.space.as_dict() if self.space else None,
            "reference": self.reference.as_dict(),
            "correlation_retained": round(self.correlation_retained, 6),
            "correlation_tolerance": self.correlation_tolerance,
            "reach_multiplier": round(self.reach_multiplier, 4),
            "determinant_reduction": round(self.determinant_reduction(), 2),
            "candidates_evaluated": self.candidates_evaluated,
            "rejected_for_validity": list(self.rejected_for_validity),
            "reason": self.reason,
        }


# ── Quantities ───────────────────────────────────────────────────────────────

def qubit_count(n_orbitals: int) -> int:
    """Spin-orbitals for a spatial-orbital count: the qubit requirement."""
    if n_orbitals < 0:
        raise ValueError("n_orbitals must be non-negative")
    return 2 * n_orbitals


def determinant_count(n_electrons: int, n_orbitals: int) -> int:
    """Closed-shell CASCI determinant count, C(n_o, n_e/2)².

    Raises on an invalid space rather than returning a number that looks
    plausible. An odd electron count has no closed-shell counting, and an
    electron count the orbitals cannot hold is not a space at all.
    """
    if n_orbitals <= 0:
        raise ValueError("n_orbitals must be positive")
    if n_electrons < 0:
        raise ValueError("n_electrons must be non-negative")
    if n_electrons % 2:
        raise ValueError(
            f"closed-shell counting needs an even electron count, got {n_electrons}"
        )
    if n_electrons > 2 * n_orbitals:
        raise ValueError(
            f"{n_electrons} electrons cannot occupy {n_orbitals} orbitals"
        )
    return comb(n_orbitals, n_electrons // 2) ** 2


def correlation_mass(occupations: np.ndarray) -> np.ndarray:
    """Per-orbital deviation from integer occupancy: min(n, 2 − n).

    Zero for a closed or empty orbital, maximal at half filling. This is the
    quantity the selection budget is denominated in.
    """
    occ = np.asarray(occupations, dtype=float)
    if occ.ndim != 1:
        raise ValueError("occupations must be 1-D")
    if np.any(occ < -_INTEGER_TOL) or np.any(occ > 2.0 + _INTEGER_TOL):
        raise ValueError("occupations must lie in [0, 2]")
    return np.minimum(np.clip(occ, 0.0, 2.0), 2.0 - np.clip(occ, 0.0, 2.0))


def _validity_error(n_electrons: int, n_orbitals: int) -> str | None:
    """Why this (n_e, n_o) is not a usable space, or None if it is."""
    if n_orbitals <= 0:
        return "no orbitals in the window"
    if n_electrons <= 0:
        return f"({n_electrons}e, {n_orbitals}o): no electrons to correlate"
    if n_electrons % 2:
        return (
            f"({n_electrons}e, {n_orbitals}o): odd electron count has no "
            "closed-shell determinant counting"
        )
    if n_electrons > 2 * n_orbitals:
        return (
            f"({n_electrons}e, {n_orbitals}o): {n_electrons} electrons cannot "
            f"occupy {n_orbitals} orbitals"
        )
    if n_electrons == 2 * n_orbitals:
        return (
            f"({n_electrons}e, {n_orbitals}o): closed shell — one determinant, "
            "no correlation to recover"
        )
    return None


# ── Selection ────────────────────────────────────────────────────────────────

def ranked_spaces(
    occupations: np.ndarray,
    *,
    max_orbitals: int | None = None,
    min_orbitals: int = 2,
) -> list[ActiveSpace]:
    """Every nested space the ranking implies, one per size, smallest first.

    `select_active_space` answers "which space clears this tolerance"; this
    answers "what is the whole sequence". They share the ranking, so the
    spaces returned here are nested — each is a subset of the next — which is
    the property a convergence argument needs and a tolerance ladder does not
    guarantee, because a ladder can jump several orbitals between rungs and
    leave the sequence too coarse to tell convergence from a big step.

    Sizes that are not chemically valid are skipped rather than returned, on
    the same grounds `select_active_space` rejects them.
    """
    occ = np.asarray(occupations, dtype=float)
    mass = correlation_mass(occ)
    order = np.lexsort((np.arange(occ.size), -mass))
    correlated = [int(i) for i in order if mass[i] > _INTEGER_TOL]
    if not correlated:
        return []

    ceiling = min(len(correlated), max_orbitals or len(correlated))
    spaces: list[ActiveSpace] = []
    for size in range(min_orbitals, ceiling + 1):
        candidate = _space_from_indices(occ, tuple(sorted(correlated[:size])))
        if _validity_error(candidate.n_electrons, candidate.n_orbitals) is None:
            spaces.append(candidate)
    return spaces


def select_active_space(
    occupations: np.ndarray,
    *,
    correlation_tolerance: float = 0.95,
    max_orbitals: int | None = None,
    min_orbitals: int = 2,
    reference_space: ActiveSpace | None = None,
) -> SelectionReport:
    """Choose the smallest active space retaining enough correlation mass.

    ``occupations`` are natural orbital occupation numbers in [0, 2], one per
    spatial orbital, in any order.

    ``reference_space`` is what the selection is compared against. Pass the
    space a run actually used — the one a chemist chose by hand — and reach and
    cost are reported against that choice, which is the comparison that decides
    whether the selection was worth making. Omit it and the reference defaults
    to every orbital carrying any correlation at all, which answers a narrower
    question: how much of the correlated set was needed. The two differ
    whenever a space was rounded up, which is the usual case and the reason
    this parameter exists.

    The search walks candidate window sizes from smallest upward and returns
    the first that both (a) retains at least ``correlation_tolerance`` of the
    total correlation mass and (b) is a chemically valid space. Spaces failing
    (b) are recorded in the report rather than skipped silently: a reader
    should be able to see that 26 electrons in 13 orbitals was considered and
    rejected as a closed shell.
    """
    occ = np.asarray(occupations, dtype=float)
    mass = correlation_mass(occ)
    total_mass = float(mass.sum())

    # Orbitals ranked by how much correlation they carry; ties broken by index
    # so the selection is deterministic for a given input.
    order = np.lexsort((np.arange(occ.size), -mass))

    correlated = [int(i) for i in order if mass[i] > _INTEGER_TOL]
    derived_reference = (
        _space_from_indices(occ, tuple(sorted(correlated))) if correlated else None
    )
    reference = reference_space or derived_reference

    if reference is None or total_mass <= _INTEGER_TOL:
        reference = reference_space
        empty = reference or ActiveSpace(0, max(occ.size, 1), tuple(range(occ.size)))
        return SelectionReport(
            verdict="refused",
            space=None,
            reference=empty,
            correlation_retained=0.0,
            correlation_tolerance=correlation_tolerance,
            reason=(
                "no orbital deviates from integer occupancy; the reference "
                "determinant already describes this system and there is no "
                "correlation for an active space to capture"
            ),
        )

    ceiling = min(len(correlated), max_orbitals or len(correlated))
    rejected: list[str] = []
    evaluated = 0

    for size in range(min_orbitals, ceiling + 1):
        evaluated += 1
        indices = tuple(sorted(correlated[:size]))
        retained = float(mass[list(indices)].sum()) / total_mass
        if retained < correlation_tolerance:
            continue

        candidate = _space_from_indices(occ, indices)
        problem = _validity_error(candidate.n_electrons, candidate.n_orbitals)
        if problem is not None:
            rejected.append(problem)
            continue

        return SelectionReport(
            verdict="selected",
            space=candidate,
            reference=reference,
            correlation_retained=retained,
            correlation_tolerance=correlation_tolerance,
            candidates_evaluated=evaluated,
            rejected_for_validity=tuple(rejected),
            reason=(
                f"{candidate.label} at {candidate.qubits} qubits retains "
                f"{retained:.1%} of the correlation mass, at or above the "
                f"{correlation_tolerance:.0%} the caller required"
            ),
        )

    return SelectionReport(
        verdict="refused",
        space=None,
        reference=reference,
        correlation_retained=(
            float(mass[[i for i in _indices_of(reference) if i < mass.size]].sum())
            / total_mass
        ),
        correlation_tolerance=correlation_tolerance,
        candidates_evaluated=evaluated,
        rejected_for_validity=tuple(rejected),
        reason=(
            f"no valid space within {ceiling} orbitals retained "
            f"{correlation_tolerance:.0%} of the correlation mass; this "
            "molecule does not support reduction at this tolerance"
        ),
    )


def _indices_of(space: ActiveSpace) -> tuple[int, ...]:
    return space.orbital_indices


def _space_from_indices(occ: np.ndarray, indices: tuple[int, ...]) -> ActiveSpace:
    """Build a space from chosen orbitals, rounding electrons to even.

    The occupations of the chosen orbitals sum to a non-integer in general.
    Rounding to the nearest even count is what makes it a closed-shell CASCI
    space; the rounding is reported through the validity checks rather than
    hidden, because rounding up into a closed shell is exactly the failure
    this module exists to catch.
    """
    if not indices:
        return ActiveSpace(0, 0, ())
    electrons = float(occ[list(indices)].sum())
    n_e = int(2 * round(electrons / 2.0))
    return ActiveSpace(n_e, len(indices), indices)
