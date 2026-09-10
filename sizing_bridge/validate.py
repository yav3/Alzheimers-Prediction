# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/validate.py at 55e18a0.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Does the smaller space still give the right answer? Measure it, then say so.

The whole value of choosing an active space is that the smaller problem
answers the same question as the larger one. That claim is checkable and it is
checkable classically: solve the projected Hamiltonian exactly and compare
against a reference energy. No device is involved, and none would help — the
reference is a classical FCI or CCSD number either way.

This module turns that check into an artifact. `certify_reduction` walks the
selection tolerance upward, solves each candidate space, and returns the
smallest space whose energy sits within the threshold of the reference,
together with every candidate it rejected on the way. A caller gets the space
*and* the evidence, which is the difference between a recommendation and a
certificate.

**It refuses rather than guesses.** With no reference energy and a full space
too large to solve, there is nothing to certify against, and the report says
so instead of returning the selection with an implied blessing. A space that
was never checked is reported as unchecked.

**What the sweep measured.** On four molecules with exact FCI references
(H2O/STO-3G, N2/STO-3G, stretched N2/STO-3G at 2.1 Å, LiH/6-31G), a
correlation tolerance of 0.95 preserved the energy on one of four. 0.99
preserved three of four; the stretched-N2 case — the strongly correlated one,
where MP2 itself errs by 0.4 Hartree — needed 0.999. A fixed tolerance is
therefore not a safe default at any value, which is the reason this module
sweeps instead of asking the caller to pick one. Fixtures and the test that
pins these numbers are in `tests/fixtures/fcidump/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .active_space import ActiveSpace, select_active_space
from .casci import MAX_DETERMINANTS, CASCIResult, solve_casci
from .hamiltonian import ElectronicHamiltonian
from .resources import CHEMICAL_ACCURACY_HARTREE

__all__ = [
    "DEFAULT_TOLERANCE_LADDER",
    "Candidate",
    "ReductionCertificate",
    "certify_reduction",
    "validate_space",
]

#: Swept low to high, so the first space that passes is the smallest that does.
DEFAULT_TOLERANCE_LADDER = (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999)

HARTREE_TO_KCAL = 627.5094740631


@dataclass(frozen=True)
class Candidate:
    """One space that was tried, and what happened when it was."""

    correlation_tolerance: float
    space: ActiveSpace | None
    energy: float | None
    error_hartree: float | None
    n_determinants: int | None
    qubits: int | None
    status: str
    detail: str = ""

    @property
    def error_kcal(self) -> float | None:
        if self.error_hartree is None:
            return None
        return self.error_hartree * HARTREE_TO_KCAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "correlation_tolerance": self.correlation_tolerance,
            "space": self.space.as_dict() if self.space else None,
            "energy": self.energy,
            "error_hartree": self.error_hartree,
            "error_kcal_per_mol": self.error_kcal,
            "n_determinants": self.n_determinants,
            "qubits": self.qubits,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class ReductionCertificate:
    """The smallest space that provably preserved the answer, or why none did."""

    verdict: str
    space: ActiveSpace | None
    reference_energy: float | None
    reference_source: str
    threshold_hartree: float
    error_hartree: float | None
    full_qubits: int
    full_determinants: int | None
    candidates: list[Candidate] = field(default_factory=list)
    reason: str = ""

    @property
    def certified(self) -> bool:
        return self.verdict == "certified"

    @property
    def qubits_saved(self) -> int | None:
        if self.space is None:
            return None
        return self.full_qubits - self.space.qubits

    @property
    def determinant_reduction(self) -> float | None:
        """How many times smaller the CI problem became."""
        if self.space is None or not self.full_determinants:
            return None
        return self.full_determinants / self.space.n_determinants

    @property
    def error_kcal(self) -> float | None:
        if self.error_hartree is None:
            return None
        return self.error_hartree * HARTREE_TO_KCAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "certified": self.certified,
            "space": self.space.as_dict() if self.space else None,
            "reference_energy": self.reference_energy,
            "reference_source": self.reference_source,
            "threshold_hartree": self.threshold_hartree,
            "error_hartree": self.error_hartree,
            "error_kcal_per_mol": self.error_kcal,
            "full_qubits": self.full_qubits,
            "full_determinants": self.full_determinants,
            "qubits_saved": self.qubits_saved,
            "determinant_reduction": self.determinant_reduction,
            "candidates": [c.as_dict() for c in self.candidates],
            "reason": self.reason,
        }

    def summary(self) -> str:
        if not self.certified or self.space is None:
            return f"not certified: {self.reason}"
        return (
            f"{self.space.label} certified at "
            f"{self.error_kcal:+.3f} kcal/mol against {self.reference_source}; "
            f"{self.space.qubits} qubits ({self.qubits_saved} fewer), "
            f"{self.determinant_reduction:.1f}x smaller CI problem"
        )


def validate_space(
    hamiltonian: ElectronicHamiltonian,
    space: ActiveSpace,
    reference_energy: float,
    *,
    threshold: float = CHEMICAL_ACCURACY_HARTREE,
    frozen: tuple[int, ...] | None = None,
) -> tuple[CASCIResult, float, bool]:
    """Solve ``space`` exactly and compare against ``reference_energy``.

    Returns the solve, the signed error in Hartree, and whether it is within
    ``threshold``. A solve that did not converge is never within threshold,
    regardless of the number it produced.
    """
    projected = hamiltonian.project(space, frozen)
    result = solve_casci(projected)
    error = result.energy - reference_energy
    return result, error, bool(result.converged and abs(error) <= threshold)


def certify_reduction(
    hamiltonian: ElectronicHamiltonian,
    one_rdm: np.ndarray,
    *,
    reference_energy: float | None = None,
    threshold: float = CHEMICAL_ACCURACY_HARTREE,
    tolerances: tuple[float, ...] = DEFAULT_TOLERANCE_LADDER,
    max_orbitals: int | None = None,
    max_determinants: int = MAX_DETERMINANTS,
) -> ReductionCertificate:
    """Find the smallest active space that reproduces the reference energy.

    ``one_rdm`` is a spatial one-particle density matrix from any correlated
    method the caller can afford — it selects the natural orbitals and ranks
    them, and it does not have to be accurate to do that job. The stretched-N2
    fixture is the demonstration: MP2 gets that energy wrong by 0.4 Hartree and
    its natural orbitals still rank the space correctly.

    ``reference_energy`` is the number the reduction has to reproduce. Omit it
    and the full space is solved exactly to produce one, which is only possible
    when the full space fits under ``max_determinants``; when it does not, and
    no reference was supplied, the result is a refusal.
    """
    natural, occupations = hamiltonian.to_natural_orbitals(one_rdm)
    full_space = ActiveSpace(
        n_electrons=hamiltonian.n_electrons,
        n_orbitals=hamiltonian.n_orbitals,
        orbital_indices=tuple(range(hamiltonian.n_orbitals)),
    )
    full_determinants = full_space.n_determinants

    source = "supplied reference"
    if reference_energy is None:
        if full_determinants > max_determinants:
            return ReductionCertificate(
                verdict="refused",
                space=None,
                reference_energy=None,
                reference_source="none",
                threshold_hartree=threshold,
                error_hartree=None,
                full_qubits=full_space.qubits,
                full_determinants=full_determinants,
                reason=(
                    f"no reference energy was supplied and the full space "
                    f"({full_determinants} determinants) is past the "
                    f"{max_determinants} solver cap, so there is nothing to "
                    f"certify against; supply a reference energy from the "
                    f"method you trust"
                ),
            )
        exact = solve_casci(natural)
        if not exact.converged:
            return ReductionCertificate(
                verdict="refused",
                space=None,
                reference_energy=None,
                reference_source="none",
                threshold_hartree=threshold,
                error_hartree=None,
                full_qubits=full_space.qubits,
                full_determinants=full_determinants,
                reason="the full-space solve did not converge, so it cannot serve as a reference",
            )
        reference_energy = exact.energy
        source = "full-space exact solve"

    candidates: list[Candidate] = []
    seen: set[tuple[int, ...]] = set()

    for tolerance in sorted(tolerances):
        report = select_active_space(
            occupations,
            correlation_tolerance=tolerance,
            max_orbitals=max_orbitals,
        )
        if not report.selected or report.space is None:
            candidates.append(
                Candidate(tolerance, None, None, None, None, None, "refused", report.reason)
            )
            continue

        space = report.space
        key = tuple(sorted(space.orbital_indices))
        if key in seen:
            continue
        seen.add(key)

        if space.n_determinants > max_determinants:
            candidates.append(
                Candidate(
                    tolerance,
                    space,
                    None,
                    None,
                    space.n_determinants,
                    space.qubits,
                    "too-large",
                    f"{space.n_determinants} determinants is past the {max_determinants} cap",
                )
            )
            continue

        try:
            result, error, passed = validate_space(
                natural, space, reference_energy, threshold=threshold
            )
        except ValueError as exc:
            candidates.append(
                Candidate(
                    tolerance, space, None, None, None, space.qubits, "invalid", str(exc)
                )
            )
            continue

        status = (
            "preserved"
            if passed
            else ("unconverged" if not result.converged else "degraded")
        )
        candidates.append(
            Candidate(
                tolerance,
                space,
                result.energy,
                error,
                result.n_determinants,
                space.qubits,
                status,
            )
        )

        if passed:
            return ReductionCertificate(
                verdict="certified",
                space=space,
                reference_energy=reference_energy,
                reference_source=source,
                threshold_hartree=threshold,
                error_hartree=error,
                full_qubits=full_space.qubits,
                full_determinants=full_determinants,
                candidates=candidates,
                reason="",
            )

    best = min(
        (c for c in candidates if c.error_hartree is not None),
        key=lambda c: abs(c.error_hartree or 0.0),
        default=None,
    )
    return ReductionCertificate(
        verdict="not-certified",
        space=None,
        reference_energy=reference_energy,
        reference_source=source,
        threshold_hartree=threshold,
        error_hartree=best.error_hartree if best else None,
        full_qubits=full_space.qubits,
        full_determinants=full_determinants,
        candidates=candidates,
        reason=(
            f"no space on the tolerance ladder reproduced the reference to "
            f"{threshold:.3g} Hartree; the closest was "
            f"{best.space.label if best and best.space else 'none'} at "
            f"{best.error_kcal:+.3f} kcal/mol"
            if best
            else "no candidate space could be solved"
        ),
    )
