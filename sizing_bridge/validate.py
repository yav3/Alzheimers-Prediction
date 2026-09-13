# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/validate.py at bba1add.
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
nested spaces from smallest upward, solves each one exactly, and returns the
smallest whose energy sits within the threshold of the reference, together
with every candidate it rejected on the way. A caller gets the space *and* the
evidence, which is the difference between a recommendation and a certificate.

The sweep walks one space per size. An earlier version walked a ladder of
correlation tolerances and skipped sizes, returning the first rung that
happened to clear — two to six qubits larger than necessary on three of the
four reference molecules. Correlation retained is still reported for each
candidate, because it is informative, but it is an observation rather than a
dial: stretched N2 at 99.60% retained is 3.9 kcal/mol out, and at 99.84% it is
0.6 kcal/mol in. Nothing about that boundary is knowable in advance.

**It refuses rather than guesses.** With no reference energy and a full space
too large to solve, there is nothing to certify against, and the report says
so instead of returning the selection with an implied blessing. A space that
was never checked is reported as unchecked.

**Above the exact-solve ceiling.** An exact reference stops being computable
at roughly 20 to 24 qubits — (10e,10o) is 63,504 determinants and solves in
seconds, (12e,12o) is 853,776 and does not. The molecules worth putting on a
device are all past that line, so `certify_reduction` cannot produce its own
reference for them. `certify_by_convergence` is the answer: it certifies a
small space against the largest space that *is* affordable, and refuses when
that larger space has not itself converged. Nested complete active spaces are
variational in each other, so the sequence is monotone and a converged tail is
evidence rather than coincidence. It is the criterion a chemist already uses
to decide an active space is big enough; this makes it mechanical and records
what it measured. What it cannot do is see correlation that lies outside every
space on the ladder — convergence in active-space size is not convergence in
basis set, and this reports the first only.

**What the sweep measured.** On five molecules with exact FCI references —
H2O/STO-3G, N2/STO-3G, stretched N2/STO-3G at 2.1 Å, LiH/6-31G and triplet
O2/STO-3G — the certified spaces save 4 to 14 qubits and shrink the CI problem
by 4x to 189x, all inside chemical accuracy. Taking instead the space that a
fixed 0.95 correlation tolerance returns, and scoring it the same way, holds
for one of those molecules and silently degrades the rest.

The density matrix that ranks the orbitals does not have to be accurate: MP2
misses stretched N2 by 0.4 hartree and its natural orbitals still order the
space correctly. Fixtures and the tests that pin these numbers are in
`tests/fixtures/fcidump/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .active_space import (
    ActiveSpace,
    correlation_mass,
    ranked_spaces,
)
from .casci import MAX_DETERMINANTS, CASCIResult, solve_casci
from .hamiltonian import ElectronicHamiltonian
from .manifest import build_manifest, fingerprint
from .resources import CHEMICAL_ACCURACY_HARTREE

__all__ = [
    "Candidate",
    "certify_across_rankings",
    "ExternalSpaceVerdict",
    "score_supplied_space",
    "ReductionCertificate",
    "certify_by_convergence",
    "certify_reduction",
    "validate_space",
]

HARTREE_TO_KCAL = 627.5094740631


@dataclass(frozen=True)
class Candidate:
    """One space that was tried, and what happened when it was."""

    #: Fraction of the total correlation mass this space carries. Reported
    #: rather than dialled: the sweep walks sizes, and how much correlation a
    #: size happens to retain is an observation about the molecule, not a
    #: setting. It was a parameter until measurement showed no value of it
    #: generalised across molecules.
    correlation_retained: float | None
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
            "correlation_retained": self.correlation_retained,
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
    #: Hash-addressed record of what produced this certificate. Same inputs,
    #: same id — the clock is attached after the hash so it never perturbs it.
    manifest: dict[str, Any] = field(default_factory=dict)
    #: Only set by `certify_by_convergence`: whether the reference space itself
    #: had stopped moving, and by how much it moved on its last step.
    reference_converged: bool | None = None
    reference_step_hartree: float | None = None

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
            "reference_converged": self.reference_converged,
            "reference_step_hartree": self.reference_step_hartree,
            "bound_kcal_per_mol": self.bound_kcal,
            "manifest": dict(self.manifest),
        }

    @property
    def bound_kcal(self) -> float | None:
        """Distance to the converged answer, not just to the reference.

        Under convergence certification the reference is itself only settled
        to within its last step, so the honest figure is the sum. Equal to the
        error alone when the reference was exact.
        """
        if self.error_hartree is None:
            return None
        step = self.reference_step_hartree or 0.0
        return (abs(self.error_hartree) + step) * HARTREE_TO_KCAL

    def summary(self) -> str:
        if not self.certified or self.space is None:
            return f"not certified: {self.reason}"
        bound = (
            ""
            if self.reference_step_hartree is None
            else f" (bound on the converged answer: {self.bound_kcal:.3f} kcal/mol)"
        )
        return (
            f"{self.space.label} certified at "
            f"{self.error_kcal:+.3f} kcal/mol against {self.reference_source}"
            f"{bound}; {self.space.qubits} qubits ({self.qubits_saved} fewer), "
            f"{self.determinant_reduction:.1f}x smaller CI problem"
        )


def _retention(occupations: np.ndarray):
    """How much of the molecule's correlation a space carries, as a fraction."""
    mass = correlation_mass(np.asarray(occupations, dtype=float))
    total = float(mass.sum())

    def retained(space: ActiveSpace) -> float | None:
        if total <= 0.0:
            return None
        return float(mass[list(space.orbital_indices)].sum()) / total

    return retained


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


def _stamped(
    certificate: ReductionCertificate,
    *,
    analysis: str,
    hamiltonian: ElectronicHamiltonian,
    one_rdm: np.ndarray,
    parameters: dict[str, Any],
) -> ReductionCertificate:
    """Attach the run manifest. Every exit path goes through here.

    A certificate is a claim about a specific Hamiltonian, a specific density
    matrix and a specific threshold, and it is worth nothing to a reader who
    cannot tell which. The manifest fingerprints all three, so two
    certificates can be compared without re-deriving what produced them, and a
    refusal is recorded as durably as a pass.
    """
    payload = certificate.as_dict()
    certificate.manifest = build_manifest(
        analysis=analysis,
        parameters=parameters,
        seed=None,
        input_fingerprint=fingerprint(
            {
                "one_body": np.asarray(hamiltonian.one_body).round(12).tolist(),
                "two_body_norm": float(np.linalg.norm(hamiltonian.two_body)),
                "n_electrons": hamiltonian.n_electrons,
                "ms2": hamiltonian.ms2,
                "core_energy": round(float(hamiltonian.core_energy), 12),
                "one_rdm": np.asarray(one_rdm).round(12).tolist(),
            }
        ),
        stages=[
            {
                "candidate": c.space.label if c.space else None,
                "qubits": c.qubits,
                "status": c.status,
            }
            for c in certificate.candidates
        ],
        output_payload=payload,
    )
    return certificate


def _certify_reduction(
    hamiltonian: ElectronicHamiltonian,
    one_rdm: np.ndarray,
    *,
    reference_energy: float | None = None,
    threshold: float = CHEMICAL_ACCURACY_HARTREE,
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
    retained = _retention(occupations)

    # One space per size, smallest first. A tolerance ladder was the first
    # design and it skipped sizes: on three of the four reference molecules it
    # returned a space two to six qubits larger than the smallest one that
    # holds, because no rung of the ladder landed on the smaller space.
    for space in ranked_spaces(occupations, max_orbitals=max_orbitals):
        if space.n_determinants > max_determinants:
            candidates.append(
                Candidate(
                    retained(space),
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
                    retained(space), space, None, None, None, space.qubits, "invalid", str(exc)
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
                retained(space),
                space,
                result.energy,
                error,
                result.n_determinants,
                space.qubits,
                status,
            )
        )

        if passed and space.qubits < full_space.qubits:
            # A space the size of the full one is not a reduction, whatever
            # its error against the reference says — and when the reference is
            # the full-space solve, that error is zero by construction. Record
            # it as a candidate, never as a certificate.
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
            f"no space smaller than the full one reproduced the reference to "
            f"{threshold:.3g} Hartree; the closest was "
            f"{best.space.label if best and best.space else 'none'} at "
            f"{best.error_kcal:+.3f} kcal/mol"
            if best
            else "no candidate space could be solved"
        ),
    )


def _certify_by_convergence(
    hamiltonian: ElectronicHamiltonian,
    one_rdm: np.ndarray,
    *,
    threshold: float = CHEMICAL_ACCURACY_HARTREE,
    max_orbitals: int | None = None,
    max_determinants: int = MAX_DETERMINANTS,
) -> ReductionCertificate:
    """Certify a space against a converged larger one, when exact is out of reach.

    `certify_reduction` needs a reference energy, and above roughly 24 qubits
    there is not one to be had: the full space cannot be solved and a
    coupled-cluster number is not a substitute, because it carries dynamic
    correlation that a complete active space is not meant to reproduce.
    Comparing a CAS energy to CCSD(T) measures what the active space leaves
    out by design, not whether the space was well chosen.

    What is available is the sequence itself. The ladder's spaces are nested,
    so their energies fall monotonically, and a tail that has stopped falling
    is the standard evidence that the active space is large enough. This
    solves the ladder as far as the determinant budget allows, takes the
    largest solved space as the reference, and returns the smallest space
    within ``threshold`` of it.

    The refusal that matters: if the reference's own last step was larger than
    ``threshold``, the sequence had not converged, the reference is still
    moving, and no space is certified against it. That case is reported as
    ``reference-not-converged`` with the step size, rather than as a
    certificate against a moving target.
    """
    natural, occupations = hamiltonian.to_natural_orbitals(one_rdm)
    full_space = ActiveSpace(
        n_electrons=hamiltonian.n_electrons,
        n_orbitals=hamiltonian.n_orbitals,
        orbital_indices=tuple(range(hamiltonian.n_orbitals)),
    )
    # One space per size rather than one per tolerance: a tolerance ladder can
    # jump four orbitals between rungs, and a step that large cannot
    # distinguish "converged" from "not yet started".
    retained = _retention(occupations)
    ladder = [
        (retained(space), space)
        for space in ranked_spaces(occupations, max_orbitals=max_orbitals)
    ]

    candidates: list[Candidate] = []
    solved: list[tuple[float, ActiveSpace, float]] = []

    for tolerance, space in ladder:
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
                    f"{space.n_determinants} determinants is past the "
                    f"{max_determinants} budget",
                )
            )
            continue
        try:
            projected = natural.project(space)
            result = solve_casci(projected)
        except ValueError as exc:
            candidates.append(
                Candidate(
                    tolerance, space, None, None, None, space.qubits, "invalid", str(exc)
                )
            )
            continue
        if not result.converged:
            candidates.append(
                Candidate(
                    tolerance,
                    space,
                    result.energy,
                    None,
                    result.n_determinants,
                    space.qubits,
                    "unconverged",
                    "the solve did not converge, so its energy is not usable",
                )
            )
            continue
        solved.append((tolerance, space, result.energy))

    def _empty(reason: str) -> ReductionCertificate:
        return ReductionCertificate(
            verdict="refused",
            space=None,
            reference_energy=None,
            reference_source="none",
            threshold_hartree=threshold,
            error_hartree=None,
            full_qubits=full_space.qubits,
            full_determinants=full_space.n_determinants,
            candidates=candidates,
            reason=reason,
        )

    if len(solved) < 2:
        return _empty(
            f"only {len(solved)} space on the ladder could be solved within the "
            f"{max_determinants}-determinant budget; convergence needs at least "
            f"two so the reference's own movement can be measured"
        )

    _, reference_space, reference_energy = solved[-1]
    previous_energy = solved[-2][2]
    step = abs(reference_energy - previous_energy)
    source = f"largest converged space {reference_space.label}"

    for tolerance, space, energy in solved:
        error = energy - reference_energy
        candidates.append(
            Candidate(
                tolerance,
                space,
                energy,
                error,
                space.n_determinants,
                space.qubits,
                "preserved" if abs(error) + step <= threshold else "degraded",
            )
        )
    candidates.sort(key=lambda c: (c.qubits or 0, c.correlation_retained or 0.0))

    if step > threshold:
        return ReductionCertificate(
            verdict="reference-not-converged",
            space=None,
            reference_energy=reference_energy,
            reference_source=source,
            threshold_hartree=threshold,
            error_hartree=None,
            full_qubits=full_space.qubits,
            full_determinants=full_space.n_determinants,
            candidates=candidates,
            reference_converged=False,
            reference_step_hartree=step,
            reason=(
                f"the ladder had not converged where the determinant budget ran "
                f"out: the last step, {solved[-2][1].label} to "
                f"{reference_space.label}, still moved the energy by "
                f"{step * HARTREE_TO_KCAL:.3f} kcal/mol. Nothing is certified "
                f"against a reference that is still falling — raise the budget, "
                f"or supply a reference energy you trust"
            ),
        )

    for _, space, energy in solved[:-1]:
        # The reference itself is excluded: its error against itself is zero by
        # construction, so "certifying" it would be a statement with no
        # comparison in it. A certificate has to name something smaller than
        # the largest thing that was solved, or it is not a reduction.
        error = energy - reference_energy
        # The reference is itself only converged to within `step`, so a
        # candidate `error` away from it is at most `error + step` from the
        # limit. Certifying on `error` alone would spend the budget twice.
        if abs(error) + step <= threshold:
            return ReductionCertificate(
                verdict="certified",
                space=space,
                reference_energy=reference_energy,
                reference_source=source,
                threshold_hartree=threshold,
                error_hartree=error,
                full_qubits=full_space.qubits,
                full_determinants=full_space.n_determinants,
                candidates=candidates,
                reference_converged=True,
                reference_step_hartree=step,
            )

    reference_certificate = ReductionCertificate(
        verdict="reference-only",
        space=None,
        reference_energy=reference_energy,
        reference_source=source,
        threshold_hartree=threshold,
        error_hartree=None,
        full_qubits=full_space.qubits,
        full_determinants=full_space.n_determinants,
        candidates=candidates,
        reference_converged=True,
        reference_step_hartree=step,
        reason=(
            f"the sequence converged — its last step was "
            f"{step * HARTREE_TO_KCAL:.3f} kcal/mol — but no smaller space came "
            f"within {threshold * HARTREE_TO_KCAL:.3f} kcal/mol of "
            f"{reference_space.label} once that residual movement is counted "
            f"against the budget. {reference_space.label} at "
            f"{reference_space.qubits} qubits is the smallest space this run "
            f"can stand behind, and it is a real reduction from "
            f"{full_space.qubits}; nothing below it is"
        ),
    )
    return reference_certificate


def certify_reduction(
    hamiltonian: ElectronicHamiltonian,
    one_rdm: np.ndarray,
    *,
    reference_energy: float | None = None,
    threshold: float = CHEMICAL_ACCURACY_HARTREE,
    max_orbitals: int | None = None,
    max_determinants: int = MAX_DETERMINANTS,
) -> ReductionCertificate:
    """Certify against a reference energy, stamping the run manifest."""
    certificate = _certify_reduction(
        hamiltonian,
        one_rdm,
        reference_energy=reference_energy,
        threshold=threshold,
        max_orbitals=max_orbitals,
        max_determinants=max_determinants,
    )
    return _stamped(
        certificate,
        analysis="active_space_certification",
        hamiltonian=hamiltonian,
        one_rdm=one_rdm,
        parameters={
            "threshold": threshold,
            "max_orbitals": max_orbitals,
            "max_determinants": max_determinants,
            "reference_supplied": reference_energy is not None,
        },
    )


def certify_by_convergence(
    hamiltonian: ElectronicHamiltonian,
    one_rdm: np.ndarray,
    *,
    threshold: float = CHEMICAL_ACCURACY_HARTREE,
    max_orbitals: int | None = None,
    max_determinants: int = MAX_DETERMINANTS,
) -> ReductionCertificate:
    """Certify against a converged larger space, stamping the run manifest."""
    certificate = _certify_by_convergence(
        hamiltonian,
        one_rdm,
        threshold=threshold,
        max_orbitals=max_orbitals,
        max_determinants=max_determinants,
    )
    return _stamped(
        certificate,
        analysis="active_space_convergence_certification",
        hamiltonian=hamiltonian,
        one_rdm=one_rdm,
        parameters={
            "threshold": threshold,
            "max_orbitals": max_orbitals,
            "max_determinants": max_determinants,
        },
    )


@dataclass
class ExternalSpaceVerdict:
    """What a space chosen elsewhere cost, on the same terms as one of ours."""

    verdict: str
    energy: float
    reference_energy: float
    reference_source: str
    error_hartree: float
    threshold_hartree: float
    qubits: int
    n_determinants: int
    reference_step_hartree: float | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def preserved(self) -> bool:
        return self.verdict == "preserved"

    @property
    def error_kcal(self) -> float:
        return self.error_hartree * HARTREE_TO_KCAL

    @property
    def bound_kcal(self) -> float:
        step = self.reference_step_hartree or 0.0
        return (abs(self.error_hartree) + step) * HARTREE_TO_KCAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "preserved": self.preserved,
            "energy": self.energy,
            "reference_energy": self.reference_energy,
            "reference_source": self.reference_source,
            "error_hartree": self.error_hartree,
            "error_kcal_per_mol": self.error_kcal,
            "bound_kcal_per_mol": self.bound_kcal,
            "threshold_hartree": self.threshold_hartree,
            "qubits": self.qubits,
            "n_determinants": self.n_determinants,
            "reference_step_hartree": self.reference_step_hartree,
            "manifest": dict(self.manifest),
        }

    def summary(self) -> str:
        return (
            f"a supplied {self.qubits}-qubit space lands "
            f"{self.error_kcal:+.3f} kcal/mol from {self.reference_source} "
            f"(bound {self.bound_kcal:.3f} kcal/mol): {self.verdict}"
        )


def score_supplied_space(
    projected: ElectronicHamiltonian,
    *,
    reference_energy: float,
    reference_source: str = "supplied reference",
    threshold: float = CHEMICAL_ACCURACY_HARTREE,
    reference_step: float | None = None,
) -> ExternalSpaceVerdict:
    """Score an active space this package did not choose.

    ``projected`` is an already-projected active-space Hamiltonian — what a
    package writes out when it has selected a space by its own criteria and
    folded the frozen core itself. AVAS, DMET and a chemist working by hand all
    produce one, and each does it in its own orbital basis, so there is nothing
    to re-project here and nothing of ours to impose.

    The comparison is still sound across bases, and that is the point. A CASCI
    wavefunction in any active space is a valid N-electron wavefunction in the
    full one-particle basis, so its energy is an upper bound on the same
    full-basis exact answer. Two spaces chosen by different methods, in
    different orbitals, are therefore directly comparable as long as the
    underlying basis set is the same — lower is better, with no reference
    needed to establish the ordering and no way for the choice of reference to
    favour one selector.

    ``reference_step`` carries the residual movement of a reference produced by
    `certify_by_convergence`, so a supplied space is judged on the same bound
    as one of ours rather than on a more forgiving one.

    We would rather be the thing that scores a space than one more thing that
    chooses one. Where our own selection is weakest — a nearly-filled d shell,
    where ranking by occupancy is blind — this is the entry point that still
    gives an honest answer.
    """
    result = solve_casci(projected)
    error = result.energy - reference_energy
    step = reference_step or 0.0

    if not result.converged:
        verdict = "unconverged"
    elif abs(error) + step <= threshold:
        verdict = "preserved"
    else:
        verdict = "degraded"

    scored = ExternalSpaceVerdict(
        verdict=verdict,
        energy=result.energy,
        reference_energy=reference_energy,
        reference_source=reference_source,
        error_hartree=error,
        threshold_hartree=threshold,
        qubits=projected.n_qubits,
        n_determinants=result.n_determinants,
        reference_step_hartree=reference_step,
    )
    scored.manifest = build_manifest(
        analysis="supplied_space_scoring",
        parameters={
            "threshold": threshold,
            "reference_energy": reference_energy,
            "reference_source": reference_source,
            "reference_step": reference_step,
        },
        seed=None,
        input_fingerprint=fingerprint(
            {
                "one_body": np.asarray(projected.one_body).round(12).tolist(),
                "two_body_norm": float(np.linalg.norm(projected.two_body)),
                "n_electrons": projected.n_electrons,
                "ms2": projected.ms2,
                "core_energy": round(float(projected.core_energy), 12),
            }
        ),
        stages=[{"candidate": "supplied", "qubits": scored.qubits, "status": verdict}],
        output_payload=scored.as_dict(),
    )
    return scored


def certify_across_rankings(
    hamiltonian: ElectronicHamiltonian,
    rankings: dict[str, np.ndarray],
    *,
    reference_energy: float | None = None,
    threshold: float = CHEMICAL_ACCURACY_HARTREE,
    max_orbitals: int | None = None,
    max_determinants: int = MAX_DETERMINANTS,
) -> ReductionCertificate:
    """Try several orbital rankings and keep the best space any of them found.

    No cheap ranking is reliable everywhere. Occupations from perturbation
    theory are blind to a nearly-filled d shell and collapse entirely when the
    perturbation series does; seniority-zero occupations see static
    correlation but not dynamic, and give up a little on weakly correlated
    molecules. Measured on the benchmark set, swapping one for the other moves
    copper hydride by 88 kcal/mol in our favour and nitrogen by 5 against.

    Choosing between them in advance is the problem this package exists not to
    have. Each ranking produces a ladder, every candidate is solved and scored
    against the same reference, and the smallest space that holds wins on the
    measurement rather than on the reputation of the method that proposed it.
    The certificate names which ranking produced it, and the candidates from
    all of them are kept, so a reader can see that the other ranking was tried
    and what it offered.

    ``rankings`` maps a name to a one-particle density matrix. For a ranking
    that is already diagonal in the working basis — seniority-zero occupations
    are — pass ``np.diag(occupations)``; the rotation then reduces to the
    sort, and the orbital indices carry through unchanged.
    """
    if not rankings:
        raise ValueError("at least one ranking is required")

    best: ReductionCertificate | None = None
    everything: list[Candidate] = []

    for name, one_rdm in rankings.items():
        certificate = certify_reduction(
            hamiltonian,
            one_rdm,
            reference_energy=reference_energy,
            threshold=threshold,
            max_orbitals=max_orbitals,
            max_determinants=max_determinants,
        )
        for candidate in certificate.candidates:
            everything.append(
                Candidate(
                    candidate.correlation_retained,
                    candidate.space,
                    candidate.energy,
                    candidate.error_hartree,
                    candidate.n_determinants,
                    candidate.qubits,
                    candidate.status,
                    f"ranking={name}"
                    + (f"; {candidate.detail}" if candidate.detail else ""),
                )
            )
        if not certificate.certified or certificate.space is None:
            if best is None:
                best = certificate
            continue
        if (
            best is None
            or not best.certified
            or best.space is None
            or certificate.space.qubits < best.space.qubits
            or (
                certificate.space.qubits == best.space.qubits
                and abs(certificate.error_hartree or 0.0)
                < abs(best.error_hartree or 0.0)
            )
        ):
            best = certificate
            best.reference_source = f"{certificate.reference_source} (ranking={name})"

    assert best is not None
    best.candidates = sorted(everything, key=lambda c: (c.qubits or 0, c.detail))
    return best
