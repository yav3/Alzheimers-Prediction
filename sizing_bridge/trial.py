# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/trial.py at bba1add.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""How good is a space as a *trial state*, which is not what it was certified on.

The certificate in `validate.py` asks whether a smaller active space reproduces
an energy. For a variational method that is the right question. For a
projector method it is not the only one: phaseless auxiliary-field quantum
Monte Carlo uses a trial wavefunction to control the sign problem, and the
bias it carries depends on how well that trial describes the state, not on how
close its energy happens to land.

Those two are not the same quantity and the difference is not small print. A
variational energy is stationary at the exact wavefunction, so its error is
*second order* in the error of the wavefunction. A space can therefore be
within chemical accuracy on energy while its wavefunction is a good deal
further from exact than that suggests — and a trial state is judged on the
wavefunction.

This module measures the other quantity. `trial_overlap` embeds an
active-space CI vector back into the full determinant space and reports
|<Psi_CAS|Psi_FCI>|, alongside the energy error, so the two can be read
together rather than one standing in for the other.

**It needs the full answer**, so it only applies where exact diagonalisation
is affordable — which is where the certificate can be exact too. Above that
this module has nothing to say, and says so rather than estimating.

The embedding is exact and sign-free. A frozen orbital is doubly occupied in
every determinant of the active space, so a CAS string maps to the full string
that is the sorted union of the frozen orbitals with the active ones it
selects. Both enumerations are in increasing orbital order, so the map is a
relabelling and carries no phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from .active_space import ActiveSpace
from .casci import solve_casci
from .hamiltonian import ElectronicHamiltonian

__all__ = ["TrialCertificate", "TrialQuality", "certify_trial_state", "trial_overlap"]

HARTREE_TO_KCAL = 627.5094740631


@dataclass(frozen=True)
class TrialQuality:
    """Two measurements of the same space: its energy, and its wavefunction."""

    overlap: float
    energy_error_hartree: float
    n_active_determinants: int
    n_full_determinants: int
    qubits: int

    @property
    def energy_error_kcal(self) -> float:
        return self.energy_error_hartree * HARTREE_TO_KCAL

    @property
    def missing_weight(self) -> float:
        """Weight of the exact state lying outside the active space."""
        return 1.0 - self.overlap**2

    def as_dict(self) -> dict[str, Any]:
        return {
            "overlap": self.overlap,
            "missing_weight": self.missing_weight,
            "energy_error_hartree": self.energy_error_hartree,
            "energy_error_kcal_per_mol": self.energy_error_kcal,
            "n_active_determinants": self.n_active_determinants,
            "n_full_determinants": self.n_full_determinants,
            "qubits": self.qubits,
        }

    def summary(self) -> str:
        return (
            f"{self.qubits} qubits: energy {self.energy_error_kcal:+.3f} kcal/mol, "
            f"overlap {self.overlap:.6f} "
            f"({self.missing_weight:.2e} of the state outside the space)"
        )


def _string_index(n_orbitals: int, n_electrons: int) -> dict[tuple[int, ...], int]:
    return {s: i for i, s in enumerate(combinations(range(n_orbitals), n_electrons))}


def _embed(
    coefficients: np.ndarray,
    active: tuple[int, ...],
    frozen: tuple[int, ...],
    n_orbitals: int,
    n_alpha_full: int,
    n_beta_full: int,
    n_alpha_active: int,
    n_beta_active: int,
) -> np.ndarray:
    """Place an active-space CI vector into the full determinant space."""
    full_alpha = _string_index(n_orbitals, n_alpha_full)
    full_beta = (
        full_alpha
        if n_alpha_full == n_beta_full
        else _string_index(n_orbitals, n_beta_full)
    )
    embedded = np.zeros((len(full_alpha), len(full_beta)))

    alpha_rows = [
        full_alpha[tuple(sorted(frozen + tuple(active[i] for i in picked)))]
        for picked in combinations(range(len(active)), n_alpha_active)
    ]
    beta_rows = [
        full_beta[tuple(sorted(frozen + tuple(active[i] for i in picked)))]
        for picked in combinations(range(len(active)), n_beta_active)
    ]
    embedded[np.ix_(alpha_rows, beta_rows)] = coefficients
    return embedded


def trial_overlap(
    hamiltonian: ElectronicHamiltonian,
    space: ActiveSpace,
    *,
    frozen: tuple[int, ...] | None = None,
    exact: np.ndarray | None = None,
    exact_energy: float | None = None,
) -> TrialQuality:
    """Overlap of the active-space state with the exact one, and the energy error.

    ``hamiltonian`` must be in the same orbital basis the space refers to —
    the output of `to_natural_orbitals`, not the original file, if the space
    was selected from natural orbitals. ``exact`` and ``exact_energy`` let a
    caller reuse one full solve across many spaces, which is the usual case.
    """
    if exact is None or exact_energy is None:
        full = solve_casci(hamiltonian)
        if not full.converged or full.coefficients is None:
            raise ValueError("the full-space solve did not converge")
        exact = full.coefficients
        exact_energy = full.energy

    active = tuple(sorted(space.orbital_indices))
    if frozen is None:
        n_frozen_electrons = hamiltonian.n_electrons - space.n_electrons
        if n_frozen_electrons < 0 or n_frozen_electrons % 2:
            raise ValueError(
                f"cannot infer frozen orbitals: {hamiltonian.n_electrons} total "
                f"and {space.n_electrons} active leaves {n_frozen_electrons}"
            )
        candidates = [i for i in range(hamiltonian.n_orbitals) if i not in set(active)]
        frozen = tuple(candidates[: n_frozen_electrons // 2])
    frozen = tuple(sorted(frozen))

    projected = hamiltonian.project(space, frozen)
    result = solve_casci(projected)
    if not result.converged or result.coefficients is None:
        raise ValueError("the active-space solve did not converge")

    ms2 = int(hamiltonian.ms2)
    n_alpha_full = (hamiltonian.n_electrons + ms2) // 2
    n_alpha_active = (space.n_electrons + ms2) // 2

    embedded = _embed(
        result.coefficients,
        active,
        frozen,
        hamiltonian.n_orbitals,
        n_alpha_full,
        hamiltonian.n_electrons - n_alpha_full,
        n_alpha_active,
        space.n_electrons - n_alpha_active,
    )

    exact = np.asarray(exact, dtype=float)
    overlap = float(abs((embedded * exact).sum()))
    overlap /= float(np.linalg.norm(embedded)) * float(np.linalg.norm(exact))

    return TrialQuality(
        overlap=min(overlap, 1.0),
        energy_error_hartree=result.energy - float(exact_energy),
        n_active_determinants=result.n_determinants,
        n_full_determinants=int(exact.size),
        qubits=space.qubits,
    )


@dataclass
class TrialCertificate:
    """The smallest space that is good enough as a wavefunction, not just as an energy."""

    verdict: str
    space: ActiveSpace | None
    quality: TrialQuality | None
    energy_threshold: float
    overlap_threshold: float
    full_qubits: int
    measured: list[tuple[ActiveSpace, TrialQuality]] = None  # type: ignore[assignment]
    reason: str = ""

    def __post_init__(self) -> None:
        if self.measured is None:
            self.measured = []

    @property
    def certified(self) -> bool:
        return self.verdict == "certified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "certified": self.certified,
            "space": self.space.as_dict() if self.space else None,
            "quality": self.quality.as_dict() if self.quality else None,
            "energy_threshold": self.energy_threshold,
            "overlap_threshold": self.overlap_threshold,
            "full_qubits": self.full_qubits,
            "measured": [
                {"space": s.as_dict(), **q.as_dict()} for s, q in self.measured
            ],
            "reason": self.reason,
        }

    def summary(self) -> str:
        if not self.certified or self.quality is None or self.space is None:
            return f"not certified: {self.reason}"
        return f"{self.space.label} {self.quality.summary()}"


def certify_trial_state(
    hamiltonian: ElectronicHamiltonian,
    one_rdm: np.ndarray,
    *,
    energy_threshold: float = 1.5936e-3,
    overlap_threshold: float = 0.999,
    max_orbitals: int | None = None,
    max_determinants: int | None = None,
) -> TrialCertificate:
    """Smallest space passing *both* an energy and a wavefunction test.

    Use this when the reduced space is going to be a trial wavefunction rather
    than an answer — a projector method's bias tracks the wavefunction, and a
    variational energy is stationary at the exact state, so energy error is
    second order in wavefunction error and systematically flatters the space.

    Measured on the reference molecules, the two criteria do not rank the same
    spaces the same way: by energy the certified spaces span a factor of nine,
    by weight outside the space they span a factor of four hundred. A space
    comfortably inside chemical accuracy can still leave half a percent of the
    state unrepresented.

    This needs the exact wavefunction, so it applies only where full
    diagonalisation is affordable. Past that it refuses; there is no cheap
    proxy here that would not be an invention.
    """
    from .active_space import ranked_spaces

    natural, occupations = hamiltonian.to_natural_orbitals(one_rdm)
    full_qubits = 2 * hamiltonian.n_orbitals

    try:
        exact = solve_casci(natural)
    except ValueError as exc:
        return TrialCertificate(
            verdict="refused",
            space=None,
            quality=None,
            energy_threshold=energy_threshold,
            overlap_threshold=overlap_threshold,
            full_qubits=full_qubits,
            reason=(
                f"the exact wavefunction is out of reach, and a trial state "
                f"cannot be judged without it: {exc}"
            ),
        )
    if not exact.converged or exact.coefficients is None:
        return TrialCertificate(
            verdict="refused",
            space=None,
            quality=None,
            energy_threshold=energy_threshold,
            overlap_threshold=overlap_threshold,
            full_qubits=full_qubits,
            reason="the full-space solve did not converge",
        )

    measured: list[tuple[ActiveSpace, TrialQuality]] = []
    for space in ranked_spaces(occupations, max_orbitals=max_orbitals):
        if space.qubits >= full_qubits:
            continue
        if max_determinants and space.n_determinants > max_determinants:
            continue
        try:
            quality = trial_overlap(
                natural,
                space,
                exact=exact.coefficients,
                exact_energy=exact.energy,
            )
        except ValueError:
            continue
        measured.append((space, quality))
        if (
            abs(quality.energy_error_hartree) <= energy_threshold
            and quality.overlap >= overlap_threshold
        ):
            return TrialCertificate(
                verdict="certified",
                space=space,
                quality=quality,
                energy_threshold=energy_threshold,
                overlap_threshold=overlap_threshold,
                full_qubits=full_qubits,
                measured=measured,
            )

    energy_only = [
        (s, q)
        for s, q in measured
        if abs(q.energy_error_hartree) <= energy_threshold
    ]
    reason = "no space met both tests"
    if energy_only:
        space, quality = energy_only[0]
        reason = (
            f"{space.label} passes on energy at "
            f"{quality.energy_error_kcal:+.3f} kcal/mol but leaves "
            f"{quality.missing_weight:.2e} of the state outside it, short of "
            f"the {1 - overlap_threshold**2:.2e} this asks of a trial state"
        )
    return TrialCertificate(
        verdict="not-certified",
        space=None,
        quality=None,
        energy_threshold=energy_threshold,
        overlap_threshold=overlap_threshold,
        full_qubits=full_qubits,
        measured=measured,
        reason=reason,
    )
