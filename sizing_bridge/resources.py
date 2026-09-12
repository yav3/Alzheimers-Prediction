# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/resources.py at c002365.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""What a hybrid job will cost, and whether a device can run it at all.

Reducing a problem is only half of the argument. The other half is being able
to say, before an allocation is committed, what the reduced problem needs:
how many qubits, how many Hamiltonian terms survive, how many shots reach a
target precision, and whether the machine on the floor can hold the circuit.

Three things are modelled, and the boundaries between them matter.

**Counted, not assumed.** Qubit count and Pauli-term count come from the
Hamiltonian itself. The familiar O(N⁴) figure is an upper bound on the
two-electron tensor; a real molecule in a localised basis is much sparser, and
what a workload pays for is the terms that survive. `n_pauli_terms` counts.

**Derived from a stated model.** The shot budget follows from the variance of a
sum of weighted Pauli expectations: ε ≈ (Σ|c_k|)/√S under uniform allocation,
so S ≈ (Σ|c_k|/ε)². That is a standard, checkable bound, and the one-norm it
depends on is computed from the actual coefficients rather than estimated.

**Supplied by the caller.** Coherence time, gate durations, connectivity and
qubit count belong to a device, and we do not have a device. `DeviceModel` is a
parameter block the caller fills from a machine's own published or measured
figures; nothing here invents them, and a feasibility verdict is only as good
as the numbers put in.

Circuit depth is the honest weak point and is labelled as such. A depth
estimate for a chemistry ansatz depends on the ansatz, the compiler and the
connectivity; `depth_proxy` gives a transparent scaling proxy for comparing
two active spaces on the same footing, not a compiled-circuit figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .active_space import ActiveSpace, determinant_count
from .hamiltonian import INTEGRAL_TOL, ElectronicHamiltonian

__all__ = [
    "DeviceModel",
    "FeasibilityReport",
    "ResourceEstimate",
    "estimate_resources",
    "one_norm",
    "shots_for_precision",
]

Verdict = Literal["fits", "exceeds", "unknown"]


def one_norm(hamiltonian: ElectronicHamiltonian, tol: float = INTEGRAL_TOL) -> float:
    """Σ|c_k| over the Hamiltonian's terms — the quantity a shot budget scales with.

    Computed over the symmetry-unique integrals, matching how the terms would
    actually be grouped and measured. Includes the one-body terms and excludes
    the constant, which costs no measurement.
    """
    n = hamiltonian.n_orbitals
    total = 0.0
    h = hamiltonian.one_body
    for p in range(n):
        for q in range(p + 1):
            v = abs(h[p, q])
            if v > tol:
                total += v if p == q else 2.0 * v

    g = hamiltonian.two_body
    for p in range(n):
        for q in range(p + 1):
            for r in range(n):
                for s in range(r + 1):
                    if (p * (p + 1) // 2 + q) < (r * (r + 1) // 2 + s):
                        continue
                    v = abs(g[p, q, r, s])
                    if v > tol:
                        total += v
    return total


def shots_for_precision(one_norm_value: float, epsilon: float) -> int:
    """Shots to reach absolute precision ``epsilon`` under uniform allocation.

    S ≈ (Σ|c_k| / ε)². This is the standard bound and is deliberately the
    pessimistic one: grouping commuting terms or allocating by variance does
    better, which is what `allocation.py` is for. Quoting the loose bound and
    then improving on it is the honest direction to be wrong in.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if one_norm_value < 0:
        raise ValueError("one-norm must be non-negative")
    return int(math.ceil((one_norm_value / epsilon) ** 2))


@dataclass(frozen=True)
class DeviceModel:
    """A machine's constraints, supplied by whoever knows the machine.

    Nothing here is inferred. Populate it from published specifications or
    measured calibration data; a feasibility verdict inherits the reliability
    of these numbers and nothing better.
    """

    name: str
    n_qubits: int
    #: Median two-qubit gate duration, seconds.
    two_qubit_gate_seconds: float = 200e-9
    #: Coherence time bounding a circuit's usable duration, seconds.
    coherence_seconds: float = 100e-6
    #: Median two-qubit gate fidelity, for an aggregate-error estimate.
    two_qubit_fidelity: float = 0.99
    #: True when any qubit pair can interact; False for nearest-neighbour.
    all_to_all: bool = False
    provenance: str = ""

    @property
    def max_two_qubit_depth(self) -> int:
        """Two-qubit layers that fit inside coherence."""
        if self.two_qubit_gate_seconds <= 0:
            raise ValueError("two_qubit_gate_seconds must be positive")
        return int(self.coherence_seconds // self.two_qubit_gate_seconds)


@dataclass
class ResourceEstimate:
    """What one active space costs, in units a scheduler and a buyer both read."""

    space: ActiveSpace
    n_qubits: int
    n_pauli_terms: int
    n_one_body_terms: int
    n_two_body_terms: int
    one_norm: float
    n_determinants: int
    depth_proxy: int
    shots_for_chemical_accuracy: int
    #: The precision the shot figure targets, in Hartree.
    epsilon: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "space": self.space.as_dict(),
            "n_qubits": self.n_qubits,
            "n_pauli_terms": self.n_pauli_terms,
            "n_one_body_terms": self.n_one_body_terms,
            "n_two_body_terms": self.n_two_body_terms,
            "one_norm": round(self.one_norm, 9),
            "n_determinants": self.n_determinants,
            "depth_proxy": self.depth_proxy,
            "epsilon": self.epsilon,
            "shots_for_chemical_accuracy": self.shots_for_chemical_accuracy,
        }


@dataclass
class FeasibilityReport:
    """Whether a device can hold this job, and which constraint decides it."""

    device: str
    verdict: Verdict
    binding_constraint: str
    reasons: list[str] = field(default_factory=list)
    qubit_headroom: int = 0
    depth_headroom: int = 0

    @property
    def fits(self) -> bool:
        return self.verdict == "fits"

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "verdict": self.verdict,
            "binding_constraint": self.binding_constraint,
            "qubit_headroom": self.qubit_headroom,
            "depth_headroom": self.depth_headroom,
            "reasons": list(self.reasons),
        }


#: 1 kcal/mol in Hartree — the conventional target for "chemical accuracy".
CHEMICAL_ACCURACY_HARTREE = 1.5936e-3


def depth_proxy(n_qubits: int, n_two_body_terms: int) -> int:
    """A transparent, comparable stand-in for circuit depth.

    Not a compiled depth. A Trotter step over the two-body terms needs a number
    of two-qubit layers that grows with the term count, and each term acts on
    at most four spin-orbitals, so ``terms × n_qubits / 2`` is a scaling proxy
    with the right leading behaviour. Use it to rank two active spaces against
    each other, never as a figure to quote as a circuit depth.
    """
    if n_qubits < 0 or n_two_body_terms < 0:
        raise ValueError("counts must be non-negative")
    return int(n_two_body_terms * max(n_qubits // 2, 1))


def estimate_resources(
    hamiltonian: ElectronicHamiltonian,
    space: ActiveSpace | None = None,
    *,
    epsilon: float = CHEMICAL_ACCURACY_HARTREE,
    tol: float = INTEGRAL_TOL,
) -> ResourceEstimate:
    """Cost the Hamiltonian, optionally after projecting into ``space``.

    Passing a space projects first — including the frozen-core fold — so the
    estimate describes the job that would actually be submitted rather than the
    full-basis problem it came from.
    """
    if space is not None:
        hamiltonian = hamiltonian.project(space)
        described = space
    else:
        described = ActiveSpace(
            hamiltonian.n_electrons,
            hamiltonian.n_orbitals,
            tuple(range(hamiltonian.n_orbitals)),
        )

    n_one = hamiltonian.n_nonzero_one_body(tol)
    n_two = hamiltonian.n_nonzero_two_body(tol)
    norm = one_norm(hamiltonian, tol)

    return ResourceEstimate(
        space=described,
        n_qubits=hamiltonian.n_qubits,
        n_pauli_terms=n_one + n_two,
        n_one_body_terms=n_one,
        n_two_body_terms=n_two,
        one_norm=norm,
        n_determinants=determinant_count(described.n_electrons, described.n_orbitals),
        depth_proxy=depth_proxy(hamiltonian.n_qubits, n_two),
        shots_for_chemical_accuracy=shots_for_precision(norm, epsilon),
        epsilon=epsilon,
    )


def check_feasibility(estimate: ResourceEstimate, device: DeviceModel) -> FeasibilityReport:
    """Can this device hold this job, and if not, which limit binds first?

    Width is checked before depth because a problem that does not fit at all is
    a different conversation from one that fits but decoheres. Naming the
    binding constraint is the useful part: it says what a reduction would have
    to buy in order to change the answer.
    """
    reasons: list[str] = []
    qubit_headroom = device.n_qubits - estimate.n_qubits
    max_depth = device.max_two_qubit_depth
    depth_headroom = max_depth - estimate.depth_proxy

    if qubit_headroom < 0:
        reasons.append(
            f"needs {estimate.n_qubits} qubits, {device.name} has {device.n_qubits}"
        )
    if depth_headroom < 0:
        reasons.append(
            f"depth proxy {estimate.depth_proxy:,} exceeds the ~{max_depth:,} "
            f"two-qubit layers that fit in {device.coherence_seconds * 1e6:.0f} µs "
            "of coherence"
        )
    if not device.all_to_all:
        reasons.append(
            "device is not all-to-all; routing will add depth this proxy does "
            "not model, so treat the depth verdict as optimistic"
        )

    if qubit_headroom < 0:
        binding = "qubit count"
        verdict: Verdict = "exceeds"
    elif depth_headroom < 0:
        binding = "coherence-limited depth"
        verdict = "exceeds"
    else:
        binding = "none — width and the depth proxy both fit"
        verdict = "fits"

    return FeasibilityReport(
        device=device.name,
        verdict=verdict,
        binding_constraint=binding,
        reasons=reasons,
        qubit_headroom=qubit_headroom,
        depth_headroom=depth_headroom,
    )
