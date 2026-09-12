# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/doci.py at c002365.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Seniority-zero CI, for ranking orbitals when perturbation theory has failed.

The selection in this package ranks orbitals by how far their occupation sits
from a whole number, and it takes those occupations from whatever cheap
correlated method the caller can afford — usually MP2. That works for
main-group molecules near equilibrium and fails in two documented ways: a
nearly-filled d shell carries the chemistry while looking uncorrelated, and a
strongly correlated system breaks the perturbation series outright. On the
[2Fe-2S] cluster, MP2 lands *above* Hartree-Fock and returns nine occupations
of exactly 2.000 for a maximally correlated molecule. A ranking built on that
is a ranking built on noise. See LANDMINES.

Doubly-occupied configuration interaction is the cheap method that does not
have this failure mode. It is a full CI restricted to closed-shell
determinants — every orbital either empty or doubly occupied, seniority zero —
which is exactly the part of the wavefunction that carries *static*
correlation, the kind MP2 misses. The restriction takes the dimension from
C(n_o, n_e/2)² to C(n_o, n_e/2): the square root. For the [2Fe-2S] active
space that is 15,504 determinants instead of 240 million.

What comes back is a set of pair occupations, and they are the ranking. In a
seniority-zero wavefunction the one-particle density matrix is diagonal in the
working basis, so the occupation of orbital p is just twice the weight of the
determinants containing it — no diagonalisation and no rotation. The ranking
is of the orbitals you were handed, which is what AVAS and occupation windows
also do.

**What it is not.** DOCI is not an accurate energy method here: without
orbital optimisation it recovers static correlation and none of the dynamic
part, and its energy is far above exact. That is fine and it is the point —
this module exists to order orbitals, not to answer. Use the energies it
returns for nothing else.

The matrix elements are the standard seniority-zero ones,

    <P|H|P>  = Σ_{p∈P} [2 h_pp + (pp|pp)] + Σ_{p<q∈P} [4 (pp|qq) − 2 (pq|qp)]
    <P|H|Q>  = (pq|pq)     for P, Q differing by one pair moved q → p

and the test suite checks them against the seniority-zero block of the full
Hamiltonian built by the validated FCI machinery, rather than against
themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

__all__ = ["DOCIResult", "MAX_PAIR_DETERMINANTS", "doci_occupations", "solve_doci"]

#: Seniority-zero dimension is the square root of the full one, so this cap is
#: far less binding than the solver's — it is here to stop a pathological ask.
MAX_PAIR_DETERMINANTS = 2_000_000


@dataclass(frozen=True)
class DOCIResult:
    """Pair occupations, and the energy that produced them — which is not an answer."""

    occupations: np.ndarray
    energy: float
    n_determinants: int
    n_orbitals: int
    n_electrons: int
    converged: bool
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "occupations": self.occupations.tolist(),
            "energy": self.energy,
            "n_determinants": self.n_determinants,
            "n_orbitals": self.n_orbitals,
            "n_electrons": self.n_electrons,
            "converged": self.converged,
            "iterations": self.iterations,
        }


def _pair_configurations(n_orbitals: int, n_pairs: int) -> np.ndarray:
    """Occupancy rows, one per closed-shell determinant."""
    rows = np.zeros((comb(n_orbitals, n_pairs), n_orbitals))
    for index, occupied in enumerate(combinations(range(n_orbitals), n_pairs)):
        rows[index, list(occupied)] = 1.0
    return rows


def _hopping_lists(
    configurations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(out, in, p, q) for every single-pair move q → p between configurations."""
    n_configurations, n_orbitals = configurations.shape
    index = {tuple(np.flatnonzero(row)): i for i, row in enumerate(configurations)}

    out: list[int] = []
    inn: list[int] = []
    ps: list[int] = []
    qs: list[int] = []
    for j, row in enumerate(configurations):
        occupied = set(np.flatnonzero(row).tolist())
        for q in sorted(occupied):
            for p in range(n_orbitals):
                if p in occupied:
                    continue
                target = tuple(sorted((occupied - {q}) | {p}))
                out.append(index[target])
                inn.append(j)
                ps.append(p)
                qs.append(q)
    return (
        np.asarray(out, dtype=np.intp),
        np.asarray(inn, dtype=np.intp),
        np.asarray(ps, dtype=np.intp),
        np.asarray(qs, dtype=np.intp),
    )


def solve_doci(
    hamiltonian: Any,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
    max_subspace: int = 24,
    max_determinants: int = MAX_PAIR_DETERMINANTS,
) -> DOCIResult:
    """Lowest seniority-zero state, and the pair occupations that rank the orbitals.

    ``hamiltonian`` is an `ElectronicHamiltonian`. An odd electron count has no
    closed-shell space to restrict to and is rejected rather than rounded.
    """
    h = np.asarray(hamiltonian.one_body, dtype=float)
    eri = np.asarray(hamiltonian.two_body, dtype=float)
    n_orbitals = h.shape[0]
    n_electrons = int(hamiltonian.n_electrons)

    if int(getattr(hamiltonian, "ms2", 0)) != 0 or n_electrons % 2:
        raise ValueError(
            f"seniority zero needs a closed-shell electron count; got "
            f"{n_electrons} electrons with MS2={getattr(hamiltonian, 'ms2', 0)}"
        )
    n_pairs = n_electrons // 2
    if not 0 <= n_pairs <= n_orbitals:
        raise ValueError(f"cannot place {n_pairs} pairs in {n_orbitals} orbitals")

    dimension = comb(n_orbitals, n_pairs)
    if dimension > max_determinants:
        raise ValueError(
            f"{dimension} pair determinants exceeds the {max_determinants} cap"
        )

    coulomb = np.einsum("ppqq->pq", eri)
    exchange = np.einsum("pqqp->pq", eri)
    hopping = np.einsum("pqpq->pq", eri)

    configurations = _pair_configurations(n_orbitals, n_pairs)
    # Per-orbital and pair-interaction parts of the diagonal. The self term is
    # excluded from the pair sum and folded into the single-orbital term.
    single = 2.0 * np.diag(h) + np.diag(coulomb)
    pair = 4.0 * coulomb - 2.0 * exchange
    np.fill_diagonal(pair, 0.0)
    diagonal = configurations @ single + 0.5 * np.einsum(
        "ip,pq,iq->i", configurations, pair, configurations
    )

    out, inn, ps, qs = _hopping_lists(configurations)
    amplitudes = hopping[ps, qs]

    def apply(vector: np.ndarray) -> np.ndarray:
        result = diagonal * vector
        np.add.at(result, out, amplitudes * vector[inn])
        return result

    if dimension == 1:
        energy = float(diagonal[0])
        weights = configurations[0]
        return DOCIResult(
            occupations=2.0 * weights,
            energy=energy + float(hamiltonian.core_energy),
            n_determinants=1,
            n_orbitals=n_orbitals,
            n_electrons=n_electrons,
            converged=True,
            iterations=0,
        )

    if dimension <= 512:
        matrix = np.diag(diagonal)
        np.add.at(matrix, (out, inn), amplitudes)
        matrix = 0.5 * (matrix + matrix.T)
        values, vectors = np.linalg.eigh(matrix)
        energy, ground, converged, iterations = (
            float(values[0]),
            vectors[:, 0],
            True,
            1,
        )
    else:
        energy, ground, converged, iterations = _davidson(
            apply, diagonal, tolerance, max_iterations, max_subspace
        )

    weights = ground**2
    occupations = 2.0 * (weights @ configurations)
    return DOCIResult(
        occupations=np.clip(occupations, 0.0, 2.0),
        energy=energy + float(hamiltonian.core_energy),
        n_determinants=dimension,
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
        converged=converged,
        iterations=iterations,
    )


def _davidson(apply, diagonal, tolerance, max_iterations, max_subspace):
    dimension = diagonal.size
    guess = np.zeros(dimension)
    guess[int(np.argmin(diagonal))] = 1.0
    basis = [guess]
    products = [apply(guess)]
    energy = float(guess @ products[0])

    for iteration in range(1, max_iterations + 1):
        v = np.column_stack(basis)
        w = np.column_stack(products)
        projected = 0.5 * (v.T @ w + (v.T @ w).T)
        values, vectors = np.linalg.eigh(projected)
        energy = float(values[0])
        eigenvector = v @ vectors[:, 0]
        residual = w @ vectors[:, 0] - energy * eigenvector
        if float(np.linalg.norm(residual)) < tolerance:
            return energy, eigenvector, True, iteration

        denominator = diagonal - energy
        denominator[np.abs(denominator) < 1e-8] = 1e-8
        correction = residual / denominator
        for vector in basis:
            correction -= (vector @ correction) * vector
        norm = float(np.linalg.norm(correction))
        if norm < 1e-12:
            return energy, eigenvector, False, iteration
        correction /= norm

        if len(basis) >= max_subspace:
            basis = [eigenvector / np.linalg.norm(eigenvector)]
            products = [apply(basis[0])]
        basis.append(correction)
        products.append(apply(correction))

    v = np.column_stack(basis)
    w = np.column_stack(products)
    projected = 0.5 * (v.T @ w + (v.T @ w).T)
    values, vectors = np.linalg.eigh(projected)
    return float(values[0]), v @ vectors[:, 0], False, max_iterations


def doci_occupations(hamiltonian: Any, **kwargs: Any) -> np.ndarray:
    """The ranking alone, for callers that want it in place of an MP2 density."""
    return solve_doci(hamiltonian, **kwargs).occupations
