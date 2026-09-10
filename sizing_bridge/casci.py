# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/casci.py at 55e18a0.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Exact diagonalisation in an active space, so the reduction can be checked.

Selecting a smaller active space is only worth anything if the smaller space
still gives the right answer. That check is entirely classical: solve the
projected Hamiltonian exactly and compare against a reference energy. This
module is the solver, written against numpy alone so it travels with the rest
of the package and needs no quantum-chemistry install and no device.

**What it computes.** The lowest eigenvalue of

    H = Σ_pq h'_pq E_pq + ½ Σ_pqrs (pq|rs) E_pq E_rs,
    h'_pq = h_pq − ½ Σ_r (pr|rq),

in the determinant basis of a complete active space, with E_pq = Σ_σ a†_pσ a_qσ
the spin-summed excitation operator. Applied to the full orbital set this is
FCI; applied to a projected Hamiltonian it is CASCI. Same code either way,
which is the point — the reference and the candidate are produced by one
routine, so a difference between them is a property of the space and not of
two solvers disagreeing.

**How.** Determinants are (alpha string, beta string) pairs. Rather than form
the Hamiltonian matrix, the sigma routine applies it, in the Knowles–Handy
factorisation: build D_pq = E_pq c once, contract it with the integrals to get
G_pq, then apply E_pq a second time. Cost is O(n_orb⁴ · n_det) per iteration
instead of O(n_det²) storage. Davidson takes it from there.

**What it refuses.** The intermediate D_pq is dense, (n_orb, n_orb, n_det)
floats, so the determinant count is capped and exceeding it raises rather than
consumes the machine. Davidson non-convergence is reported in the result, not
smoothed over: an energy that did not converge is not an energy.

Spin: Ms = 0 or a fixed Ms via the electron split; this is a spin-restricted,
real-orbital code, matching the FCIDUMP convention used in `hamiltonian.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

__all__ = [
    "CASCIResult",
    "DeterminantSpace",
    "MAX_DETERMINANTS",
    "solve_casci",
]

#: Above this the dense (n_orb, n_orb, n_det) intermediate stops being sane.
MAX_DETERMINANTS = 100_000


@dataclass(frozen=True)
class CASCIResult:
    """A ground-state energy and the evidence that it is one."""

    energy: float
    electronic_energy: float
    core_energy: float
    n_determinants: int
    n_orbitals: int
    n_electrons: int
    converged: bool
    iterations: int
    residual: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "energy": self.energy,
            "electronic_energy": self.electronic_energy,
            "core_energy": self.core_energy,
            "n_determinants": self.n_determinants,
            "n_orbitals": self.n_orbitals,
            "n_electrons": self.n_electrons,
            "converged": self.converged,
            "iterations": self.iterations,
            "residual": self.residual,
        }


def _strings(n_orbitals: int, n_electrons: int) -> list[tuple[int, ...]]:
    """Occupation lists for every string, in lexicographic order."""
    return list(combinations(range(n_orbitals), n_electrons))


def _excitation_table(
    n_orbitals: int, n_electrons: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All (out, in, p, q, sign) with a†_p a_q |in> = sign · |out>.

    Signs follow the standard convention: annihilating q carries (−1) to the
    number of occupied orbitals below q, then creating p carries (−1) to the
    number of occupied orbitals below p in what is left. Getting this wrong is
    invisible in the diagonal and fatal off it, which is why the test suite
    checks the built Hamiltonian against an independent spin-orbital
    construction rather than against itself.
    """
    strings = _strings(n_orbitals, n_electrons)
    index = {s: i for i, s in enumerate(strings)}
    out: list[int] = []
    inn: list[int] = []
    ps: list[int] = []
    qs: list[int] = []
    signs: list[int] = []

    for j, s in enumerate(strings):
        occupied = set(s)
        for q in s:
            sign_annihilate = -1 if sum(1 for o in s if o < q) % 2 else 1
            rest = tuple(o for o in s if o != q)
            for p in range(n_orbitals):
                if p != q and p in occupied:
                    continue
                sign_create = -1 if sum(1 for o in rest if o < p) % 2 else 1
                target = tuple(sorted(rest + (p,)))
                out.append(index[target])
                inn.append(j)
                ps.append(p)
                qs.append(q)
                signs.append(sign_annihilate * sign_create)

    return (
        np.asarray(out, dtype=np.intp),
        np.asarray(inn, dtype=np.intp),
        np.asarray(ps, dtype=np.intp),
        np.asarray(qs, dtype=np.intp),
        np.asarray(signs, dtype=float),
    )


class DeterminantSpace:
    """The determinant basis of a (n_e, n_o) space and the operators on it."""

    def __init__(self, n_orbitals: int, n_alpha: int, n_beta: int) -> None:
        if n_orbitals < 1:
            raise ValueError("an active space needs at least one orbital")
        if not 0 <= n_alpha <= n_orbitals or not 0 <= n_beta <= n_orbitals:
            raise ValueError(
                f"cannot place {n_alpha}a/{n_beta}b electrons in "
                f"{n_orbitals} orbitals"
            )
        self.n_orbitals = n_orbitals
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_alpha_strings = comb(n_orbitals, n_alpha)
        self.n_beta_strings = comb(n_orbitals, n_beta)
        self.n_determinants = self.n_alpha_strings * self.n_beta_strings
        if self.n_determinants > MAX_DETERMINANTS:
            raise ValueError(
                f"{self.n_determinants} determinants exceeds the "
                f"{MAX_DETERMINANTS} cap for this solver; reduce the active "
                f"space or raise MAX_DETERMINANTS with the memory to match"
            )
        self._alpha = _excitation_table(n_orbitals, n_alpha)
        self._beta = (
            self._alpha
            if n_alpha == n_beta
            else _excitation_table(n_orbitals, n_beta)
        )
        self._occ_alpha = self._occupancy(n_orbitals, n_alpha)
        self._occ_beta = (
            self._occ_alpha
            if n_alpha == n_beta
            else self._occupancy(n_orbitals, n_beta)
        )

    @staticmethod
    def _occupancy(n_orbitals: int, n_electrons: int) -> np.ndarray:
        strings = _strings(n_orbitals, n_electrons)
        occ = np.zeros((len(strings), n_orbitals))
        for i, s in enumerate(strings):
            occ[i, list(s)] = 1.0
        return occ

    # ── E_pq applied to a coefficient matrix ────────────────────────────────

    def apply_excitations(self, c: np.ndarray) -> np.ndarray:
        """D[p, q] = E_pq c, for every (p, q) at once."""
        no, na, nb = self.n_orbitals, self.n_alpha_strings, self.n_beta_strings
        d = np.zeros((no, no, na, nb))

        out, inn, p, q, sign = self._alpha
        np.add.at(d, (p, q, out), sign[:, None] * c[inn])

        out, inn, p, q, sign = self._beta
        d_beta = np.zeros((no, no, nb, na))
        ct = np.ascontiguousarray(c.T)
        np.add.at(d_beta, (p, q, out), sign[:, None] * ct[inn])
        d += d_beta.transpose(0, 1, 3, 2)
        return d

    def contract_excitations(self, g: np.ndarray) -> np.ndarray:
        """Σ_pq E_pq g[p, q] — the adjoint half of the sigma build."""
        no, na, nb = self.n_orbitals, self.n_alpha_strings, self.n_beta_strings
        result = np.zeros((na, nb))

        out, inn, p, q, sign = self._alpha
        flat = g.reshape(no * no * na, nb)
        source = (p * no + q) * na + inn
        np.add.at(result, out, sign[:, None] * flat[source])

        out, inn, p, q, sign = self._beta
        flat_t = np.ascontiguousarray(g.transpose(0, 1, 3, 2)).reshape(
            no * no * nb, na
        )
        source = (p * no + q) * nb + inn
        result_t = np.zeros((nb, na))
        np.add.at(result_t, out, sign[:, None] * flat_t[source])
        result += result_t.T
        return result

    # ── The Hamiltonian ─────────────────────────────────────────────────────

    def sigma(
        self, c: np.ndarray, h_effective: np.ndarray, eri: np.ndarray
    ) -> np.ndarray:
        """H c, without ever forming H."""
        d = self.apply_excitations(c)
        s = np.einsum("pq,pqab->ab", h_effective, d, optimize=True)
        g = np.einsum("pqrs,rsab->pqab", eri, d, optimize=True)
        s += 0.5 * self.contract_excitations(g)
        return s

    def diagonal(self, h: np.ndarray, eri: np.ndarray) -> np.ndarray:
        """<D|H|D> for every determinant, for the Davidson preconditioner."""
        coulomb = np.einsum("ppqq->pq", eri)
        exchange = np.einsum("pqqp->pq", eri)
        h_diagonal = np.diag(h)
        oa, ob = self._occ_alpha, self._occ_beta

        same = coulomb - exchange
        ea = oa @ h_diagonal + 0.5 * np.einsum("ip,pq,iq->i", oa, same, oa)
        eb = ob @ h_diagonal + 0.5 * np.einsum("ip,pq,iq->i", ob, same, ob)
        cross = oa @ coulomb @ ob.T
        return ea[:, None] + eb[None, :] + cross


def _effective_one_body(h: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """h'_pq = h_pq − ½ Σ_r (pr|rq), the price of using E_pq E_rs."""
    return h - 0.5 * np.einsum("prrq->pq", eri)


def _davidson(
    space: DeterminantSpace,
    h_effective: np.ndarray,
    eri: np.ndarray,
    diagonal: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    max_subspace: int,
) -> tuple[float, np.ndarray, bool, int, float]:
    shape = diagonal.shape
    dim = diagonal.size
    flat_diagonal = diagonal.ravel()

    def apply(vector: np.ndarray) -> np.ndarray:
        return space.sigma(vector.reshape(shape), h_effective, eri).ravel()

    if dim <= 64:
        matrix = np.column_stack([apply(np.eye(dim)[:, i]) for i in range(dim)])
        matrix = 0.5 * (matrix + matrix.T)
        values, vectors = np.linalg.eigh(matrix)
        return float(values[0]), vectors[:, 0].reshape(shape), True, 1, 0.0

    guess = np.zeros(dim)
    guess[int(np.argmin(flat_diagonal))] = 1.0
    basis = [guess]
    products = [apply(guess)]
    energy = float(guess @ products[0])
    residual_norm = float("inf")

    for iteration in range(1, max_iterations + 1):
        v = np.column_stack(basis)
        w = np.column_stack(products)
        projected = v.T @ w
        projected = 0.5 * (projected + projected.T)
        values, vectors = np.linalg.eigh(projected)
        energy = float(values[0])
        coefficients = vectors[:, 0]
        eigenvector = v @ coefficients
        residual = w @ coefficients - energy * eigenvector
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm < tolerance:
            return energy, eigenvector.reshape(shape), True, iteration, residual_norm

        denominator = flat_diagonal - energy
        denominator[np.abs(denominator) < 1e-8] = 1e-8
        correction = residual / denominator

        for vector in basis:
            correction -= (vector @ correction) * vector
        norm = float(np.linalg.norm(correction))
        if norm < 1e-12:
            # The subspace already spans the correction; more iterations here
            # would spin without moving. Report what was reached.
            return (
                energy,
                eigenvector.reshape(shape),
                residual_norm < tolerance,
                iteration,
                residual_norm,
            )
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
    return (
        float(values[0]),
        (v @ vectors[:, 0]).reshape(shape),
        False,
        max_iterations,
        residual_norm,
    )


def solve_casci(
    hamiltonian: Any,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
    max_subspace: int = 24,
) -> CASCIResult:
    """Ground-state energy of ``hamiltonian`` by exact diagonalisation.

    ``hamiltonian`` is an `ElectronicHamiltonian` — typically the output of
    `ElectronicHamiltonian.project`, in which case this is a CASCI energy in
    the selected space, or the unprojected object, in which case it is FCI.

    The returned energy includes ``core_energy``, so a projected and an
    unprojected solve are directly comparable — which is the whole reason the
    frozen-core fold lives in the projection rather than in the caller.
    """
    h = np.asarray(hamiltonian.one_body, dtype=float)
    eri = np.asarray(hamiltonian.two_body, dtype=float)
    n_orbitals = h.shape[0]
    n_electrons = int(hamiltonian.n_electrons)
    ms2 = int(getattr(hamiltonian, "ms2", 0))

    if (n_electrons + ms2) % 2:
        raise ValueError(
            f"{n_electrons} electrons with MS2={ms2} is not a whole number of "
            f"alpha electrons"
        )
    n_alpha = (n_electrons + ms2) // 2
    n_beta = n_electrons - n_alpha

    space = DeterminantSpace(n_orbitals, n_alpha, n_beta)
    h_effective = _effective_one_body(h, eri)
    diagonal = space.diagonal(h, eri)

    energy, _, converged, iterations, residual = _davidson(
        space,
        h_effective,
        eri,
        diagonal,
        tolerance=tolerance,
        max_iterations=max_iterations,
        max_subspace=max_subspace,
    )

    core = float(hamiltonian.core_energy)
    return CASCIResult(
        energy=energy + core,
        electronic_energy=energy,
        core_energy=core,
        n_determinants=space.n_determinants,
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
        converged=converged,
        iterations=iterations,
        residual=residual,
    )
