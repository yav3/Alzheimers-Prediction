# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/hamiltonian.py at c002365.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Electronic Hamiltonians, and the standard file that carries them.

Everything else in this package works on abstract matrices. This module is
where the bridge meets an actual molecule: FCIDUMP is the interchange format
essentially every quantum-chemistry package reads and writes, so being able to
consume one is the difference between a method and a thing a chemist can point
at their data.

Three capabilities live here.

**Parsing and writing FCIDUMP.** The format is a Fortran namelist header
(``NORB``, ``NELEC``, ``MS2``, ``ORBSYM``, ``ISYM``) followed by integral
records ``value i j k l`` with 1-based indices, where ``i j k l`` nonzero is
the two-electron integral (ij|kl) in chemists' notation, ``i j 0 0`` is the
one-electron h_ij, and ``0 0 0 0`` is the scalar core energy. Files store one
representative of each 8-fold-symmetric orbit; the reader expands the orbit so
downstream code can index freely.

**Active-space projection with frozen-core folding.** Restricting to an active
space is not simply slicing the integral tensors: the doubly-occupied orbitals
left behind still interact with the ones kept. Their effect folds into a
scalar core energy and a correction to the one-body term,

    E_core   = E_nuc + 2 Σ_i h_ii + Σ_ij [ 2 (ii|jj) − (ij|ji) ]
    h_eff_pq = h_pq + Σ_i [ 2 (pq|ii) − (pi|iq) ]

with i, j running over frozen orbitals and p, q over active ones. Dropping the
fold is a silent error that shifts every energy, so it is done here rather than
left to the caller.

**Natural occupations from a one-particle density matrix.** The eigenvalues of
the spatial 1-RDM are the natural orbital occupation numbers, which is the
input `active_space.select_active_space` consumes. That closes the loop: a
density matrix from a cheap classical method selects the space, and this module
projects the Hamiltonian into it.

Real-orbital 8-fold symmetry is assumed throughout — the standard case for
molecular FCIDUMPs. Complex orbitals carry only 4-fold symmetry and are
rejected rather than silently mishandled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .active_space import ActiveSpace

__all__ = [
    "ElectronicHamiltonian",
    "natural_occupations",
    "parse_fcidump",
    "read_fcidump",
    "write_fcidump",
]

#: Integrals below this are treated as absent when writing and counting.
INTEGRAL_TOL = 1e-12


@dataclass
class ElectronicHamiltonian:
    """A molecular electronic Hamiltonian in a spatial-orbital basis.

    ``one_body`` is h_pq with shape (n_orbitals, n_orbitals); ``two_body`` is
    (pq|rs) in chemists' notation with shape (n,)*4; ``core_energy`` is the
    scalar term, which carries nuclear repulsion and, after a projection, the
    folded contribution of the frozen orbitals.
    """

    one_body: np.ndarray
    two_body: np.ndarray
    n_electrons: int
    core_energy: float = 0.0
    ms2: int = 0
    orbital_symmetries: tuple[int, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.one_body = np.asarray(self.one_body, dtype=float)
        self.two_body = np.asarray(self.two_body, dtype=float)
        n = self.n_orbitals
        if self.one_body.shape != (n, n):
            raise ValueError(f"one_body must be ({n}, {n}), got {self.one_body.shape}")
        if self.two_body.shape != (n, n, n, n):
            raise ValueError(f"two_body must be {(n,) * 4}, got {self.two_body.shape}")
        if self.n_electrons < 0:
            raise ValueError("n_electrons must be non-negative")
        if self.n_electrons > 2 * n:
            raise ValueError(
                f"{self.n_electrons} electrons cannot occupy {n} spatial orbitals"
            )
        if self.orbital_symmetries and len(self.orbital_symmetries) != n:
            raise ValueError("orbital_symmetries must have one entry per orbital")

    @property
    def n_orbitals(self) -> int:
        return int(self.one_body.shape[0])

    @property
    def n_qubits(self) -> int:
        """Spin-orbitals, which is the qubit count under a standard mapping."""
        return 2 * self.n_orbitals

    def n_nonzero_two_body(self, tol: float = INTEGRAL_TOL) -> int:
        """Distinct symmetry-unique two-electron integrals above ``tol``.

        Counted rather than assumed: the O(N⁴) figure is an upper bound, and a
        real molecule in a localised basis is far sparser than that. What a
        workload actually pays for is the terms that survive.
        """
        n = self.n_orbitals
        seen: set[tuple[int, int, int, int]] = set()
        for p in range(n):
            for q in range(p + 1):
                for r in range(n):
                    for s in range(r + 1):
                        if (p * (p + 1) // 2 + q) < (r * (r + 1) // 2 + s):
                            continue
                        if abs(self.two_body[p, q, r, s]) > tol:
                            seen.add((p, q, r, s))
        return len(seen)

    def n_nonzero_one_body(self, tol: float = INTEGRAL_TOL) -> int:
        n = self.n_orbitals
        return int(sum(
            1 for p in range(n) for q in range(p + 1)
            if abs(self.one_body[p, q]) > tol
        ))

    def project(self, space: ActiveSpace, frozen: tuple[int, ...] | None = None) -> ElectronicHamiltonian:
        """Restrict to ``space``, folding the frozen orbitals into the core.

        ``frozen`` defaults to the orbitals below the active window that the
        electron count implies are doubly occupied. Passing it explicitly is
        the safer habit when the active orbitals are not contiguous.
        """
        active = tuple(sorted(space.orbital_indices))
        if not active:
            raise ValueError("active space contains no orbitals")
        if any(i < 0 or i >= self.n_orbitals for i in active):
            raise ValueError("active orbital index out of range")

        if frozen is None:
            n_frozen_electrons = self.n_electrons - space.n_electrons
            if n_frozen_electrons < 0 or n_frozen_electrons % 2:
                raise ValueError(
                    f"cannot infer frozen orbitals: {self.n_electrons} total "
                    f"electrons and {space.n_electrons} active leaves "
                    f"{n_frozen_electrons}, which must be even and non-negative"
                )
            n_frozen = n_frozen_electrons // 2
            candidates = [i for i in range(self.n_orbitals) if i not in set(active)]
            frozen = tuple(candidates[:n_frozen])
        frozen = tuple(sorted(frozen))
        if set(frozen) & set(active):
            raise ValueError("an orbital cannot be both frozen and active")

        h, g = self.one_body, self.two_body
        core = float(self.core_energy)
        for i in frozen:
            core += 2.0 * h[i, i]
        for i in frozen:
            for j in frozen:
                core += 2.0 * g[i, i, j, j] - g[i, j, j, i]

        idx = np.ix_(active, active)
        h_eff = h[idx].copy()
        for k, p in enumerate(active):
            for m, q in enumerate(active):
                for i in frozen:
                    h_eff[k, m] += 2.0 * g[p, q, i, i] - g[p, i, i, q]

        g_act = g[np.ix_(active, active, active, active)].copy()

        return ElectronicHamiltonian(
            one_body=h_eff,
            two_body=g_act,
            n_electrons=space.n_electrons,
            core_energy=core,
            ms2=self.ms2,
            orbital_symmetries=tuple(self.orbital_symmetries[i] for i in active)
            if self.orbital_symmetries
            else (),
            provenance={
                **self.provenance,
                "projected_from": self.n_orbitals,
                "active_orbitals": list(active),
                "frozen_orbitals": list(frozen),
            },
        )

    def transform(self, coefficients: np.ndarray) -> ElectronicHamiltonian:
        """Rotate into a new orthonormal orbital basis.

        ``coefficients`` is the matrix U whose columns are the new orbitals
        expressed in the current ones. The integrals transform as
        h' = Uᵀ h U and (pq|rs)' = Σ U U U U (..), which is a quartic
        contraction done here once rather than open-coded at each call site.

        Orbital symmetries are carried through when the rotation provably
        preserves them and dropped when it does not. A rotation that mixes two
        irreps destroys the labels, and claiming one that no longer holds is
        worse than carrying none — but the rotation that matters here, into
        natural orbitals, is usually block-diagonal by irrep, because the
        density matrix commutes with the symmetry operations. Discarding the
        labels unconditionally would throw away information the receiving code
        can use, so the block structure is checked rather than assumed.
        """
        u = np.asarray(coefficients, dtype=float)
        n = self.n_orbitals
        if u.shape != (n, n):
            raise ValueError(f"coefficients must be ({n}, {n}), got {u.shape}")
        if not np.allclose(u.T @ u, np.eye(n), atol=1e-8):
            raise ValueError("coefficients must be orthonormal")

        h = u.T @ self.one_body @ u
        g = np.einsum("pqrs,pa,qb,rc,sd->abcd", self.two_body, u, u, u, u, optimize=True)
        return ElectronicHamiltonian(
            one_body=h,
            two_body=g,
            n_electrons=self.n_electrons,
            core_energy=self.core_energy,
            ms2=self.ms2,
            orbital_symmetries=_rotated_symmetries(u, self.orbital_symmetries),
            provenance={**self.provenance, "basis_rotated": True},
        )

    def to_natural_orbitals(
        self, one_rdm: np.ndarray
    ) -> tuple[ElectronicHamiltonian, np.ndarray]:
        """Rotate into the natural orbitals of ``one_rdm``, occupations first.

        This is the step that makes a selection applicable to an actual
        Hamiltonian. `select_active_space` ranks orbitals by how far their
        occupation sits from an integer, and that ranking is only meaningful in
        the basis that diagonalises the density matrix. Returning the rotated
        Hamiltonian alongside the occupations keeps the two in the same orbital
        order, which is the pairing every downstream projection assumes.

        The density matrix is the spatial 1-RDM from whatever correlated method
        the caller can afford — MP2 and CCSD are the usual choices. It does not
        need to be accurate; it needs to rank orbitals.
        """
        rdm = np.asarray(one_rdm, dtype=float)
        n = self.n_orbitals
        if rdm.shape != (n, n):
            raise ValueError(f"one_rdm must be ({n}, {n}), got {rdm.shape}")
        if not np.allclose(rdm, rdm.T, atol=1e-8):
            raise ValueError("one_rdm must be symmetric")

        values, vectors = np.linalg.eigh(rdm)
        order = np.argsort(values)[::-1]
        occupations = np.clip(values[order], 0.0, 2.0)
        return self.transform(vectors[:, order]), occupations

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_orbitals": self.n_orbitals,
            "n_electrons": self.n_electrons,
            "n_qubits": self.n_qubits,
            "core_energy": self.core_energy,
            "ms2": self.ms2,
            "n_nonzero_one_body": self.n_nonzero_one_body(),
            "n_nonzero_two_body": self.n_nonzero_two_body(),
            "provenance": dict(self.provenance),
        }


# ── Natural occupations ──────────────────────────────────────────────────────

def _rotated_symmetries(
    u: np.ndarray, symmetries: tuple[int, ...], tol: float = 1e-8
) -> tuple[int, ...]:
    """Symmetry labels after a rotation, or none if the rotation mixed irreps.

    A new orbital keeps a label only if every old orbital contributing to it
    carried that same label. One mixed column invalidates the whole set, not
    just its own entry: ORBSYM is read as a block structure by the codes that
    consume it, and a partially-correct block structure is not a weaker claim
    than none, it is a wrong one.
    """
    if not symmetries or len(symmetries) != u.shape[0]:
        return ()
    labels = np.asarray(symmetries)
    rotated: list[int] = []
    for column in range(u.shape[1]):
        contributing = np.unique(labels[np.abs(u[:, column]) > tol])
        if contributing.size != 1:
            return ()
        rotated.append(int(contributing[0]))
    return tuple(rotated)


def natural_occupations(one_rdm: np.ndarray) -> np.ndarray:
    """Natural orbital occupation numbers: eigenvalues of the spatial 1-RDM.

    Returned in descending order and clipped into [0, 2], which is where a
    physical occupation lives; small negative eigenvalues are a normal artefact
    of an approximate density matrix and are not an error.
    """
    rdm = np.asarray(one_rdm, dtype=float)
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError("one_rdm must be square")
    if not np.allclose(rdm, rdm.T, atol=1e-8):
        raise ValueError("one_rdm must be symmetric")
    eigenvalues = np.linalg.eigvalsh(rdm)
    return np.clip(eigenvalues[::-1], 0.0, 2.0)


# ── FCIDUMP ──────────────────────────────────────────────────────────────────

_HEADER_END = re.compile(r"[&$]END|/\s*$", re.IGNORECASE)


def _parse_header(lines: list[str]) -> tuple[dict[str, Any], int]:
    """Read the Fortran namelist header; return its fields and where it ends."""
    text_parts: list[str] = []
    end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        text_parts.append(stripped)
        if _HEADER_END.search(stripped):
            end = i + 1
            break
    else:
        raise ValueError("FCIDUMP header has no &END terminator")

    text = " ".join(text_parts)
    fields: dict[str, Any] = {}
    for key in ("NORB", "NELEC", "MS2", "ISYM"):
        m = re.search(rf"\b{key}\s*=\s*(-?\d+)", text, re.IGNORECASE)
        if m:
            fields[key] = int(m.group(1))
    m = re.search(r"\bORBSYM\s*=\s*([0-9,\s]+)", text, re.IGNORECASE)
    if m:
        fields["ORBSYM"] = tuple(
            int(v) for v in re.split(r"[,\s]+", m.group(1).strip()) if v
        )
    if "NORB" not in fields or "NELEC" not in fields:
        raise ValueError("FCIDUMP header must define NORB and NELEC")
    return fields, end


def parse_fcidump(text: str) -> ElectronicHamiltonian:
    """Parse FCIDUMP text into a Hamiltonian, expanding 8-fold symmetry.

    Files carry one representative per symmetry orbit. Expanding on read means
    downstream code indexes ``two_body[p, q, r, s]`` without having to know
    which representative the file happened to store.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    fields, start = _parse_header(lines)

    n = int(fields["NORB"])
    h = np.zeros((n, n))
    g = np.zeros((n, n, n, n))
    core = 0.0

    for lineno, line in enumerate(lines[start:], start=start + 1):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"line {lineno}: expected 'value i j k l', got {line!r}")
        try:
            value = float(parts[0].replace("D", "E").replace("d", "e"))
            p, q, r, s = (int(v) for v in parts[1:5])
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc

        if p == q == r == s == 0:
            core = value
        elif r == 0 and s == 0:
            if not (1 <= p <= n and 1 <= q <= n):
                raise ValueError(f"line {lineno}: orbital index out of range")
            h[p - 1, q - 1] = value
            h[q - 1, p - 1] = value
        else:
            if not all(1 <= v <= n for v in (p, q, r, s)):
                raise ValueError(f"line {lineno}: orbital index out of range")
            _set_eightfold(g, p - 1, q - 1, r - 1, s - 1, value)

    return ElectronicHamiltonian(
        one_body=h,
        two_body=g,
        n_electrons=int(fields["NELEC"]),
        core_energy=core,
        ms2=int(fields.get("MS2", 0)),
        orbital_symmetries=tuple(fields.get("ORBSYM", ())),
        provenance={"source": "fcidump"},
    )


def _set_eightfold(g: np.ndarray, p: int, q: int, r: int, s: int, value: float) -> None:
    """Populate the full 8-fold orbit of (pq|rs) for real orbitals."""
    for a, b, c, d in (
        (p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
        (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p),
    ):
        g[a, b, c, d] = value


def read_fcidump(path: str | Path) -> ElectronicHamiltonian:
    """Read an FCIDUMP file from disk."""
    hamiltonian = parse_fcidump(Path(path).read_text())
    hamiltonian.provenance["path"] = str(path)
    return hamiltonian


def write_fcidump(hamiltonian: ElectronicHamiltonian, stream: TextIO, tol: float = INTEGRAL_TOL) -> None:
    """Write a Hamiltonian in FCIDUMP form, one record per symmetry orbit.

    Round-trips through `parse_fcidump`, which the tests assert — a writer that
    cannot be read back by its own reader is worse than no writer.
    """
    n = hamiltonian.n_orbitals
    symmetries = hamiltonian.orbital_symmetries or tuple([1] * n)
    stream.write(f" &FCI NORB={n},NELEC={hamiltonian.n_electrons},MS2={hamiltonian.ms2},\n")
    stream.write("  ORBSYM=" + ",".join(str(s) for s in symmetries) + ",\n")
    stream.write("  ISYM=1,\n &END\n")

    g = hamiltonian.two_body
    for p in range(n):
        for q in range(p + 1):
            for r in range(n):
                for s in range(r + 1):
                    if (p * (p + 1) // 2 + q) < (r * (r + 1) // 2 + s):
                        continue
                    v = g[p, q, r, s]
                    if abs(v) > tol:
                        stream.write(f"{v:>23.16e} {p + 1:3d} {q + 1:3d} {r + 1:3d} {s + 1:3d}\n")

    h = hamiltonian.one_body
    for p in range(n):
        for q in range(p + 1):
            if abs(h[p, q]) > tol:
                stream.write(f"{h[p, q]:>23.16e} {p + 1:3d} {q + 1:3d}   0   0\n")

    stream.write(f"{hamiltonian.core_energy:>23.16e}   0   0   0   0\n")
