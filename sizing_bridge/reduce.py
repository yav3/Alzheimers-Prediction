# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/reduce.py at fd54432.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Sizing search — how small can this problem be, and is that defensible?

Two mechanisms, and the choice between them is the substance of this module.

**Spectral (default).** A data-dependent basis: centre, take the SVD, keep the
leading k directions. Because the basis is fitted to the data, a problem with
exploitable structure collapses to its true width essentially exactly, while
one without it does not — which is what makes the refusal meaningful.

**Random (Johnson–Lindenstrauss).** The B2 lens mechanism in
`grn_pack_b.distinguishability`. It is *oblivious by construction*: it
preserves pairwise distances whatever the data looks like, and therefore
cannot distinguish a compressible problem from an incompressible one. Measured
on a rank-8 matrix embedded in 24 dimensions it scores no better than on
isotropic noise of the same shape. It is the right tool when a caller wants a
data-independent guarantee and the wrong one for deciding how small a problem
can be, so it is available but not the default.

Both are gated on *measured* worst-case pairwise distortion against the
caller's tolerance. Nothing is accepted on a theoretical bound alone, and a
reduction is only ever reported when the width actually fell.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .linalg import effective_rank, jl_project
from .spec import ReductionReport

__all__ = ["search_reduction", "candidate_dims", "worst_pairwise_distortion"]

Method = Literal["spectral", "random"]


def worst_pairwise_distortion(X: np.ndarray, X_reduced: np.ndarray) -> float:
    """Largest relative change in any pairwise distance, measured not assumed."""
    if X.shape[0] < 2:
        return 0.0
    orig = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    proj = np.linalg.norm(X_reduced[:, None, :] - X_reduced[None, :, :], axis=-1)
    mask = orig > 1e-12
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(proj[mask] / orig[mask] - 1.0)))


def candidate_dims(
    input_dim: int, floor: int = 2, *, hint: int | None = None
) -> list[int]:
    """Candidate widths to try, smallest first.

    Geometric rather than exhaustive: the search must stay cheap relative to
    the evaluation it protects, and distortion is monotone enough in k that a
    dense sweep buys nothing.

    ``hint`` seeds the ladder with the data's own structural estimate — the
    effective rank. Without it a geometric ladder can step straight over the
    true width (a rank-8 problem lands on 9 because 8 was never tried), which
    costs the caller a dimension it did not need to spend.
    """
    if input_dim <= floor:
        return []
    dims: set[int] = {floor}
    k = input_dim
    while k > floor:
        k = int(k * 0.75)
        if k >= floor:
            dims.add(k)
    if hint is not None and floor <= hint < input_dim:
        dims.add(hint)
    return sorted(d for d in dims if d < input_dim)


def _reduce_to(X: np.ndarray, k: int, method: Method, seed: int) -> np.ndarray:
    if method == "spectral":
        centred = X - X.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centred, full_matrices=False)
        return u[:, :k] * s[:k]
    if method == "random":
        reduced, _ = jl_project(X, k=k, seed=seed)
        return reduced
    raise ValueError(f"unknown method {method!r}")


def _reference(X: np.ndarray, method: Method) -> np.ndarray:
    """Distances are compared in the frame the reduction works in."""
    return X - X.mean(axis=0, keepdims=True) if method == "spectral" else X


def search_reduction(
    X: np.ndarray,
    *,
    distortion_tolerance: float = 0.15,
    floor_dim: int = 2,
    method: Method = "spectral",
    seed: int = 0,
) -> tuple[np.ndarray, ReductionReport]:
    """Find the smallest width whose measured distortion stays within tolerance.

    Three outcomes, and only the first is a reduction:

      reduced         a smaller width held the tolerance
      passed_through  the structure does not support reduction at this tolerance
      refused         there is nothing to search (too narrow, or degenerate)

    The last two are not failures. A layer that always returns a smaller number
    is not a guard rail, and the refusal is what makes the reductions it does
    return worth acting on.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite — run audit_matrix first")

    n_rows, input_dim = X.shape
    er = round(float(effective_rank(X)), 6)

    if input_dim <= floor_dim or n_rows < 2:
        return X.copy(), ReductionReport(
            verdict="refused",
            input_dim=input_dim,
            output_dim=input_dim,
            effective_rank=er,
            distortion_observed=None,
            distortion_tolerance=distortion_tolerance,
            reason=(
                f"input is {n_rows}×{input_dim}; nothing to search below "
                f"floor_dim={floor_dim}"
            ),
        )

    reference = _reference(X, method)
    evaluated = 0
    hint = int(np.ceil(er)) if er > 0 else None
    for k in candidate_dims(input_dim, floor=floor_dim, hint=hint):
        evaluated += 1
        reduced = _reduce_to(X, k, method, seed)
        distortion = worst_pairwise_distortion(reference, reduced)
        if distortion <= distortion_tolerance:
            return reduced, ReductionReport(
                verdict="reduced",
                input_dim=input_dim,
                output_dim=k,
                effective_rank=er,
                distortion_observed=round(distortion, 6),
                distortion_tolerance=distortion_tolerance,
                candidates_evaluated=evaluated,
                reason=(
                    f"{method} basis, width {input_dim} → {k} at measured "
                    f"worst-case distortion {distortion:.4f} ≤ "
                    f"{distortion_tolerance:.3g}"
                ),
            )

    return X.copy(), ReductionReport(
        verdict="passed_through",
        input_dim=input_dim,
        output_dim=input_dim,
        effective_rank=er,
        distortion_observed=None,
        distortion_tolerance=distortion_tolerance,
        candidates_evaluated=evaluated,
        reason=(
            f"no width held worst-case distortion within "
            f"{distortion_tolerance:.3g} using the {method} basis; "
            f"structure does not support reduction"
        ),
    )
