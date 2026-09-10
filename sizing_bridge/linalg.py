# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/linalg.py at 4a3e245.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Portable linear-algebra primitives for the bridge.

These are the Z.5 (Johnson–Lindenstrauss projection) and Z.6 (effective rank)
operators. `app/engines/grn_pack_b/distinguishability.py` holds the canonical
reduction-to-practice of those operators for the B2 claim slot and is the
legal artifact; it is not modified by the bridge and must not be.

They are restated here so the bridge depends on numpy and nothing else, which
is what lets the same package be vendored into repositories that do not carry
the engines tree. `tests/test_bridge.py` asserts the two implementations agree
numerically, so a change to either without the other fails the suite rather
than drifting quietly.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["effective_rank", "jl_min_k", "jl_project"]


def jl_min_k(n: int, epsilon: float) -> int:
    """Achlioptas (2003) lower bound: k ≥ (4 / (ε²/2 − ε³/3)) · ln n."""
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must lie in (0, 1)")
    denom = (epsilon**2) / 2.0 - (epsilon**3) / 3.0
    return int(math.ceil((4.0 * math.log(max(n, 2))) / denom))


def jl_project(X: np.ndarray, k: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Project X (n, d) to (n, k) through a Gaussian R (d, k), scale 1/√k.

    Returns (X_proj, R) so a caller can re-apply the identical projection.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    rng = np.random.default_rng(seed)
    R = rng.normal(loc=0.0, scale=1.0 / math.sqrt(k), size=(X.shape[1], k))
    return X @ R, R


def effective_rank(X: np.ndarray, eps: float = 1e-12) -> float:
    """exp(Shannon entropy of the normalised singular spectrum)."""
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if X.size == 0:
        return 0.0
    singular = np.linalg.svd(X, compute_uv=False)
    total = float(np.sum(singular))
    if total <= 0.0:
        return 0.0
    p = singular / total
    entropy = -float(np.sum(p * np.log(p + eps)))
    return float(np.exp(entropy))
