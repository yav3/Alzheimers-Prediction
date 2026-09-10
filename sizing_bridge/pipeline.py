# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/pipeline.py at 55e18a0.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""The bridge, end to end.

One entry point that composes the stages a hybrid workload needs before a
circuit exists:

    audit  →  reduce (or refuse)  →  forecast  →  manifest

Everything is classical, runs before job submission, and is never in the
latency path of anything. The output is a specification plus the record of
how it was reached.
"""

from __future__ import annotations

import numpy as np

from .audit import audit_matrix
from .manifest import build_manifest, fingerprint
from .reduce import search_reduction
from .spec import CostForecast, SizingReport

__all__ = ["size_problem", "forecast_cost"]

DEFAULT_ANALYSIS = "bridge.size_problem"


def forecast_cost(input_dim: int, output_dim: int, exponent: float) -> CostForecast:
    """What a width change is worth at a caller-supplied cost exponent.

    The exponent belongs to the caller's workload — for a hybrid chemistry
    post-processing step it is the published scaling exponent. This is
    arithmetic on their number, not a claim about it.
    """
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("dimensions must be positive")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    ratio = output_dim / input_dim
    multiplier = ratio**exponent
    return CostForecast(
        exponent=exponent,
        cost_multiplier=multiplier,
        speedup=(1.0 / multiplier) if multiplier > 0 else float("inf"),
    )


def size_problem(
    X: np.ndarray,
    labels: list[str] | None = None,
    *,
    distortion_tolerance: float = 0.15,
    floor_dim: int = 2,
    method: str = "spectral",
    cost_exponent: float | None = None,
    seed: int = 0,
    analysis: str = DEFAULT_ANALYSIS,
) -> tuple[np.ndarray, SizingReport]:
    """Audit, size and record a problem before anything expensive runs.

    Returns the sized matrix and a full report. The report always carries a
    verdict — ``reduced``, ``passed_through`` or ``refused`` — and a manifest
    id that re-derives from identical inputs.

    ``cost_exponent`` is optional; supply the caller's own scaling exponent to
    get the forecast alongside the size.
    """
    input_fp = fingerprint(
        {"shape": list(np.shape(X)), "data": np.asarray(X, dtype=float).round(9).tolist()}
    )

    clean, kept_labels, audit = audit_matrix(X, labels)
    sized, reduction = search_reduction(
        clean,
        distortion_tolerance=distortion_tolerance,
        floor_dim=floor_dim,
        method=method,  # type: ignore[arg-type]
        seed=seed,
    )

    forecast = None
    if cost_exponent is not None and reduction.reduced:
        forecast = forecast_cost(
            reduction.input_dim, reduction.output_dim, cost_exponent
        )

    output_payload = {
        "audit": audit.as_dict(),
        "reduction": reduction.as_dict(),
        "forecast": forecast.as_dict() if forecast else None,
        "labels_out": kept_labels,
    }
    manifest = build_manifest(
        analysis=analysis,
        parameters={
            "distortion_tolerance": distortion_tolerance,
            "floor_dim": floor_dim,
            "method": method,
            "cost_exponent": cost_exponent,
        },
        seed=seed,
        input_fingerprint=input_fp,
        stages=[
            {"stage": "audit", **audit.as_dict()},
            {"stage": "reduce", **reduction.as_dict()},
        ],
        output_payload=output_payload,
    )

    return sized, SizingReport(
        audit=audit, reduction=reduction, forecast=forecast, manifest=manifest
    )
