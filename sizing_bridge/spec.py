# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/spec.py at bba1add.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Typed results for the classical→quantum sizing bridge.

Every stage reports what it did, what it removed and why, so a caller can
defend the specification it hands downstream. Nothing here is domain-specific:
the bridge operates on a labelled numeric matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Verdict = Literal["reduced", "passed_through", "refused"]


@dataclass(frozen=True)
class DroppedColumn:
    """One column removed by the audit, with the reason it was removed."""

    label: str
    index: int
    reason: str


@dataclass
class AuditReport:
    """Result of the pre-flight audit. Nothing is silently discarded."""

    n_rows_in: int
    n_cols_in: int
    n_rows_out: int
    n_cols_out: int
    dropped_columns: list[DroppedColumn] = field(default_factory=list)
    n_nonfinite_replaced: int = 0
    n_duplicate_labels_merged: int = 0
    clean: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows_in": self.n_rows_in,
            "n_cols_in": self.n_cols_in,
            "n_rows_out": self.n_rows_out,
            "n_cols_out": self.n_cols_out,
            "n_nonfinite_replaced": self.n_nonfinite_replaced,
            "n_duplicate_labels_merged": self.n_duplicate_labels_merged,
            "clean": self.clean,
            "dropped_columns": [
                {"label": d.label, "index": d.index, "reason": d.reason}
                for d in self.dropped_columns
            ],
        }


@dataclass
class ReductionReport:
    """Result of the sizing search — a reduction, a pass-through, or a refusal."""

    verdict: Verdict
    input_dim: int
    output_dim: int
    effective_rank: float
    distortion_observed: float | None
    distortion_tolerance: float
    reason: str
    candidates_evaluated: int = 0

    @property
    def reduced(self) -> bool:
        return self.verdict == "reduced"

    @property
    def ratio(self) -> float:
        """output_dim / input_dim. 1.0 when nothing was reduced."""
        return self.output_dim / self.input_dim if self.input_dim else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "ratio": round(self.ratio, 6),
            "effective_rank": self.effective_rank,
            "distortion_observed": self.distortion_observed,
            "distortion_tolerance": self.distortion_tolerance,
            "candidates_evaluated": self.candidates_evaluated,
            "reason": self.reason,
        }


@dataclass
class CostForecast:
    """What the sizing is worth at a caller-supplied cost exponent.

    The exponent is the caller's, not ours — for a hybrid chemistry workload
    it is the published post-processing exponent. We only do the arithmetic.
    """

    exponent: float
    cost_multiplier: float
    speedup: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "exponent": self.exponent,
            "cost_multiplier": round(self.cost_multiplier, 6),
            "speedup": round(self.speedup, 6),
        }


@dataclass
class SizingReport:
    """The bridge's whole output: audit, reduction, forecast and a run identity."""

    audit: AuditReport
    reduction: ReductionReport
    forecast: CostForecast | None
    manifest: dict[str, Any]

    @property
    def manifest_id(self) -> str:
        return str(self.manifest.get("manifest_id", ""))

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "audit": self.audit.as_dict(),
            "reduction": self.reduction.as_dict(),
            "forecast": self.forecast.as_dict() if self.forecast else None,
        }
