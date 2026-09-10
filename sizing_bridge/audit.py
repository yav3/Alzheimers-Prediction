# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/audit.py at 0f51295.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Pre-flight audit — domain-neutral.

`app/../methods/expression_io.py` in the sibling platform does this well but
assumes gene identifiers. The bridge must audit an active-space integral
matrix, a portfolio covariance or a subgraph incidence matrix just as
readily, so this is the same discipline with the domain assumptions removed.

Rule that governs every branch: nothing is discarded silently. Every removed
column comes back with a reason attached.
"""

from __future__ import annotations

import numpy as np

from .spec import AuditReport, DroppedColumn

__all__ = ["audit_matrix"]

_CONSTANT_TOL = 1e-12


def audit_matrix(
    X: np.ndarray,
    labels: list[str] | None = None,
    *,
    drop_constant: bool = True,
    max_missing_fraction: float = 0.5,
) -> tuple[np.ndarray, list[str], AuditReport]:
    """Audit and clean a 2-D numeric matrix before anything expensive runs.

    Returns the cleaned matrix, its surviving labels, and a report. The report
    is the product as much as the matrix is: a caller that cannot say what was
    removed cannot defend the result computed from what remained.

    Order matters — non-finite values become missing first, so the
    missingness and constancy tests see the same view of the data.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n_rows_in, n_cols_in = X.shape
    if labels is None:
        labels = [f"c{i}" for i in range(n_cols_in)]
    if len(labels) != n_cols_in:
        raise ValueError(
            f"labels has {len(labels)} entries for {n_cols_in} columns"
        )

    work = np.asarray(X, dtype=float).copy()

    # 1. Non-finite values (Inf, -Inf, NaN) become missing, and are counted.
    nonfinite = ~np.isfinite(work)
    n_nonfinite = int(nonfinite.sum())
    work[nonfinite] = np.nan

    dropped: list[DroppedColumn] = []
    keep = np.ones(n_cols_in, dtype=bool)

    for j in range(n_cols_in):
        col = work[:, j]
        missing_fraction = float(np.isnan(col).mean()) if col.size else 1.0
        if missing_fraction > max_missing_fraction:
            keep[j] = False
            dropped.append(
                DroppedColumn(labels[j], j, f"missing {missing_fraction:.0%} of values")
            )
            continue
        finite = col[~np.isnan(col)]
        if finite.size == 0:
            keep[j] = False
            dropped.append(DroppedColumn(labels[j], j, "no finite values"))
            continue
        if drop_constant and float(np.ptp(finite)) <= _CONSTANT_TOL:
            value = float(finite[0])
            reason = "all-zero" if abs(value) <= _CONSTANT_TOL else "constant"
            keep[j] = False
            dropped.append(DroppedColumn(labels[j], j, reason))

    work = work[:, keep]
    kept_labels = [lab for j, lab in enumerate(labels) if keep[j]]

    # 2. Duplicate labels are aggregated by mean rather than dropped — a
    #    repeated label is usually a repeated measurement, not an error.
    work, kept_labels, n_merged = _aggregate_duplicate_labels(work, kept_labels)

    # 3. Remaining missing values are imputed by column mean so the matrix is
    #    finite for the reduction stage. Rows are never removed here: the
    #    caller's row semantics (walkers, samples, trials) are not ours to edit.
    if work.size:
        col_means = np.nanmean(work, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(np.isnan(work))
        work[inds] = np.take(col_means, inds[1])

    report = AuditReport(
        n_rows_in=n_rows_in,
        n_cols_in=n_cols_in,
        n_rows_out=int(work.shape[0]),
        n_cols_out=int(work.shape[1]),
        dropped_columns=dropped,
        n_nonfinite_replaced=n_nonfinite,
        n_duplicate_labels_merged=n_merged,
        clean=(n_nonfinite == 0 and not dropped and n_merged == 0),
    )
    return work, kept_labels, report


def _aggregate_duplicate_labels(
    X: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, list[str], int]:
    """Collapse duplicate-labelled columns by nan-aware mean, order preserved."""
    seen: dict[str, list[int]] = {}
    for j, lab in enumerate(labels):
        seen.setdefault(lab, []).append(j)
    if all(len(v) == 1 for v in seen.values()):
        return X, labels, 0

    out_cols: list[np.ndarray] = []
    out_labels: list[str] = []
    merged = 0
    for lab in dict.fromkeys(labels):
        idx = seen[lab]
        if len(idx) == 1:
            out_cols.append(X[:, idx[0]])
        else:
            with np.errstate(invalid="ignore"):
                out_cols.append(np.nanmean(X[:, idx], axis=1))
            merged += len(idx) - 1
        out_labels.append(lab)
    return np.column_stack(out_cols), out_labels, merged
