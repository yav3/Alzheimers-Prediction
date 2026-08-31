"""Size a feature matrix before fitting an expensive model.

The notebooks in this repository fit multi-class classifiers over ADNI
features. Clinical feature tables are usually more redundant than their column
count suggests — correlated volumetrics, repeated cognitive subscores, derived
ratios — so the width a model actually needs is often well below the width it
is given.

`sizing_bridge` answers that before the fit: it audits the table, searches
classically for the smallest width that holds a measured distortion bound, and
refuses when the data does not support reduction. Every run carries a manifest
id that re-derives from identical inputs, so a reported result traces back to
the exact table and settings that produced it.

Run:  python3 examples/size_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sizing_bridge import size_problem  # noqa: E402


def synthetic_adni_like(
    n_subjects: int = 300, seed: int = 7
) -> tuple[np.ndarray, list[str]]:
    """A stand-in table with the redundancy real clinical panels carry.

    Six independent latent factors — atrophy, cognition, demographics and so on
    — observed through thirty correlated measurements, plus one dead column and
    a missing value, because real exports arrive that way.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_subjects, 6))
    loadings = rng.normal(size=(6, 30))
    table = latent @ loadings + rng.normal(scale=0.05, size=(n_subjects, 30))

    labels = [f"feat_{i:02d}" for i in range(30)]
    table = np.column_stack([table, np.zeros(n_subjects)])
    labels.append("all_zero_export_artifact")
    table[0, 0] = np.nan
    return table, labels


def main() -> int:
    table, labels = synthetic_adni_like()

    reduced, report = size_problem(
        table,
        labels,
        distortion_tolerance=0.10,
        cost_exponent=2.0,  # a fit whose cost grows ~quadratically in width
    )

    audit = report.audit
    print(f"AUDIT     {audit.n_cols_in} columns in, {audit.n_cols_out} out")
    print(f"          {audit.n_nonfinite_replaced} non-finite value(s) handled")
    for dropped in audit.dropped_columns:
        print(f"          dropped {dropped.label!r}: {dropped.reason}")

    reduction = report.reduction
    print(f"\nSIZING    {reduction.verdict}")
    print(f"          {reduction.reason}")
    print(f"          effective rank {reduction.effective_rank:.2f}")

    if report.forecast:
        print(
            f"\nFORECAST  {report.forecast.speedup:.1f}x at exponent "
            f"{report.forecast.exponent}"
        )

    print(f"\nRUN       {report.manifest_id}")
    print(f"          matrix for the model: {reduced.shape}")

    if not reduction.reduced:
        print("\nNothing was reduced — fit on the audited table as-is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
