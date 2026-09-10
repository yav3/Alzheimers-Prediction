"""Smoke tests for the vendored sizing bridge.

The package is vendored from lotwhitelabelnt `backend/app/bridge/` via
`scripts/agent/sync_bridge.py` in that repository. These tests check the two
things checkable from inside this one: the copy still behaves, and it still
declares where it came from so nobody edits it in place.

Run:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sizing_bridge import search_reduction, size_problem  # noqa: E402

BRIDGE_DIR = Path(__file__).resolve().parents[1] / "sizing_bridge"


def test_every_vendored_file_declares_its_source():
    modules = sorted(BRIDGE_DIR.glob("*.py"))
    assert modules, "vendored sizing_bridge package is missing"
    for path in modules:
        head = path.read_text()[:400]
        assert "Vendored from lotwhitelabelnt" in head, f"{path.name} lost its header"


def test_a_redundant_feature_table_is_reduced_to_its_latent_width():
    rng = np.random.default_rng(7)
    table = rng.normal(size=(300, 6)) @ rng.normal(size=(6, 30))

    _, report = search_reduction(table, distortion_tolerance=0.10)

    assert report.verdict == "reduced"
    assert report.output_dim == 6


def test_unstructured_features_are_not_reduced():
    table = np.random.default_rng(1).normal(size=(120, 15))

    _, report = search_reduction(table, distortion_tolerance=0.05)

    assert report.verdict == "passed_through"


def test_audit_handles_a_dirty_export_without_dropping_it_silently():
    rng = np.random.default_rng(7)
    table = rng.normal(size=(120, 6)) @ rng.normal(size=(6, 12))
    table = np.column_stack([table, np.zeros(120)])
    table[0, 0] = np.inf

    _, report = size_problem(table, distortion_tolerance=0.10)

    assert report.audit.n_nonfinite_replaced == 1
    assert [d.reason for d in report.audit.dropped_columns] == ["all-zero"]


def test_runs_are_reproducible():
    rng = np.random.default_rng(7)
    table = rng.normal(size=(120, 6)) @ rng.normal(size=(6, 12))

    _, first = size_problem(table, distortion_tolerance=0.10)
    _, second = size_problem(table, distortion_tolerance=0.10)

    assert first.manifest_id == second.manifest_id


def test_the_example_script_runs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import size_features

    assert size_features.main() == 0


# ── Modules added when the reference package deepened ────────────────────────
#
# The sync vendors the whole package rather than a subset, so this repository
# now carries the quantum-chemistry modules (Hamiltonians, resource estimation,
# shot allocation) alongside the sizing it actually uses. They are not relevant
# to an ADNI feature table and are not exercised beyond importing cleanly —
# vendoring a partial package would be worse, because then "matches source"
# would stop meaning anything.

def test_the_whole_vendored_package_imports():
    """A partial or broken sync shows up here rather than at first use."""
    import sizing_bridge

    for name in (
        "audit_matrix",
        "search_reduction",
        "size_problem",
        "select_active_space",
        "estimate_resources",
        "allocate_shots",
        "Campaign",
    ):
        assert hasattr(sizing_bridge, name), f"{name} missing from vendored package"


def test_campaign_memory_works_on_repeated_fits():
    """The one new module that earns its place here.

    A campaign keyed by the problem lets repeated fits on the same feature
    table accumulate what earlier runs measured, instead of each fit starting
    from the same bound.
    """
    from sizing_bridge import Campaign, MeasurementGroup, ProblemKey, allocate_shots

    groups = [
        MeasurementGroup("volumetric", coefficient=2.0, variance=1.0),
        MeasurementGroup("cognitive", coefficient=1.0, variance=1.0),
    ]
    cold = allocate_shots(groups, budget=10_000)

    campaign = Campaign(key=ProblemKey("adni-panel", n_electrons=4, n_orbitals=6))
    campaign.seed(groups)
    for _ in range(4):
        campaign.observe({"volumetric": 0.05, "cognitive": 0.9})

    warm = allocate_shots(campaign.groups(), budget=10_000)

    assert warm.shots["volumetric"] < cold.shots["volumetric"]
    assert campaign.summary()["n_runs"] == 4
