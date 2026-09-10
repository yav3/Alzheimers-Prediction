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

from sizing_bridge import (  # noqa: E402
    certify_reduction,
    search_reduction,
    size_problem,
    solve_casci,
)
from sizing_bridge.hamiltonian import ElectronicHamiltonian  # noqa: E402

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


def _small_hamiltonian(n_orbitals: int, n_electrons: int, seed: int, core: float = 0.0):
    """A random but physically shaped Hamiltonian: symmetric h, 8-fold ERI."""
    rng = np.random.default_rng(seed)
    one_body = rng.normal(size=(n_orbitals, n_orbitals))
    one_body = 0.5 * (one_body + one_body.T)
    factor = rng.normal(size=(n_orbitals, n_orbitals, 3))
    factor = 0.5 * (factor + factor.transpose(1, 0, 2))
    two_body = np.einsum("pqx,rsx->pqrs", factor, factor)
    return ElectronicHamiltonian(
        one_body=one_body,
        two_body=two_body,
        n_electrons=n_electrons,
        core_energy=core,
    )


def test_the_solver_is_blind_to_a_change_of_orbital_basis():
    """A rotation is bookkeeping, not physics, so the energy must not move.

    This is the cheapest property that a mis-vendored solver fails: it
    exercises the excitation signs, the integral transform and the Davidson
    path at once, and it needs no reference data to check against.
    """
    hamiltonian = _small_hamiltonian(4, 4, seed=3, core=1.1)
    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(4, 4)))

    plain = solve_casci(hamiltonian).energy
    rotated = solve_casci(hamiltonian.transform(rotation)).energy
    assert abs(plain - rotated) < 1e-9


def test_the_core_energy_shifts_the_answer_and_nothing_else():
    without = solve_casci(_small_hamiltonian(4, 4, seed=5, core=0.0)).energy
    with_core = solve_casci(_small_hamiltonian(4, 4, seed=5, core=-2.5)).energy
    assert abs(with_core - (without - 2.5)) < 1e-9


def _hamiltonian_with_one_decoupled_orbital(seed: int):
    """Four interacting orbitals plus a fifth that touches nothing.

    A random Hamiltonian is the wrong fixture for a truncation test: random
    integrals couple every orbital to every other, so no orbital is safe to
    drop and a correct selector will decline to drop one. Here the fifth
    orbital carries no integral connecting it to the rest and sits high in
    energy, so removing it is exactly right and the error should be zero
    rather than merely small.
    """
    interacting = _small_hamiltonian(4, 4, seed)
    one_body = np.zeros((5, 5))
    one_body[:4, :4] = interacting.one_body
    one_body[4, 4] = 5.0
    two_body = np.zeros((5,) * 4)
    two_body[:4, :4, :4, :4] = interacting.two_body
    two_body[4, 4, 4, 4] = 1.0
    return ElectronicHamiltonian(
        one_body=one_body, two_body=two_body, n_electrons=4
    )


def test_a_reduction_is_certified_against_the_untruncated_answer():
    """The end-to-end path: rank, select, project, solve, compare."""
    hamiltonian = _hamiltonian_with_one_decoupled_orbital(seed=6)
    occupations = np.diag([1.98, 1.90, 0.10, 0.02, 0.0])

    certificate = certify_reduction(hamiltonian, occupations)

    assert certificate.certified, certificate.reason
    assert certificate.space.n_orbitals < hamiltonian.n_orbitals
    assert certificate.qubits_saved > 0
    assert abs(certificate.error_hartree) < 1e-9
    assert "kcal/mol" in certificate.summary()


def test_a_space_that_drops_a_correlated_orbital_is_rejected_on_the_way():
    """The rejected candidates are the evidence, so they are kept."""
    hamiltonian = _hamiltonian_with_one_decoupled_orbital(seed=6)
    occupations = np.diag([1.98, 1.90, 0.10, 0.02, 0.0])

    certificate = certify_reduction(hamiltonian, occupations)

    degraded = [c for c in certificate.candidates if c.status == "degraded"]
    assert degraded, "a three-orbital space should not have survived"
    assert all(abs(c.error_hartree) > 1e-3 for c in degraded)


def test_a_reduction_that_cannot_be_checked_is_refused_not_assumed():
    hamiltonian = _hamiltonian_with_one_decoupled_orbital(seed=7)
    occupations = np.diag([1.98, 1.90, 0.10, 0.02, 0.0])

    certificate = certify_reduction(hamiltonian, occupations, max_determinants=4)

    assert certificate.verdict == "refused"
    assert certificate.space is None
