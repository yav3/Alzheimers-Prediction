# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/__init__.py at 4a3e245.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Classical→quantum bridge: everything a hybrid job needs before it is submitted.

    hamiltonian   FCIDUMP in and out, active-space projection with the
                  frozen-core fold, natural occupations from a 1-RDM
    active_space  selection under a measured bound, with a refusal and
                  chemical-validity enforcement
    resources     qubits, Pauli terms counted not assumed, one-norm, shot
                  budget for a target precision, device feasibility
    allocation    Neyman allocation of a finite shot budget, against the
                  uniform baseline it has to beat
    campaign      run-to-run accumulation keyed by the problem, so a later
                  job allocates against evidence rather than a bound

Plus the domain-neutral pipeline the package started as:

    audit_matrix     domain-neutral pre-flight QC; nothing dropped silently
    search_reduction B2 projection lens driven by a classical search, with an
                     honest refusal when the structure does not compress
    forecast_cost    what the width change is worth at the caller's exponent
    build_manifest   hash-addressed run identity, clock excluded

Claim correspondence is declared in `app.engines.patent_registry`, not here —
that registry is the single source of truth and this package is one of its
code anchors.

`linalg.py` restates the B2 Z.5/Z.6 operators so this package depends on
numpy alone and can be vendored into repositories without the engines tree.
The canonical claim RTP stays in `app/engines/grn_pack_b/distinguishability.py`
and is unmodified; a test pins the two to agree.

Known consolidation debt, deliberately not resolved in this change:
  * `manifest.py` duplicates the sibling platform's `runs.build_manifest`.
    They should become one module with one manifest version.
  * The sizing search is a geometric sweep. The response-surface search in
    `services/ml-engine/app/core/doe_engine.py` is the better instrument and
    is not importable from here today.
"""

from .active_space import (
    ActiveSpace,
    SelectionReport,
    determinant_count,
    qubit_count,
    select_active_space,
)
from .allocation import (
    AllocationReport,
    MeasurementGroup,
    allocate_shots,
    uniform_allocation,
)
from .audit import audit_matrix
from .campaign import Campaign, CampaignStore, GroupBelief, ProblemKey
from .hamiltonian import (
    ElectronicHamiltonian,
    natural_occupations,
    parse_fcidump,
    read_fcidump,
    write_fcidump,
)
from .linalg import effective_rank, jl_min_k, jl_project
from .manifest import build_manifest
from .pipeline import forecast_cost, size_problem
from .reduce import candidate_dims, search_reduction, worst_pairwise_distortion
from .resources import (
    DeviceModel,
    FeasibilityReport,
    ResourceEstimate,
    check_feasibility,
    estimate_resources,
    one_norm,
    shots_for_precision,
)
from .spec import (
    AuditReport,
    CostForecast,
    DroppedColumn,
    ReductionReport,
    SizingReport,
)

__all__ = [
    "ActiveSpace",
    "AllocationReport",
    "AuditReport",
    "Campaign",
    "CampaignStore",
    "DeviceModel",
    "ElectronicHamiltonian",
    "FeasibilityReport",
    "GroupBelief",
    "MeasurementGroup",
    "ProblemKey",
    "ResourceEstimate",
    "SelectionReport",
    "allocate_shots",
    "check_feasibility",
    "determinant_count",
    "estimate_resources",
    "natural_occupations",
    "one_norm",
    "parse_fcidump",
    "qubit_count",
    "read_fcidump",
    "select_active_space",
    "shots_for_precision",
    "uniform_allocation",
    "write_fcidump",
    "CostForecast",
    "DroppedColumn",
    "ReductionReport",
    "SizingReport",
    "audit_matrix",
    "build_manifest",
    "effective_rank",
    "candidate_dims",
    "forecast_cost",
    "jl_min_k",
    "jl_project",
    "search_reduction",
    "size_problem",
    "worst_pairwise_distortion",
]
