# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/campaign.py at c002365.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Run-to-run accumulation: making the hundredth job cheaper than the first.

A hybrid workload today starts cold every time. A circuit is prepared, samples
are drawn, an energy comes back, and what the run established about the system
— which terms actually carry variance, which groups were over-measured —
evaporates into a notebook. Run one hundred spends its budget the same way run
one did.

This module is the memory. Observed variances are folded into a running
estimate per measurement group, keyed by a fingerprint of the *problem* rather
than of the run, so a later job on the same Hamiltonian and active space
inherits what earlier jobs measured and allocates against evidence instead of
against a bound.

That is claims 2 to 5 of the training family in operational form: estimate
where confident, check the estimate against real measurements, and replace the
estimate with the real value as it arrives. The check is not optional here —
`observe` records both the prior and the observation, and `drift` reports where
the accumulated belief and the latest evidence disagree, because a memory that
cannot notice it has gone stale is worse than no memory.

Accumulation is a shrinkage estimator: with n prior observations of a group,
a new one updates the mean with weight 1/(n+1). Early observations therefore
move the estimate a lot and later ones refine it, which is the behaviour a
campaign wants — and it means a single anomalous run cannot capture the belief
once a campaign has history.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .allocation import MeasurementGroup
from .manifest import fingerprint

__all__ = ["Campaign", "CampaignStore", "GroupBelief", "ProblemKey"]

#: Fractional gap that counts as drift when the observations have no spread.
_RELATIVE_DRIFT = 0.5


@dataclass(frozen=True)
class ProblemKey:
    """What makes two runs the same problem.

    Deliberately not the run: two jobs differing only in shot budget or seed
    describe the same physics and should share what they learned. Changing the
    Hamiltonian or the active space makes a different problem and starts a
    different memory.
    """

    hamiltonian_fingerprint: str
    n_electrons: int
    n_orbitals: int
    label: str = ""

    def as_id(self) -> str:
        return fingerprint(
            {
                "h": self.hamiltonian_fingerprint,
                "n_e": self.n_electrons,
                "n_o": self.n_orbitals,
            }
        )[:32]


@dataclass
class GroupBelief:
    """The accumulated variance estimate for one measurement group."""

    label: str
    coefficient: float
    variance_mean: float
    n_observations: int = 0
    #: Running sum of squared deviations, for a spread estimate.
    m2: float = 0.0
    last_observed: float | None = None
    #: How far the last observation sat from the belief it arrived at, in
    #: standard deviations, measured *before* it was folded in. Recorded at
    #: observation time because afterwards the outlier has moved both the mean
    #: and the spread it would be compared against.
    last_deviation_sigmas: float | None = None

    @property
    def variance_std(self) -> float:
        """Spread of the observations themselves — how settled the belief is.

        `m2` is clamped at zero: Welford's recurrence can leave it a few ulps
        negative when every observation is identical, and a domain error there
        would take down a caller for the most benign possible input.
        """
        if self.n_observations < 2:
            return math.inf
        return math.sqrt(max(self.m2, 0.0) / (self.n_observations - 1))

    @property
    def settled(self) -> bool:
        """Enough observations, and consistent enough, to allocate against."""
        if self.n_observations < 3:
            return False
        if self.variance_mean <= 0:
            return True
        return self.variance_std / self.variance_mean < 0.5

    def observe(self, variance: float) -> None:
        """Fold one observation in by Welford's method.

        The surprise is scored first, against the belief as it stands. Scoring
        afterwards would compare the observation with a mean it had already
        pulled toward itself, which is how a memory fails to notice it has gone
        stale exactly when the contradiction is largest.
        """
        if variance < 0:
            raise ValueError(f"{self.label}: observed variance must be non-negative")

        self.last_deviation_sigmas = self._surprise(variance)
        self.n_observations += 1
        delta = variance - self.variance_mean
        self.variance_mean += delta / self.n_observations
        self.m2 += delta * (variance - self.variance_mean)
        self.last_observed = variance

    def _surprise(self, variance: float) -> float | None:
        """Deviation of an incoming observation from the current belief, in σ.

        None while the belief is too thin to be surprised by anything. Where
        the observations so far agree exactly, σ is zero and a relative gap
        stands in — that is the case where a contradiction is most obvious,
        not least.
        """
        if not self.settled:
            return None
        spread = self.variance_std
        gap = abs(variance - self.variance_mean)
        if math.isinf(spread):
            return None
        if spread > 0:
            return gap / spread
        scale = max(abs(self.variance_mean), 1e-12)
        return gap / (_RELATIVE_DRIFT * scale)

    def as_group(self) -> MeasurementGroup:
        """Hand this belief to the allocator, labelled with where it came from."""
        return MeasurementGroup(
            label=self.label,
            coefficient=self.coefficient,
            variance=self.variance_mean,
            source="prior_campaign" if self.n_observations else "assumed",
        )


@dataclass
class Campaign:
    """The accumulated state of repeated runs on one problem."""

    key: ProblemKey
    beliefs: dict[str, GroupBelief] = field(default_factory=dict)
    n_runs: int = 0

    def seed(self, groups: list[MeasurementGroup]) -> None:
        """Start from a bound, or extend an existing campaign with new groups.

        Existing beliefs are left alone: seeding must never overwrite what has
        been measured, or a re-seed silently discards the campaign's history.
        """
        for g in groups:
            if g.label not in self.beliefs:
                self.beliefs[g.label] = GroupBelief(
                    label=g.label, coefficient=g.coefficient, variance_mean=g.variance
                )

    def observe(self, observations: dict[str, float]) -> dict[str, tuple[float, float]]:
        """Record measured variances; return {label: (prior, observed)}.

        Returning both is the point. An accumulation that reports only its new
        state gives a caller no way to see that the evidence disagreed with the
        belief, which is exactly when a campaign needs looking at.
        """
        unknown = set(observations) - set(self.beliefs)
        if unknown:
            raise KeyError(f"unseeded measurement groups: {sorted(unknown)}")

        moves: dict[str, tuple[float, float]] = {}
        for label, variance in observations.items():
            belief = self.beliefs[label]
            moves[label] = (belief.variance_mean, variance)
            belief.observe(variance)
        self.n_runs += 1
        return moves

    def drift(self, threshold: float = 2.0) -> list[str]:
        """Groups whose latest observation sits far from the accumulated mean.

        A settled belief that a new run contradicts is a signal the problem
        changed underneath the campaign — a different geometry, a recalibrated
        device — and the memory should be questioned rather than trusted.

        Scored at observation time against the prior belief, so an outlier is
        still detectable after it has been folded in.
        """
        return sorted(
            b.label
            for b in self.beliefs.values()
            if b.last_deviation_sigmas is not None and b.last_deviation_sigmas > threshold
        )

    def groups(self) -> list[MeasurementGroup]:
        """Current beliefs, ready for `allocate_shots`."""
        return [b.as_group() for b in self.beliefs.values()]

    def summary(self) -> dict[str, Any]:
        settled = sum(1 for b in self.beliefs.values() if b.settled)
        return {
            "problem_id": self.key.as_id(),
            "label": self.key.label,
            "n_runs": self.n_runs,
            "n_groups": len(self.beliefs),
            "n_settled": settled,
            "fraction_settled": settled / len(self.beliefs) if self.beliefs else 0.0,
            "drifted": self.drift(),
        }


class CampaignStore:
    """Durable campaigns on disk, one JSON file per problem.

    A directory of plain JSON rather than a database: a campaign is small, and
    a buyer's engineer should be able to read what the layer believes without
    running anything.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: ProblemKey) -> Path:
        return self.root / f"{key.as_id()}.json"

    def load(self, key: ProblemKey) -> Campaign:
        """Load the campaign for this problem, or start an empty one."""
        path = self._path(key)
        if not path.is_file():
            return Campaign(key=key)
        raw = json.loads(path.read_text())
        beliefs = {
            label: GroupBelief(**belief) for label, belief in raw.get("beliefs", {}).items()
        }
        return Campaign(key=key, beliefs=beliefs, n_runs=int(raw.get("n_runs", 0)))

    def save(self, campaign: Campaign) -> Path:
        path = self._path(campaign.key)
        path.write_text(
            json.dumps(
                {
                    "key": asdict(campaign.key),
                    "n_runs": campaign.n_runs,
                    "beliefs": {k: asdict(v) for k, v in campaign.beliefs.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return path

    def list_problems(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))
