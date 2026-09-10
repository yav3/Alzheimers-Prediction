# ── vendored ──
# Vendored from lotwhitelabelnt backend/app/bridge/manifest.py at 55e18a0.
# Do not edit here. Change the source, then re-run:
#     python3 scripts/agent/sync_bridge.py <this directory>
# Verify with --check. See app/bridge/__init__.py for the contract.
# ── vendored ──

"""Run identity for a sizing decision.

Mirrors the manifest contract already used by the sibling platform's
`runs.build_manifest`: the id is a SHA-256 over the hash-relevant fields, and
wall-clock time is attached *after* the id is computed so it never perturbs
identity. Same inputs, same id.

Kept deliberately small and dependency-free. See ARCHITECTURE note in
`app/bridge/__init__.py` on unifying this with the sibling implementation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from typing import Any

MANIFEST_VERSION = "bridge-1"

__all__ = ["build_manifest", "fingerprint", "canonical_json"]


def canonical_json(obj: Any) -> str:
    """Stable JSON — sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _runtime_fingerprint() -> dict[str, str]:
    import platform

    import numpy as np

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_sha": _git_sha(),
    }


def build_manifest(
    *,
    analysis: str,
    parameters: dict[str, Any],
    seed: int | None,
    input_fingerprint: str,
    stages: list[dict[str, Any]],
    output_payload: dict[str, Any],
    created_at: float | None = None,
) -> dict[str, Any]:
    """Assemble the manifest; `manifest_id` is the hash of everything but the clock."""
    hashable = {
        "manifest_version": MANIFEST_VERSION,
        "analysis": analysis,
        "parameters": parameters,
        "seed": seed,
        "input_fingerprint": input_fingerprint,
        "stages": stages,
        "output_fingerprint": fingerprint(output_payload),
        "runtime": _runtime_fingerprint(),
    }
    return {
        "manifest_id": "bridge_" + sha256_hex(canonical_json(hashable)),
        "created_at": created_at if created_at is not None else time.time(),
        **hashable,
    }
