# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-data identity contract for MiniCPM-o runtime voice prompts.

This module deliberately has no torch/numpy/soundfile dependency.  The
benchmark manifest builder and the live Token2Wav producer both feed it the
exact contiguous float32 payload bytes produced by their respective decode
paths.  Equality of the resulting digest therefore joins a live observation to
one source-manifest row without relying on request arrival order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

IDENTITY_SCHEMA = "minicpmo45-runtime-prompt-identity-v1"


def prompt_identity_from_f32_bytes(
    payload: bytes,
    *,
    sample_rate: int,
    samples: int,
) -> dict[str, Any]:
    """Return the immutable identity used by source and live prompt paths."""
    sample_rate = int(sample_rate)
    samples = int(samples)
    if sample_rate <= 0 or samples <= 0:
        raise ValueError("runtime prompt identity requires positive rate and samples")
    if len(payload) != samples * 4:
        raise ValueError(
            "runtime prompt identity expects contiguous float32 payload bytes "
            f"expected={samples * 4} actual={len(payload)}"
        )
    cache_digest = hashlib.sha256()
    cache_digest.update(payload)
    cache_digest.update(str(sample_rate).encode("ascii"))
    decoded_payload_sha256 = hashlib.sha256(payload).hexdigest()
    contract = {
        "schema": IDENTITY_SCHEMA,
        "decoded_sample_rate": sample_rate,
        "decoded_samples": samples,
        "decoded_f32_payload_sha256": decoded_payload_sha256,
        "expected_cache_key": cache_digest.hexdigest(),
    }
    identity_digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**contract, "runtime_prompt_identity_sha256": identity_digest}


def build_source_identity_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a unique digest-to-row index, rejecting ambiguous source WAVs."""
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        digest = str(row.get("runtime_prompt_identity_sha256", ""))
        if len(digest) != 64:
            raise RuntimeError("source reference lacks a decoded runtime identity")
        if digest in result:
            raise RuntimeError("source reference runtime identity is duplicated")
        result[digest] = dict(row)
    return result


def resolve_source_identity(
    index: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one live identity and verify every precomputed identity field."""
    digest = str(observed.get("runtime_prompt_identity_sha256", ""))
    source = index.get(digest)
    if source is None:
        raise RuntimeError("live runtime prompt does not match a source-manifest WAV")
    for key in (
        "decoded_sample_rate",
        "decoded_samples",
        "decoded_f32_payload_sha256",
        "expected_cache_key",
        "runtime_prompt_identity_sha256",
    ):
        if source.get(key) != observed.get(key):
            raise RuntimeError(
                f"live runtime prompt identity mismatch key={key} "
                f"expected={source.get(key)!r} actual={observed.get(key)!r}"
            )
    return dict(source)


__all__ = [
    "IDENTITY_SCHEMA",
    "build_source_identity_index",
    "prompt_identity_from_f32_bytes",
    "resolve_source_identity",
]
