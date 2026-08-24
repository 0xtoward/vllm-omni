# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-data schema contract for sealed MiniCPM-o runtime prompts.

This module intentionally imports neither torch nor audio libraries.  The live
producer, the direct G0/G1 oracle, and the experiment controller all consume
this exact validator so schema drift is detected before graph capture.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "minicpmo45-runtime-prompt-manifest-v2"
SOURCE_SCHEMA = "minicpmo45-default-prompt-source-v1"
COUNT = 1
BENCHMARK_SOURCE_ROLES = {
    "benchmark_patch",
    "benchmark_wrapper",
    "dataset_loader",
}
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and not (set(text) - _HEX)


def validate_runtime_prompt_manifest_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate the v2 structural/identity contract without touching files."""
    rows = payload.get("rows")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("sealed") is not True
        or not isinstance(rows, list)
        or len(rows) != COUNT
    ):
        raise RuntimeError("runtime prompt manifest is not sealed v2 model-default-only")
    if not str(payload.get("source_manifest_path", "")) or not _is_sha256(payload.get("source_manifest_sha256")):
        raise RuntimeError("runtime prompt manifest source identity is incomplete")
    dataset = payload.get("dataset")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("locale") != "zh"
        or not str(dataset.get("dataset_root", ""))
        or not str(dataset.get("metadata_path", ""))
        or not _is_sha256(dataset.get("metadata_sha256"))
    ):
        raise RuntimeError("runtime prompt manifest dataset contract is incomplete")
    sources = payload.get("benchmark_sources")
    if (
        not isinstance(sources, list)
        or len(sources) != len(BENCHMARK_SOURCE_ROLES)
        or {str(row.get("role", "")) for row in sources if isinstance(row, Mapping)} != BENCHMARK_SOURCE_ROLES
    ):
        raise RuntimeError("runtime prompt manifest benchmark sources are incomplete")
    for source in sources:
        if not isinstance(source, Mapping) or not str(source.get("path", "")) or not _is_sha256(source.get("sha256")):
            raise RuntimeError("runtime prompt manifest benchmark source is invalid")

    result: list[dict[str, Any]] = []
    utterance_ids: set[str] = set()
    cache_keys: set[str] = set()
    identities: set[str] = set()
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or int(raw.get("row_ordinal", -1)) != ordinal:
            raise RuntimeError("runtime prompt manifest ordinals are not contiguous")
        utterance_id = str(raw.get("utterance_id", ""))
        cache_key = str(raw.get("cache_key", ""))
        identity = str(raw.get("runtime_prompt_identity_sha256", ""))
        required_hashes = (
            "ref_file_sha256",
            "decoded_f32_sha256",
            "decoded_f32_payload_sha256",
            "expected_cache_key",
            "runtime_prompt_identity_sha256",
            "canonical_wav_sha256",
            "prompt_mel_sha256",
        )
        if (
            not utterance_id
            or utterance_id in utterance_ids
            or not str(raw.get("ref_path", ""))
            or not str(raw.get("canonical_wav_path", ""))
            or int(raw.get("decoded_sample_rate", 0)) <= 0
            or int(raw.get("decoded_samples", 0)) <= 0
            or int(raw.get("prompt_frames", 0)) <= 0
            or any(not _is_sha256(raw.get(key)) for key in required_hashes)
            or cache_key != raw.get("expected_cache_key")
            or cache_key in cache_keys
            or identity in identities
        ):
            raise RuntimeError(f"runtime prompt manifest row {ordinal} is invalid")
        utterance_ids.add(utterance_id)
        cache_keys.add(cache_key)
        identities.add(identity)
        result.append(dict(raw))
    return tuple(result)


def load_runtime_prompt_manifest_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("runtime prompt manifest must be a JSON object")
    validate_runtime_prompt_manifest_payload(payload)
    return dict(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    load_runtime_prompt_manifest_payload(args.manifest.resolve(strict=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BENCHMARK_SOURCE_ROLES",
    "COUNT",
    "SCHEMA",
    "SOURCE_SCHEMA",
    "load_runtime_prompt_manifest_payload",
    "validate_runtime_prompt_manifest_payload",
]
