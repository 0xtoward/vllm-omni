# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed producer/consumer contract for runtime voice prompts.

The producer observes the exact WAV path consumed by ``prepare_prompt`` (or
the in-memory reference-audio materialization path), persists a canonical WAV,
and records the resulting prompt features.  Fresh graph consumers only read
the sealed manifest and verify every live prompt against it; they never
discover or extend the allowlist.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from .runtime_prompt_identity import (
    build_source_identity_index,
    prompt_identity_from_f32_bytes,
    resolve_source_identity,
)
from .runtime_prompt_schema import BENCHMARK_SOURCE_ROLES as _BENCHMARK_SOURCE_ROLES
from .runtime_prompt_schema import COUNT as _COUNT
from .runtime_prompt_schema import SCHEMA, SOURCE_SCHEMA, validate_runtime_prompt_manifest_payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_benchmark_provenance(payload: Mapping[str, Any]) -> None:
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RuntimeError("runtime prompt manifest lacks benchmark dataset provenance")
    metadata = Path(str(dataset.get("metadata_path", ""))).resolve(strict=True)
    if file_sha256(metadata) != dataset.get("metadata_sha256"):
        raise RuntimeError("Seed-TTS metadata changed after prompt sealing")
    sources = payload.get("benchmark_sources")
    if (
        not isinstance(sources, list)
        or len(sources) != len(_BENCHMARK_SOURCE_ROLES)
        or {str(row.get("role", "")) for row in sources if isinstance(row, Mapping)} != _BENCHMARK_SOURCE_ROLES
    ):
        raise RuntimeError("runtime prompt manifest benchmark sources are incomplete")
    for row in sources:
        if not isinstance(row, Mapping):
            raise TypeError("runtime prompt manifest benchmark source must be an object")
        source = Path(str(row.get("path", ""))).resolve(strict=True)
        if file_sha256(source) != row.get("sha256"):
            raise RuntimeError(f"benchmark implementation source changed: {source}")


def _load_identity(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != SOURCE_SCHEMA
        or not isinstance(rows, list)
        or len(rows) != _COUNT
    ):
        raise RuntimeError("runtime prompt source manifest is not model-default-only")
    source_contract = {
        "dataset": {
            "dataset_root": str(payload.get("dataset_root", "")),
            "locale": str(payload.get("locale", "")),
            "metadata_path": str(payload.get("metadata_path", "")),
            "metadata_sha256": str(payload.get("metadata_sha256", "")),
        },
        "benchmark_sources": payload.get("benchmark_sources"),
    }
    _validate_benchmark_provenance(source_contract)
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or int(raw.get("row_ordinal", -1)) != ordinal:
            raise RuntimeError("runtime prompt source manifest ordinals are not contiguous")
        path_value = Path(str(raw.get("ref_path", ""))).resolve(strict=True)
        utterance_id = str(raw.get("utterance_id", ""))
        target_sha = str(raw.get("target_text_sha256", ""))
        if (
            not utterance_id
            or utterance_id in seen_ids
            or path_value in seen_paths
            or raw.get("locale") != "zh"
            or len(target_sha) != 64
        ):
            raise RuntimeError("runtime prompt source row identity is incomplete or repeated")
        seen_ids.add(utterance_id)
        seen_paths.add(path_value)
        digest = file_sha256(path_value)
        if digest != raw.get("ref_file_sha256"):
            raise RuntimeError(f"source reference WAV changed: {path_value}")
        result.append(dict(raw))
    # Build a fail-closed identity index even though the certified profile has
    # one model-default prompt.  This keeps producer and consumer lookup exact.
    build_source_identity_index(result)
    return source_contract, tuple(result)


def load_sealed_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if path.stat().st_mode & 0o222:
        raise RuntimeError(f"runtime prompt manifest must be read-only: {path}")
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha256:
        raise RuntimeError(f"runtime prompt manifest SHA mismatch expected={expected_sha256} actual={actual_sha}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("runtime prompt manifest must be a JSON object")
    rows = validate_runtime_prompt_manifest_payload(payload)
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or int(row.get("row_ordinal", -1)) != ordinal:
            raise RuntimeError("runtime prompt manifest ordinals are invalid")
        canonical = Path(str(row.get("canonical_wav_path", ""))).resolve(strict=True)
        if canonical.stat().st_mode & 0o222:
            raise RuntimeError(f"canonical runtime prompt must be read-only: {canonical}")
        if file_sha256(canonical) != row.get("canonical_wav_sha256"):
            raise RuntimeError(f"canonical runtime prompt changed: {canonical}")
        if row.get("cache_key") != row.get("expected_cache_key"):
            raise RuntimeError("sealed runtime prompt cache key differs from source identity")
    build_source_identity_index(rows)
    _validate_benchmark_provenance(payload)
    return dict(payload)


class RuntimePromptManifest:
    """Record exact live prompts or verify them against a sealed manifest."""

    def __init__(
        self,
        *,
        mode: str,
        manifest_path: Path,
        source_manifest_path: Path | None = None,
        manifest_sha256: str = "",
    ) -> None:
        if mode not in {"producer", "consumer"}:
            raise ValueError(f"invalid runtime prompt manifest mode {mode!r}")
        self.mode = mode
        self.path = manifest_path.expanduser().resolve(strict=False)
        self._pending: dict[str, dict[str, Any]] = {}
        self._request_keys: dict[str, str] = {}
        if mode == "producer":
            if source_manifest_path is None:
                raise RuntimeError("producer requires a fixed request source manifest")
            self._source_path = source_manifest_path.expanduser().resolve(strict=True)
            self._source_sha256 = file_sha256(self._source_path)
            self._source_contract, self._source_rows = _load_identity(self._source_path)
            self._source_by_identity = build_source_identity_index(self._source_rows)
            if self.path.exists():
                raise RuntimeError(f"producer refuses to overwrite prompt manifest: {self.path}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._assets = self.path.parent / "canonical-prompts"
            self._assets.mkdir(parents=False, exist_ok=False)
            self.payload: dict[str, Any] | None = None
        else:
            self.payload = load_sealed_manifest(self.path, manifest_sha256)
            self._source_path = Path(str(self.payload["source_manifest_path"]))
            self._source_sha256 = str(self.payload["source_manifest_sha256"])
            if file_sha256(self._source_path) != self._source_sha256:
                raise RuntimeError("runtime prompt source identity manifest changed")
            self._source_rows = tuple(dict(row) for row in self.payload["rows"])
            self._assets = self.path.parent / "canonical-prompts"
            self._by_cache = {str(row["cache_key"]): dict(row) for row in self.payload["rows"]}
            if len(self._by_cache) != _COUNT:
                raise RuntimeError("sealed runtime prompt manifest repeats cache keys")

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        if self.payload is None:
            return ()
        return tuple(dict(row) for row in self.payload["rows"])

    def observe_materialized(
        self,
        *,
        cache_key: str,
        waveform: torch.Tensor,
        sample_rate: int,
        canonical_wav: Path,
    ) -> str:
        waveform = waveform.detach().cpu().contiguous()
        identity = prompt_identity_from_f32_bytes(
            waveform.numpy().tobytes(),
            sample_rate=int(sample_rate),
            samples=int(waveform.numel()),
        )
        if identity["expected_cache_key"] != cache_key:
            raise RuntimeError("live runtime prompt cache key violates identity contract")
        observed = {
            "cache_key": cache_key,
            "decoded_sample_rate": int(sample_rate),
            "decoded_samples": int(waveform.numel()),
            "decoded_f32_sha256": tensor_sha256(waveform),
            "canonical_wav_sha256": file_sha256(canonical_wav),
            **identity,
        }
        if self.mode == "consumer":
            expected = self._by_cache.get(cache_key)
            if expected is None:
                raise RuntimeError("live runtime prompt is absent from sealed manifest")
            for key, actual in observed.items():
                if expected.get(key) != actual:
                    raise RuntimeError(
                        f"live runtime prompt mismatch key={key} expected={expected.get(key)!r} actual={actual!r}"
                    )
            return cache_key

        entry = self._pending.get(cache_key)
        if entry is not None:
            for key, actual in observed.items():
                if entry.get(key) != actual:
                    raise RuntimeError(f"producer runtime prompt changed key={key}")
            return cache_key
        if len(self._pending) >= _COUNT:
            raise RuntimeError("producer observed a non-default runtime prompt")
        # Identity, never arrival ordinal, selects the source row.  Arrival may
        # be shuffled; the final manifest remains ordered by row_ordinal.
        source = resolve_source_identity(self._source_by_identity, identity)
        ordinal = int(source["row_ordinal"])
        durable = (self._assets / f"{ordinal:02d}-{cache_key[:24]}.wav").resolve()
        shutil.copyfile(canonical_wav, durable)
        os.chmod(durable, 0o444)
        source.update(observed)
        source["canonical_wav_path"] = str(durable)
        self._pending[cache_key] = source
        return cache_key

    def observe_prompt_wav(self, prompt_wav: Path) -> str:
        """Observe the file path that the live backend passes to prepare_prompt.

        The official text-only Seed-TTS chat workload does not carry
        ``codes.ref`` into Stage2.  Token2Wav therefore uses the model asset
        ``HT_ref_audio.wav`` through ``meta.prompt_wav``.  Decode it with the
        same soundfile float32, frames-last channel averaging contract used by
        the immutable source manifest, then reuse the normal producer/consumer
        identity checks.
        """
        prompt_wav = prompt_wav.expanduser().resolve(strict=True)
        waveform, sample_rate = sf.read(
            prompt_wav,
            dtype="float32",
            always_2d=False,
        )
        value = np.asarray(waveform, dtype=np.float32)
        if value.ndim > 1:
            value = value.mean(axis=-1, dtype=np.float32)
        value = np.ascontiguousarray(value.reshape(-1), dtype=np.float32)
        return self.observe_materialized(
            cache_key=prompt_identity_from_f32_bytes(
                value.tobytes(),
                sample_rate=int(sample_rate),
                samples=int(value.size),
            )["expected_cache_key"],
            waveform=torch.from_numpy(value),
            sample_rate=int(sample_rate),
            canonical_wav=prompt_wav,
        )

    def bind_request(self, request_id: str, cache_key: str) -> None:
        previous = self._request_keys.get(request_id)
        if previous is not None and previous != cache_key:
            raise RuntimeError("request changed its runtime prompt manifest identity")
        self._request_keys[request_id] = cache_key

    def observe_features(self, request_id: str, features: Any) -> None:
        cache_key = self._request_keys.get(request_id)
        if cache_key is None:
            return
        mels = features.mels.detach().cpu().contiguous()
        observed = {
            "prompt_mel_shape": list(mels.shape),
            "prompt_mel_sha256": tensor_sha256(mels),
            "prompt_frames": int(mels.shape[1]),
        }
        if self.mode == "consumer":
            expected = self._by_cache[cache_key]
            for key, actual in observed.items():
                if expected.get(key) != actual:
                    raise RuntimeError(
                        f"live prompt feature mismatch key={key} expected={expected.get(key)!r} actual={actual!r}"
                    )
            return
        entry = self._pending[cache_key]
        for key, actual in observed.items():
            previous = entry.get(key)
            if previous is not None and previous != actual:
                raise RuntimeError(f"producer prompt feature changed key={key}")
            entry[key] = actual
        if len(self._pending) != _COUNT or any("prompt_mel_sha256" not in row for row in self._pending.values()):
            return
        rows = [dict(row) for row in self._pending.values()]
        rows.sort(key=lambda row: int(row["row_ordinal"]))
        payload = {
            "schema": SCHEMA,
            "sealed": True,
            "source_manifest_path": str(self._source_path),
            "source_manifest_sha256": self._source_sha256,
            **self._source_contract,
            "rows": rows,
        }
        _atomic_json(self.path, payload)
        os.chmod(self.path, 0o444)
        self.payload = payload


__all__ = [
    "RuntimePromptManifest",
    "SCHEMA",
    "file_sha256",
    "load_sealed_manifest",
    "tensor_sha256",
]
