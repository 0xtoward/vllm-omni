# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-certified, immutable NPUGraph capture helpers.

This module deliberately implements a much narrower lifecycle than a graph
cache.  Graphs may only be discovered and captured during an explicit startup
warmup.  Once sealed, request-time misses run eagerly and are counted; they can
never mutate the resident graph set.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class GraphPhase(StrEnum):
    CENSUS = "census"
    WARMING = "warming"
    SEALED = "sealed"


@dataclass(frozen=True)
class TensorSignature:
    shape: tuple[int, ...]
    dtype: str
    device_type: str
    device_index: int | None
    layout: str
    stride: tuple[int, ...]
    storage_offset: int
    contiguous: bool
    npu_format: int | None


@dataclass(frozen=True)
class EstimatorSemantic:
    role: str
    request_batch: int
    prompt_frames: int
    codec_token_count: int
    last_chunk: bool
    flush_encoder: bool
    cache_present: bool
    attention_cache_length: int
    cfm_steps: int
    codec_chunk_frames: int
    left_context_frames: int


@dataclass
class _PerKeyStats:
    role: str
    captures: int = 0
    shadow_replays: int = 0
    replay_successes: int = 0


class _ReplayGraph(Protocol):
    def replay(self) -> None: ...


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    cloned = value.detach().clone(memory_format=torch.preserve_format)
    if value.device.type != "npu":
        return cloned

    import torch_npu

    getter = getattr(torch_npu, "get_npu_format", None)
    caster = getattr(torch_npu, "npu_format_cast", None)
    if not callable(getter) or not callable(caster):
        raise RuntimeError("torch_npu NPU-format clone helpers are unavailable")
    source_format = int(getter(value))
    cloned_format = int(getter(cloned))
    if cloned_format != source_format:
        cloned = caster(cloned, source_format)
    return cloned


def _npu_format(value: torch.Tensor) -> int | None:
    if value.device.type != "npu":
        return None
    try:
        import torch_npu

        getter = getattr(torch_npu, "get_npu_format", None)
        return int(getter(value)) if callable(getter) else None
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None


def tensor_signature(value: torch.Tensor) -> TensorSignature:
    device_index = value.device.index
    if device_index is None and value.device.type == "npu":
        current_device = getattr(getattr(torch, "npu", None), "current_device", None)
        if callable(current_device):
            device_index = int(current_device())
    return TensorSignature(
        shape=tuple(int(dim) for dim in value.shape),
        dtype=str(value.dtype),
        device_type=value.device.type,
        device_index=None if device_index is None else int(device_index),
        layout=str(value.layout),
        stride=tuple(int(item) for item in value.stride()),
        storage_offset=int(value.storage_offset()),
        contiguous=bool(value.is_contiguous()),
        npu_format=_npu_format(value),
    )


def _graph_input_signature(value: torch.Tensor) -> TensorSignature:
    """Signature seen by the graph-owned static input buffer.

    A live cache may be a contiguous view into a larger request-owned backing
    tensor.  Its storage offset is an address detail, not part of the graph
    program: replay copies the logical view into an offset-zero static buffer.
    All other descriptor fields remain exact.
    """
    return replace(tensor_signature(value), storage_offset=0)


def _key_hash(key: tuple[object, ...]) -> str:
    return sha256(repr(key).encode("utf-8")).hexdigest()[:16]


def _same_contract(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    return (
        tensor_signature(expected) == tensor_signature(actual)
        and torch.equal(torch.isfinite(expected), torch.isfinite(actual))
        and torch.equal(expected, actual)
    )


def _same_graph_input_contract(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    return (
        _graph_input_signature(expected) == _graph_input_signature(actual)
        and torch.equal(torch.isfinite(expected), torch.isfinite(actual))
        and torch.equal(expected, actual)
    )


def _storage_id(value: torch.Tensor) -> int:
    return int(value.untyped_storage().data_ptr())


@dataclass
class CapturedDeviceGraph:
    graph: _ReplayGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_outputs: tuple[torch.Tensor, ...]
    capture_footprint: dict[str, Any] = field(
        default_factory=lambda: {
            "capture_memory_available": False,
            "static_input_bytes": 0,
            "static_output_bytes": 0,
            "capture_before": None,
            "capture_after": None,
        }
    )

    def replay(
        self,
        inputs: tuple[torch.Tensor, ...],
        *,
        ephemeral_output_indices: frozenset[int] = frozenset(),
    ) -> tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.static_inputs):
            raise ValueError(f"graph expected {len(self.static_inputs)} inputs, got {len(inputs)}")
        with torch.inference_mode():
            for static, current in zip(self.static_inputs, inputs, strict=True):
                if _graph_input_signature(static) != _graph_input_signature(current):
                    raise RuntimeError("NPUGraph replay input signature changed")
                static.copy_(current)
            self.graph.replay()
            # Persistent outputs must be detached from graph-owned buffers.
            # A narrowly certified ephemeral output may be borrowed until the
            # next replay so its exact view/stride descriptor is preserved and
            # no extra device copy is inserted.  Its consumer must enqueue all
            # reads on the same stream before the next replay.
            return tuple(
                output if index in ephemeral_output_indices else _clone_tensor(output)
                for index, output in enumerate(self.static_outputs)
            )


class SealedNPUExactGraphRunner:
    """Capture a fixed startup allowlist and never mutate it at request time."""

    def __init__(
        self,
        *,
        mode: str,
        expected_roles: Iterable[str],
        component_name: str = "MiniCPM-o Stage2 estimator",
        contract_metadata: dict[str, Any] | None = None,
        ephemeral_output_indices: Iterable[int] = (),
    ) -> None:
        if mode not in {"runtime_only", "on"}:
            raise ValueError(f"sealed graph runner mode must be runtime_only/on, got {mode!r}")
        self.mode = mode
        self.component_name = component_name
        self.expected_roles = frozenset(str(role) for role in expected_roles)
        self.ephemeral_output_indices = frozenset(int(index) for index in ephemeral_output_indices)
        metadata = dict(contract_metadata or {})
        metadata["ephemeral_output_indices"] = sorted(self.ephemeral_output_indices)
        metadata_json = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._contract_metadata_json = metadata_json
        self.contract_fingerprint = sha256(metadata_json.encode("utf-8")).hexdigest()
        self.phase = GraphPhase.CENSUS if mode == "runtime_only" else GraphPhase.WARMING
        self._graphs: dict[tuple[object, ...], CapturedDeviceGraph] = {}
        self._role_keys: dict[str, tuple[object, ...]] = {}
        self._per_key: dict[tuple[object, ...], _PerKeyStats] = {}
        self._graph_pool: object | None = None
        self._fatal_error: RuntimeError | None = None

        self.capture_requests = 0
        self.captures = 0
        self.capture_failures = 0
        self.shadow_replay_successes = 0
        self.shadow_replay_failures = 0
        self.replay_successes = 0
        self.replay_failures = 0
        self.eager_warm_calls = 0
        self.runtime_only_eager_calls = 0
        self.tail_eager_calls = 0
        self.ineligible_eager_calls = 0
        self.sealed_misses = 0

    @staticmethod
    def is_supported() -> bool:
        npu = getattr(torch, "npu", None)
        return npu is not None and all(
            callable(getattr(npu, name, None))
            for name in (
                "NPUGraph",
                "graph",
                "graph_pool_handle",
                "is_current_stream_capturing",
                "synchronize",
            )
        )

    @staticmethod
    def _stream_is_capturing() -> bool:
        npu = getattr(torch, "npu", None)
        is_capturing = getattr(npu, "is_current_stream_capturing", None)
        if not callable(is_capturing):
            return False
        try:
            return bool(is_capturing())
        except (RuntimeError, TypeError):
            return False

    @classmethod
    def _eligible(cls, inputs: tuple[torch.Tensor, ...]) -> bool:
        return (
            bool(inputs)
            and cls.is_supported()
            and not cls._stream_is_capturing()
            and all(
                value.device.type == "npu" and value.layout == torch.strided and value.is_contiguous()
                for value in inputs
            )
        )

    @classmethod
    def _eligibility_diagnostics(cls, inputs: tuple[torch.Tensor, ...]) -> str:
        rows = []
        for index, value in enumerate(inputs):
            rows.append(
                {
                    "index": index,
                    "signature": repr(tensor_signature(value)),
                    "is_npu": value.device.type == "npu",
                    "is_strided": value.layout == torch.strided,
                    "contiguous": bool(value.is_contiguous()),
                    "storage_offset": int(value.storage_offset()),
                }
            )
        return json.dumps(
            {
                "has_inputs": bool(inputs),
                "graph_api_supported": cls.is_supported(),
                "stream_is_capturing": cls._stream_is_capturing(),
                "inputs": rows,
            },
            sort_keys=True,
        )

    def _raise_if_fatal(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self.component_name} graph process is poisoned and must restart"
            ) from self._fatal_error

    def _poison(self, message: str, exc: BaseException) -> RuntimeError:
        failure = RuntimeError(f"{self.component_name} {message}; restart required")
        failure.__cause__ = exc
        self._fatal_error = failure
        return failure

    @staticmethod
    def make_key(
        operation: str,
        semantic: EstimatorSemantic,
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[object, ...]:
        return (
            operation,
            semantic,
            tuple(_graph_input_signature(value) for value in inputs),
        )

    def _ensure_graph_pool(self) -> object:
        if self._graph_pool is None:
            self._graph_pool = torch.npu.graph_pool_handle()
        return self._graph_pool

    @staticmethod
    def _memory_snapshot() -> dict[str, int] | None:
        npu = getattr(torch, "npu", None)
        if npu is None:
            return None
        names = (
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
        )
        getters = {name: getattr(npu, name, None) for name in names}
        mem_get_info = getattr(npu, "mem_get_info", None)
        if not all(callable(value) for value in getters.values()) or not callable(mem_get_info):
            return None
        try:
            free, total = mem_get_info()
            return {
                **{name: int(getters[name]()) for name in names},
                "free": int(free),
                "total": int(total),
            }
        except (RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def _tensor_bytes(values: tuple[torch.Tensor, ...]) -> int:
        return sum(int(value.numel()) * int(value.element_size()) for value in values)

    def _capture(
        self,
        inputs: tuple[torch.Tensor, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
    ) -> CapturedDeviceGraph:
        npu = torch.npu
        npu.synchronize()
        before = self._memory_snapshot()
        static_inputs = tuple(_clone_tensor(value) for value in inputs)
        static_signatures = tuple(_graph_input_signature(value) for value in static_inputs)
        live_signatures = tuple(_graph_input_signature(value) for value in inputs)
        if static_signatures != live_signatures:
            differences = [
                f"input {index}: live={live!r} static={static!r}"
                for index, (live, static) in enumerate(zip(live_signatures, static_signatures, strict=True))
                if live != static
            ]
            raise RuntimeError("detached graph inputs changed the certified signature: " + "; ".join(differences))
        npu.synchronize()
        graph = npu.NPUGraph()
        with (
            torch.inference_mode(),
            npu.graph(
                graph,
                pool=self._ensure_graph_pool(),
            ),
        ):
            static_outputs = tuple(compute(*static_inputs))
        npu.synchronize()
        after = self._memory_snapshot()
        telemetry: dict[str, Any] = {
            "capture_memory_available": before is not None and after is not None,
            "static_input_bytes": self._tensor_bytes(static_inputs),
            "static_output_bytes": self._tensor_bytes(static_outputs),
            "capture_before": before,
            "capture_after": after,
        }
        if before is not None and after is not None:
            telemetry["capture_delta"] = {
                name: int(after[name]) - int(before[name]) for name in ("memory_allocated", "memory_reserved", "free")
            }
        return CapturedDeviceGraph(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
            capture_footprint=telemetry,
        )

    def _replay(
        self,
        key: tuple[object, ...],
        graph: CapturedDeviceGraph,
        inputs: tuple[torch.Tensor, ...],
        *,
        shadow: bool,
    ) -> tuple[torch.Tensor, ...]:
        try:
            outputs = graph.replay(
                inputs,
                ephemeral_output_indices=self.ephemeral_output_indices,
            )
        except Exception as exc:
            if shadow:
                self.shadow_replay_failures += 1
                raise self._poison("startup shadow replay failed", exc)
            self.replay_failures += 1
            raise self._poison("request replay failed", exc)
        stats = self._per_key[key]
        if shadow:
            self.shadow_replay_successes += 1
            stats.shadow_replays += 1
        else:
            self.replay_successes += 1
            stats.replay_successes += 1
        return outputs

    @staticmethod
    def _assert_shadow_contract(
        stock_outputs: tuple[torch.Tensor, ...],
        shadow_outputs: tuple[torch.Tensor, ...],
        captured: CapturedDeviceGraph,
        inputs: tuple[torch.Tensor, ...],
        input_snapshots: tuple[torch.Tensor, ...],
        ephemeral_output_indices: frozenset[int],
    ) -> None:
        if len(stock_outputs) != len(shadow_outputs):
            raise RuntimeError("shadow output arity differs from stock")
        mismatches: list[str] = []
        for index, (stock, shadow) in enumerate(zip(stock_outputs, shadow_outputs, strict=True)):
            if not _same_contract(stock, shadow):
                stock_f32 = stock.detach().float()
                shadow_f32 = shadow.detach().float()
                difference = (stock_f32 - shadow_f32).abs()
                max_abs = float(difference.max().item())
                mean_abs = float(difference.mean().item())
                rms_error = float(difference.square().mean().sqrt().item())
                stock_rms = float(stock_f32.square().mean().sqrt().item())
                nrmse = rms_error / max(stock_rms, 1e-30)
                exact_fraction = float((stock == shadow).float().mean().item())
                mismatches.append(
                    f"shadow output {index} is not bit-exact with stock "
                    f"shape={tuple(stock.shape)}/{tuple(shadow.shape)} "
                    f"dtype={stock.dtype}/{shadow.dtype} "
                    f"max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} "
                    f"nrmse={nrmse:.9g} exact_fraction={exact_fraction:.9g} "
                    f"stock_signature={tensor_signature(stock)!r} "
                    f"shadow_signature={tensor_signature(shadow)!r} "
                    f"static_signature={tensor_signature(captured.static_outputs[index])!r}"
                )
            shadow_storage = _storage_id(shadow)
            if shadow_storage in {_storage_id(value) for value in inputs}:
                raise RuntimeError(f"shadow output {index} aliases a live input")
            if (
                shadow_storage in {_storage_id(value) for value in captured.static_outputs}
                and index not in ephemeral_output_indices
            ):
                raise RuntimeError(f"shadow output {index} aliases persistent graph output")
        if mismatches:
            raise RuntimeError("; ".join(mismatches))
        for index, (before, after) in enumerate(zip(input_snapshots, inputs, strict=True)):
            if not _same_graph_input_contract(before, after):
                raise RuntimeError(f"estimator input/cache {index} was modified in place")

    def _warm_capture(
        self,
        key: tuple[object, ...],
        semantic: EstimatorSemantic,
        inputs: tuple[torch.Tensor, ...],
        stock_compute: Callable[[], tuple[torch.Tensor, ...]],
        graph_compute: Callable[..., tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        existing = self._role_keys.get(semantic.role)
        if existing is not None and existing != key:
            raise RuntimeError(f"startup produced a second signature for role {semantic.role!r}")
        if semantic.role not in self.expected_roles:
            raise RuntimeError(f"startup produced unexpected graph role {semantic.role!r}")
        self.capture_requests += 1
        input_snapshots = tuple(_clone_tensor(value) for value in inputs)
        stock_outputs = tuple(stock_compute())
        self.eager_warm_calls += 1
        try:
            captured = self._capture(inputs, graph_compute)
        except Exception as exc:
            self.capture_failures += 1
            raise self._poison("startup capture failed", exc)

        # Publish counters/key metadata before the shadow call, but not the
        # graph itself.  A mismatch must leave resident_graphs == 0 for this key.
        self._per_key[key] = _PerKeyStats(role=semantic.role)
        shadow_outputs = self._replay(key, captured, inputs, shadow=True)
        try:
            self._assert_shadow_contract(
                stock_outputs,
                shadow_outputs,
                captured,
                inputs,
                input_snapshots,
                self.ephemeral_output_indices,
            )
        except Exception as exc:
            self.shadow_replay_failures += 1
            self.shadow_replay_successes -= 1
            self._per_key[key].shadow_replays -= 1
            self._per_key.pop(key, None)
            raise self._poison("startup shadow parity failed", exc)
        self._graphs[key] = captured
        self._role_keys[semantic.role] = key
        self.captures += 1
        self._per_key[key].captures += 1
        # Startup state must always advance from the independent stock path.
        # The captured graph is only a candidate until the whole allowlist is
        # shadow-certified and sealed.
        return stock_outputs

    def _warm_shadow_existing(
        self,
        key: tuple[object, ...],
        graph: CapturedDeviceGraph,
        inputs: tuple[torch.Tensor, ...],
        stock_compute: Callable[[], tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        """Shadow an already-captured startup signature without consuming it.

        CFM invokes one estimator signature several times with different tensor
        values.  During startup all of those calls remain stock-owned: replay is
        validation-only so a candidate can never seed the next cache state.
        """
        input_snapshots = tuple(_clone_tensor(value) for value in inputs)
        stock_outputs = tuple(stock_compute())
        self.eager_warm_calls += 1
        shadow_outputs = self._replay(key, graph, inputs, shadow=True)
        try:
            self._assert_shadow_contract(
                stock_outputs,
                shadow_outputs,
                graph,
                inputs,
                input_snapshots,
                self.ephemeral_output_indices,
            )
        except Exception as exc:
            self.shadow_replay_failures += 1
            self.shadow_replay_successes -= 1
            self._per_key[key].shadow_replays -= 1
            raise self._poison("startup shadow parity failed", exc)
        return stock_outputs

    def run(
        self,
        operation: str,
        semantic: EstimatorSemantic,
        inputs: tuple[torch.Tensor, ...],
        *,
        stock_compute: Callable[[], tuple[torch.Tensor, ...]],
        graph_compute: Callable[..., tuple[torch.Tensor, ...]] | None,
    ) -> tuple[torch.Tensor, ...]:
        self._raise_if_fatal()
        if semantic.role == "tail_b1":
            self.tail_eager_calls += 1
            return tuple(stock_compute())
        if semantic.role.startswith("ineligible_"):
            self.ineligible_eager_calls += 1
            return tuple(stock_compute())
        if self.mode == "runtime_only":
            self.runtime_only_eager_calls += 1
            key = self.make_key(operation, semantic, inputs)
            self._per_key.setdefault(key, _PerKeyStats(role=semantic.role))
            return tuple(stock_compute())
        if not self._eligible(inputs):
            if self.phase == GraphPhase.WARMING:
                raise RuntimeError(
                    f"startup role {semantic.role!r} is not eligible for NPUGraph capture: "
                    f"{self._eligibility_diagnostics(inputs)}"
                )
            self.ineligible_eager_calls += 1
            return tuple(stock_compute())

        key = self.make_key(operation, semantic, inputs)
        graph = self._graphs.get(key)
        if graph is not None:
            if self.phase == GraphPhase.WARMING:
                return self._warm_shadow_existing(
                    key,
                    graph,
                    inputs,
                    stock_compute,
                )
            return self._replay(key, graph, inputs, shadow=False)
        if self.phase == GraphPhase.WARMING:
            if graph_compute is None:
                raise RuntimeError(f"startup role {semantic.role!r} has no graph candidate")
            return self._warm_capture(
                key,
                semantic,
                inputs,
                stock_compute,
                graph_compute,
            )
        if self.phase != GraphPhase.SEALED:
            raise RuntimeError(f"invalid graph phase {self.phase}")
        self.sealed_misses += 1
        logger.error(
            "%s sealed miss key=%s role=%s; using eager without capture",
            self.component_name,
            _key_hash(key),
            semantic.role,
        )
        return tuple(stock_compute())

    def record_external_replay(
        self,
        operation: str,
        semantic: EstimatorSemantic,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        """Account for a sealed replay owned by a model-local graph program.

        Model-local resident-state graphs deliberately bypass ``run`` because
        their persistent inputs/outputs do not follow the generic copy/clone
        policy. They still share this runner's immutable signature and per-role
        lifecycle contract, so their successful request replay belongs in the
        same service ledger.
        """
        self._raise_if_fatal()
        if self.phase != GraphPhase.SEALED:
            raise RuntimeError("external replay can only be recorded after seal")
        key = self.make_key(operation, semantic, inputs)
        stats = self._per_key.get(key)
        if stats is None:
            raise RuntimeError("external replay signature was not sealed at startup")
        self.replay_successes += 1
        stats.replay_successes += 1

    def seal(self) -> None:
        self._raise_if_fatal()
        if self.phase != GraphPhase.WARMING:
            raise RuntimeError(f"only a warming runner can seal, got {self.phase}")
        observed = frozenset(self._role_keys)
        if observed != self.expected_roles:
            raise RuntimeError(f"cannot seal roles expected={sorted(self.expected_roles)} observed={sorted(observed)}")
        if len(self._graphs) != len(self.expected_roles):
            raise RuntimeError("resident graph count does not match certified roles")
        self.phase = GraphPhase.SEALED

    def snapshot(
        self,
        event: str,
        request_ordinal: int | None = None,
    ) -> dict[str, Any]:
        per_key = []
        for key, stats in sorted(self._per_key.items(), key=lambda pair: _key_hash(pair[0])):
            per_key.append({"hash": _key_hash(key), **asdict(stats)})
        resident_contract = [
            {
                "role": row["role"],
                "hash": row["hash"],
                "captures": row["captures"],
                "shadow_replays": row["shadow_replays"],
            }
            for row in per_key
            if row["captures"] > 0
        ]
        resident_fingerprint = sha256(
            json.dumps(
                resident_contract,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        capture_footprint_rows = []
        for key, graph in sorted(self._graphs.items(), key=lambda pair: _key_hash(pair[0])):
            stats = self._per_key[key]
            capture_footprint_rows.append(
                {
                    "hash": _key_hash(key),
                    "role": stats.role,
                    **graph.capture_footprint,
                }
            )
        capture_time_graph_footprint = {
            "resident_graphs": len(capture_footprint_rows),
            "capture_telemetry_complete": bool(capture_footprint_rows)
            and all(bool(row["capture_memory_available"]) for row in capture_footprint_rows),
            "total_static_input_bytes": sum(int(row["static_input_bytes"]) for row in capture_footprint_rows),
            "total_static_output_bytes": sum(int(row["static_output_bytes"]) for row in capture_footprint_rows),
            "per_graph": capture_footprint_rows,
        }
        return {
            "event": event,
            "request_ordinal": request_ordinal,
            "phase": self.phase.value,
            "capture_requests": self.capture_requests,
            "captures": self.captures,
            "capture_failures": self.capture_failures,
            "shadow_replay_successes": self.shadow_replay_successes,
            "shadow_replay_failures": self.shadow_replay_failures,
            "replay_successes": self.replay_successes,
            "replay_failures": self.replay_failures,
            "eager_warm_calls": self.eager_warm_calls,
            "runtime_only_eager_calls": self.runtime_only_eager_calls,
            "tail_eager_calls": self.tail_eager_calls,
            "ineligible_eager_calls": self.ineligible_eager_calls,
            "sealed_misses": self.sealed_misses,
            "resident_graphs": len(self._graphs),
            "expected_roles": sorted(self.expected_roles),
            "contract": json.loads(self._contract_metadata_json),
            "contract_fingerprint": self.contract_fingerprint,
            "resident_fingerprint": resident_fingerprint,
            # These are immutable capture-time measurements and static tensor
            # sizes.  They do not claim that the live allocator was sampled at
            # request boundaries.
            "capture_time_graph_footprint": capture_time_graph_footprint,
            "per_key": per_key,
        }

    def log_snapshot(self, event: str, request_ordinal: int | None = None) -> None:
        logger.info(
            "MiniCPMO45Stage2EstimatorGraph %s",
            json.dumps(self.snapshot(event, request_ordinal), sort_keys=True),
        )


__all__ = [
    "CapturedDeviceGraph",
    "EstimatorSemantic",
    "GraphPhase",
    "SealedNPUExactGraphRunner",
    "TensorSignature",
    "tensor_signature",
]
