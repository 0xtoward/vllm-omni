# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instance-local sealed NPUGraph support for MiniCPM-o 4.5 Stage2.

The certified CFM3 solve boundary captures three DiT estimator calls together
with their CFG and Euler updates. Encoder, HiFT, request state, and final/tail
chunks remain outside the graph.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.minicpmo_4_5.runtime_prompt_manifest import (
    load_sealed_manifest,
    tensor_sha256,
)
from vllm_omni.platforms.npu.graph_tools import (
    EstimatorSemantic,
    SealedNPUExactGraphRunner,
)

logger = init_logger(__name__)

_MODE_KEY = "code2wav_npu_graph_mode"
_PROFILE_KEY = "code2wav_npu_graph_profile"
_PROMPT_MANIFEST_KEY = "code2wav_npu_graph_prompt_manifest"
_PROMPT_MANIFEST_SHA_KEY = "code2wav_npu_graph_prompt_manifest_sha256"
_PROMPT_MANIFEST_MODE_KEY = "code2wav_npu_graph_prompt_manifest_mode"
_VALID_MODES = frozenset({"off", "runtime_only", "on"})
_CERTIFIED_PROFILE = "cfm3_ccf25_b1_model_default_prompt_v1"
_CERTIFIED_CFM_STEPS = 3
_CERTIFIED_CODEC_CHUNK_FRAMES = 25
_CERTIFIED_LEFT_CONTEXT_FRAMES = 3
_CERTIFIED_CODEC_INPUT_WIDTH = 28
_CERTIFIED_ESTIMATOR_WIDTH = 50
_CERTIFIED_PROMPT_WAV_COUNT = 1
_CERTIFIED_STEADY_CACHE_OFFSETS = (0, 50, 100)
_BOOTSTRAP_ATTR = "_minicpmo45_stage2_estimator_graph_bootstrap"
_SILENCE_TOKEN = 4218


@dataclass(frozen=True)
class _PromptWavSpec:
    path: str
    sha256: str
    manifest_row: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PromptCensusRow:
    path: str
    sha256: str
    prompt_frames: int
    manifest_row: dict[str, Any] = field(default_factory=dict)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt_role(prompt_frames: int) -> str:
    return f"prompt_b1_pf{prompt_frames}"


def _steady_role_for_lengths(prompt_frames: int, old_attention_length: int) -> str:
    if old_attention_length - prompt_frames not in _CERTIFIED_STEADY_CACHE_OFFSETS:
        return f"ineligible_cache_pf{prompt_frames}_ac{old_attention_length}"
    return f"steady_b1_pf{prompt_frames}_ac{old_attention_length}"


def _roles_for_prompt_frames(prompt_frames: int) -> frozenset[str]:
    if prompt_frames <= 0:
        raise ValueError(f"prompt_frames must be positive, got {prompt_frames}")
    return frozenset(
        {
            _prompt_role(prompt_frames),
            *(
                _steady_role_for_lengths(prompt_frames, prompt_frames + offset)
                for offset in _CERTIFIED_STEADY_CACHE_OFFSETS
            ),
        }
    )


def _parse_prompt_wav_specs(extra: Mapping[str, Any]) -> tuple[_PromptWavSpec, ...]:
    if str(extra.get(_PROMPT_MANIFEST_MODE_KEY, "")).lower() != "consumer":
        raise RuntimeError("MiniCPM-o Stage2 graph requires prompt manifest mode=consumer")
    path_raw = str(extra.get(_PROMPT_MANIFEST_KEY, "")).strip()
    expected_sha = str(extra.get(_PROMPT_MANIFEST_SHA_KEY, "")).strip().lower()
    if not path_raw or not Path(path_raw).is_absolute() or len(expected_sha) != 64:
        raise RuntimeError("MiniCPM-o Stage2 graph prompt manifest path/SHA is invalid")
    payload = load_sealed_manifest(Path(path_raw), expected_sha)
    raw = payload["rows"]
    result: list[_PromptWavSpec] = []
    seen_paths: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"MiniCPM-o prompt manifest row {index} must be a mapping")
        source = Path(str(row.get("canonical_wav_path", ""))).expanduser()
        if not source.is_absolute():
            raise RuntimeError(f"MiniCPM-o prompt manifest row {index} path must be absolute")
        path = source.resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".wav":
            raise RuntimeError(f"MiniCPM-o prompt manifest row {index} is not a regular WAV: {path}")
        expected_digest = str(row.get("canonical_wav_sha256", "")).lower()
        if len(expected_digest) != 64 or any(char not in "0123456789abcdef" for char in expected_digest):
            raise RuntimeError(f"MiniCPM-o prompt manifest row {index} SHA256 is invalid")
        actual_digest = _file_sha256(path)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"MiniCPM-o canonical runtime prompt WAV digest mismatch path={path} "
                f"expected={expected_digest} actual={actual_digest}"
            )
        normalized_path = str(path)
        if normalized_path in seen_paths:
            raise RuntimeError(f"MiniCPM-o prompt manifest contains duplicate path {normalized_path}")
        seen_paths.add(normalized_path)
        result.append(
            _PromptWavSpec(
                path=normalized_path,
                sha256=actual_digest,
                manifest_row=dict(row),
            )
        )
    return tuple(result)


def _mode_from_extra(extra: Mapping[str, Any]) -> str:
    raw = extra.get(_MODE_KEY, "off")
    if raw is True:
        return "on"
    if raw is False or raw is None:
        return "off"
    mode = str(raw).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"MiniCPM-o {_MODE_KEY} must be one of {sorted(_VALID_MODES)}, got {raw!r}")
    return mode


def _configure_npu_graph_runtime() -> None:
    if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        raise RuntimeError("MiniCPM-o Stage2 NPUGraph is incompatible with ASCEND_LAUNCH_BLOCKING=1")
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError("torch.npu is unavailable for Stage2 estimator graph mode")
    config = getattr(npu, "config", None)
    if config is None:
        raise RuntimeError("torch.npu.config is unavailable")
    # torch_npu exposes allow_internal_format as a write-only configuration
    # property in some releases: assigning it is supported, while getattr()
    # and hasattr() deliberately report it as absent.  Test the operation, not
    # readability, so standalone producer/oracle processes use the same valid
    # runtime contract as service workers.
    try:
        config.allow_internal_format = False
    except Exception as exc:
        raise RuntimeError("failed to disable torch.npu.config.allow_internal_format") from exc
    set_compile_mode = getattr(npu, "set_compile_mode", None)
    if not callable(set_compile_mode):
        raise RuntimeError("torch.npu.set_compile_mode is unavailable")
    set_compile_mode(jit_compile=False)

    from vllm_omni.platforms.npu.models.cosyvoice2_dit_attn import (
        apply_cosyvoice2_dit_attn_npu_patch,
    )

    apply_cosyvoice2_dit_attn_npu_patch()
    logger.info(
        "Configured MiniCPM-o Stage2 graph runtime "
        "(allow_internal_format=False jit_compile=False MATH-SDPA patch=installed)"
    )


def _flow_context(enabled: bool):
    if not enabled:
        return nullcontext()
    from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
        npu_token2wav_sdpa_context,
    )

    return npu_token2wav_sdpa_context(require_math=True)


def _graphable_estimator_step(
    backend: Any,
    estimator: Any,
    *,
    x: torch.Tensor,
    mu: torch.Tensor,
    time_embedding: torch.Tensor,
    speakers: torch.Tensor,
    cond: torch.Tensor,
    cnn_cache: torch.Tensor | None,
    att_cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = int(x.shape[-1])
    speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
    estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
    cnn_out, att_out = backend._estimator_buffers(
        estimator,
        estimator_input,
        att_cache,
    )
    old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
    old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
    result = estimator.blocks_forward_chunk(
        estimator_input,
        time_embedding,
        None,
        old_cnn,
        old_att,
        cnn_out,
        att_out,
    )
    return result, cnn_out, att_out


def _graphable_fixed_cfm3_solve(
    backend: Any,
    estimator: Any,
    decoder: Any,
    *,
    x: torch.Tensor,
    mu_cfg: torch.Tensor,
    time_embeddings: torch.Tensor,
    speakers_cfg: torch.Tensor,
    cond_cfg: torch.Tensor,
    dts: torch.Tensor,
    cnn_cache: torch.Tensor | None,
    att_cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one fixed CFM3 solve as a single graph-owned tensor program."""
    if int(time_embeddings.shape[0]) != _CERTIFIED_CFM_STEPS:
        raise RuntimeError("fixed CFM3 graph received the wrong time embedding count")
    if int(dts.shape[0]) != _CERTIFIED_CFM_STEPS:
        raise RuntimeError("fixed CFM3 graph received the wrong timestep count")

    batch_size = int(x.shape[0])
    next_cnn: list[torch.Tensor] = []
    next_att: list[torch.Tensor] = []
    for step in range(_CERTIFIED_CFM_STEPS):
        old_cnn = cnn_cache[step] if cnn_cache is not None else None
        old_att = att_cache[step] if att_cache is not None else None
        estimate, step_cnn, step_att = _graphable_estimator_step(
            backend,
            estimator,
            x=torch.cat((x, x), dim=0),
            mu=mu_cfg,
            time_embedding=time_embeddings[step],
            speakers=speakers_cfg,
            cond=cond_cfg,
            cnn_cache=old_cnn,
            att_cache=old_att,
        )
        conditional, unconditional = estimate.split(batch_size, dim=0)
        velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
        x = x + dts[step] * velocity
        next_cnn.append(step_cnn)
        next_att.append(step_att)
    return x, torch.stack(next_cnn), torch.stack(next_att)


def _compact_estimator_att_cache(
    value: torch.Tensor,
    *,
    prompt_frames: int,
) -> torch.Tensor:
    """Match ``BatchedToken2Wav.decode_batch``'s steady cache policy."""
    limit = prompt_frames + 100
    if int(value.shape[4]) <= limit:
        return value
    return torch.cat(
        (value[..., :prompt_frames, :], value[..., -100:, :]),
        dim=4,
    )


def _assert_resident_exact(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    same_signature = (
        tuple(expected.shape) == tuple(actual.shape)
        and expected.dtype == actual.dtype
        and expected.device == actual.device
    )
    if same_signature and torch.equal(expected, actual):
        return
    if not same_signature:
        raise RuntimeError(
            "resident CFM3 startup shadow signature mismatch "
            f"name={name} expected_shape={tuple(expected.shape)} "
            f"actual_shape={tuple(actual.shape)} expected_dtype={expected.dtype} "
            f"actual_dtype={actual.dtype} expected_device={expected.device} "
            f"actual_device={actual.device}"
        )
    expected_f32 = expected.detach().float()
    actual_f32 = actual.detach().float()
    difference = (expected_f32 - actual_f32).abs()
    raise RuntimeError(
        "resident CFM3 startup shadow mismatch "
        f"name={name} expected_shape={tuple(expected.shape)} "
        f"actual_shape={tuple(actual.shape)} expected_dtype={expected.dtype} "
        f"actual_dtype={actual.dtype} max_abs={float(difference.max().item()):.9g}"
    )


@dataclass
class _ResidentGraphProgram:
    graph: Any
    pool: Any
    static_small_inputs: tuple[torch.Tensor, ...]
    mel_output: torch.Tensor
    full_att_output: torch.Tensor
    source_cnn: torch.Tensor | None
    source_att: torch.Tensor | None
    target_cnn: torch.Tensor
    target_att: torch.Tensor

    def replay(
        self,
        small_inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(small_inputs) != len(self.static_small_inputs):
            raise RuntimeError("resident CFM3 small-input arity changed")
        with torch.inference_mode():
            for static, current in zip(
                self.static_small_inputs,
                small_inputs,
                strict=True,
            ):
                if (
                    tuple(static.shape) != tuple(current.shape)
                    or static.dtype != current.dtype
                    or static.device != current.device
                    or tuple(static.stride()) != tuple(current.stride())
                ):
                    raise RuntimeError("resident CFM3 small-input signature changed")
                static.copy_(current)
            self.graph.replay()
        # Preserve _decode_cfm's public contract: it returns the untrimmed
        # per-step attention cache. BatchedToken2Wav trims that cache one level
        # above this call. The graph-owned 402-frame target_att is consumed only
        # by the split-cache hook after that public operation has completed.
        return self.mel_output, self.target_cnn, self.full_att_output


class _ResidentCFM3GraphPair:
    """A resident prompt/ramp/steady CFM3 cache-state pipeline.

    The historical implementation made only the final 402-frame steady cache
    resident.  Prompt and ramp roles still copied and cloned 100+ MiB caches
    through the generic graph runner.  This pipeline seals the complete B=1
    state progression instead::

        no cache -> 302 -> 352 -> 402A -> 402B -> 402A -> ...

    Request-dependent solve tensors stay small and are copied into each graph.
    Estimator CNN/attention caches remain in graph-owned buffers throughout the
    request.  The public ``_decode_cfm`` contract still exposes the full
    attention output; the cache hook maps it to the resident compact buffer
    before request state is materialized.
    """

    def __init__(
        self,
        backend: Any,
        estimator: Any,
        decoder: Any,
        *,
        prompt_frames: int,
    ) -> None:
        self.backend = backend
        self.estimator = estimator
        self.decoder = decoder
        self.prompt_frames = int(prompt_frames)
        self.expected_roles = (
            _prompt_role(self.prompt_frames),
            _steady_role_for_lengths(self.prompt_frames, self.prompt_frames),
            _steady_role_for_lengths(self.prompt_frames, self.prompt_frames + 50),
            _steady_role_for_lengths(self.prompt_frames, self.prompt_frames + 100),
        )
        self.programs: dict[str, list[_ResidentGraphProgram]] = {}
        self._resident_outputs: dict[int, tuple[torch.Tensor, set[int]]] = {}
        self.captures = 0
        self.shadow_checks = 0
        self.replays = 0
        self.external_loads = 0

    @staticmethod
    def _storage_id(value: torch.Tensor) -> int:
        return int(value.untyped_storage().data_ptr())

    @property
    def ready(self) -> bool:
        return tuple(self.programs) == self.expected_roles and len(self.programs[self.expected_roles[-1]]) == 2

    @staticmethod
    def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
        return _ResidentCFM3GraphPair._storage_id(left) == (_ResidentCFM3GraphPair._storage_id(right))

    def _remember_output(
        self,
        program: _ResidentGraphProgram,
        *,
        full_attention_width: int,
    ) -> None:
        cnn_id = self._storage_id(program.target_cnn)
        existing = self._resident_outputs.get(cnn_id)
        if existing is None:
            widths: set[int] = set()
            self._resident_outputs[cnn_id] = (program.target_att, widths)
        else:
            resident_att, widths = existing
            if not self._same_storage(resident_att, program.target_att):
                raise RuntimeError("resident CFM3 CNN cache alias is ambiguous")
        self._resident_outputs[cnn_id][1].add(int(full_attention_width))

    def owns(
        self,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
    ) -> bool:
        output = self._resident_outputs.get(self._storage_id(cnn_cache))
        return output is not None and self._same_storage(output[0], att_cache)

    def resolve_compacted_output(
        self,
        cnn_cache: torch.Tensor,
        compacted_att_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Map a stock-compacted public output back to its resident cache pair."""
        output = self._resident_outputs.get(self._storage_id(cnn_cache))
        if output is None:
            return None
        resident_att, _ = output
        if (
            tuple(compacted_att_cache.shape) != tuple(resident_att.shape)
            or compacted_att_cache.dtype != resident_att.dtype
            or compacted_att_cache.device != resident_att.device
        ):
            raise RuntimeError("resident CFM3 compacted cache signature changed")
        return cnn_cache, resident_att

    def resolve_precompacted_output(
        self,
        cnn_cache: torch.Tensor,
        full_att_cache: torch.Tensor,
        prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the graph-owned compact cache before stock repeats the cat."""
        output = self._resident_outputs.get(self._storage_id(cnn_cache))
        if output is None:
            return None
        resident_att, allowed_full_widths = output
        if (
            int(prompt_len) != self.prompt_frames
            or int(full_att_cache.shape[4]) not in allowed_full_widths
            or tuple(full_att_cache.shape[:4]) != tuple(resident_att.shape[:4])
            or int(full_att_cache.shape[5]) != int(resident_att.shape[5])
            or full_att_cache.dtype != resident_att.dtype
            or full_att_cache.device != resident_att.device
        ):
            raise RuntimeError("resident CFM3 public cache signature changed")
        return cnn_cache, resident_att

    def _compute(
        self,
        small_inputs: tuple[torch.Tensor, ...],
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, mu_cfg, time_embeddings, speakers_cfg, cond_cfg, dts = small_inputs
        return _graphable_fixed_cfm3_solve(
            self.backend,
            self.estimator,
            self.decoder,
            x=x,
            mu_cfg=mu_cfg,
            time_embeddings=time_embeddings,
            speakers_cfg=speakers_cfg,
            cond_cfg=cond_cfg,
            dts=dts,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )

    def _capture(
        self,
        small_inputs: tuple[torch.Tensor, ...],
        *,
        source_cnn: torch.Tensor | None,
        source_att: torch.Tensor | None,
        target_cnn: torch.Tensor,
        target_att: torch.Tensor,
    ) -> _ResidentGraphProgram:
        npu = torch.npu
        static_small_inputs = tuple(value.detach().clone(memory_format=torch.preserve_format) for value in small_inputs)
        npu.synchronize()
        graph = npu.NPUGraph()
        pool = npu.graph_pool_handle()
        with torch.inference_mode(), npu.graph(graph, pool=pool):
            mel_output, full_cnn, full_att = self._compute(
                static_small_inputs,
                source_cnn,
                source_att,
            )
            target_cnn.copy_(full_cnn)
            target_att.copy_(
                _compact_estimator_att_cache(
                    full_att,
                    prompt_frames=self.prompt_frames,
                )
            )
        npu.synchronize()
        self.captures += 1
        return _ResidentGraphProgram(
            graph=graph,
            pool=pool,
            static_small_inputs=static_small_inputs,
            mel_output=mel_output,
            full_att_output=full_att,
            source_cnn=source_cnn,
            source_att=source_att,
            target_cnn=target_cnn,
            target_att=target_att,
        )

    def prepare_transition(
        self,
        role: str,
        small_inputs: tuple[torch.Tensor, ...],
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        stock_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        role_index = sum(len(values) for values in self.programs.values())
        if role_index >= len(self.expected_roles):
            raise RuntimeError("resident CFM3 state pipeline was prepared twice")
        expected_role = self.expected_roles[role_index]
        if role != expected_role:
            raise RuntimeError(f"resident CFM3 warmup role order changed expected={expected_role} actual={role}")
        expected_mel, expected_cnn, expected_full_att = stock_outputs
        expected_att = _compact_estimator_att_cache(
            expected_full_att,
            prompt_frames=self.prompt_frames,
        )
        target_cnn = torch.empty_like(expected_cnn)
        target_att = torch.empty_like(expected_att)
        program = self._capture(
            small_inputs,
            source_cnn=cnn_cache,
            source_att=att_cache,
            target_cnn=target_cnn,
            target_att=target_att,
        )
        actual_mel, actual_cnn, actual_full_att = program.replay(small_inputs)
        torch.npu.synchronize()
        _assert_resident_exact(f"{role}.mel", expected_mel, actual_mel)
        _assert_resident_exact(f"{role}.cnn", expected_cnn, actual_cnn)
        _assert_resident_exact(f"{role}.full_att", expected_full_att, actual_full_att)
        _assert_resident_exact(f"{role}.resident_att", expected_att, target_att)
        self.shadow_checks += 1
        self.programs[role] = [program]
        self._remember_output(
            program,
            full_attention_width=int(expected_full_att.shape[4]),
        )

        # The fourth transition is the first 402 -> 402 steady solve.  Capture
        # the reverse edge into its resident source so future requests can
        # alternate without copying either cache.
        if role == self.expected_roles[-1]:
            if cnn_cache is None or att_cache is None:
                raise RuntimeError("steady resident transition lost its source cache")
            expected2_mel, expected2_cnn, expected2_full_att = self._compute(
                small_inputs,
                target_cnn.detach().clone(),
                target_att.detach().clone(),
            )
            expected2_att = _compact_estimator_att_cache(
                expected2_full_att,
                prompt_frames=self.prompt_frames,
            )
            reverse = self._capture(
                small_inputs,
                source_cnn=target_cnn,
                source_att=target_att,
                target_cnn=cnn_cache,
                target_att=att_cache,
            )
            actual2_mel, actual2_cnn, actual2_full_att = reverse.replay(small_inputs)
            torch.npu.synchronize()
            _assert_resident_exact("steady_reverse.mel", expected2_mel, actual2_mel)
            _assert_resident_exact("steady_reverse.cnn", expected2_cnn, actual2_cnn)
            _assert_resident_exact("steady_reverse.full_att", expected2_full_att, actual2_full_att)
            _assert_resident_exact("steady_reverse.resident_att", expected2_att, att_cache)
            self.shadow_checks += 1
            self.programs[role].append(reverse)
            self._remember_output(
                reverse,
                full_attention_width=int(expected2_full_att.shape[4]),
            )

        if self.ready:
            logger.info(
                "MiniCPM-o Stage2 resident CFM3 state pipeline sealed "
                "prompt_frames=%d roles=%s captures=%d shadow_checks=%d",
                self.prompt_frames,
                list(self.programs),
                self.captures,
                self.shadow_checks,
            )
        return actual_mel, target_cnn, actual_full_att

    def replay(
        self,
        role: str,
        small_inputs: tuple[torch.Tensor, ...],
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.ready:
            raise RuntimeError("resident CFM3 graph pair is not sealed")
        candidates = self.programs.get(role, [])
        program = None
        for candidate in candidates:
            if candidate.source_cnn is None:
                if cnn_cache is None and att_cache is None:
                    program = candidate
                    break
            elif (
                cnn_cache is not None
                and att_cache is not None
                and self._same_storage(candidate.source_cnn, cnn_cache)
                and candidate.source_att is not None
                and self._same_storage(candidate.source_att, att_cache)
            ):
                program = candidate
                break
        if program is None:
            self.external_loads += 1
            return None
        outputs = program.replay(small_inputs)
        self.replays += 1
        return outputs

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "ready": self.ready,
            "captures": self.captures,
            "shadow_checks": self.shadow_checks,
            "replays": self.replays,
            "external_loads": self.external_loads,
            "roles": len(self.programs),
        }


class Stage2EstimatorGraphBootstrap:
    """Own the graph lifecycle for one MiniCPM Stage2 backend instance."""

    def __init__(self, model: Any, *, mode: str, profile: str) -> None:
        self.model = model
        self.mode = mode
        self.profile = profile
        self.backend: Any | None = None
        self.runner: SealedNPUExactGraphRunner | None = None
        self._semantic: dict[str, Any] | None = None
        self._installed = False
        self._internal_warmup = False
        self._request_ordinal = 0
        self._cfm_steps = -1
        self._codec_chunk_frames = -1
        self._left_context_frames = -1
        self._prompt_specs: tuple[_PromptWavSpec, ...] = ()
        self._prompt_census: tuple[_PromptCensusRow, ...] = ()
        self._allowed_prompt_frames: frozenset[int] = frozenset()
        self._expected_roles: frozenset[str] = frozenset()
        self._resident_graph: _ResidentCFM3GraphPair | None = None
        self._original_decode_cfm = None
        self._original_setup_batch = None
        self._original_decode_batch = None
        self._original_stack_flow_cache = None
        self._original_split_flow_cache = None

    @classmethod
    def from_model(cls, model: Any) -> Stage2EstimatorGraphBootstrap:
        extra_fn = getattr(model, "_extra_config", None)
        extra = extra_fn() if callable(extra_fn) else {}
        if not isinstance(extra, Mapping):
            extra = {}
        return cls(
            model,
            mode=_mode_from_extra(extra),
            profile=str(extra.get(_PROFILE_KEY, _CERTIFIED_PROFILE)),
        )

    def prepare_runtime_before_token2wav_load(self) -> None:
        if self.mode != "off":
            _configure_npu_graph_runtime()

    def _extra(self) -> Mapping[str, Any]:
        extra = self.model._extra_config()
        return extra if isinstance(extra, Mapping) else {}

    def _validate_runtime(self, backend: Any) -> None:
        extra = self._extra()
        actual = {
            "profile": self.profile,
            "cfm_steps": int(getattr(backend, "n_timesteps", -1)),
            "codec_chunk_frames": int(extra.get("codec_chunk_frames", -1)),
            "left_context_frames": int(extra.get("codec_left_context_frames", -1)),
        }
        expected = {
            "profile": _CERTIFIED_PROFILE,
            "cfm_steps": _CERTIFIED_CFM_STEPS,
            "codec_chunk_frames": _CERTIFIED_CODEC_CHUNK_FRAMES,
            "left_context_frames": _CERTIFIED_LEFT_CONTEXT_FRAMES,
        }
        self._cfm_steps = actual["cfm_steps"]
        self._codec_chunk_frames = actual["codec_chunk_frames"]
        self._left_context_frames = actual["left_context_frames"]
        if self.mode == "on" and actual != expected:
            raise RuntimeError(
                "MiniCPM-o Stage2 graph profile mismatch "
                f"expected={json.dumps(expected, sort_keys=True)} "
                f"actual={json.dumps(actual, sort_keys=True)}"
            )
        flow = getattr(backend, "flow", None)
        if bool(getattr(flow, "training", False)):
            raise RuntimeError("MiniCPM-o Stage2 graph requires flow.eval()")
        if getattr(backend, "_trt_stepper", None) is not None:
            raise RuntimeError("Stage2 estimator graph cannot wrap a TRT stepper")
        if getattr(backend, "_cfm_graph_wrapper", None) is not None:
            raise RuntimeError("Stage2 estimator graph cannot wrap Whole-CFM graph/GE")
        if self.mode == "on" and not SealedNPUExactGraphRunner.is_supported():
            raise RuntimeError("required private-pool NPUGraph APIs are unavailable")
        logger.info(
            "MiniCPM-o Stage2 graph runtime mode=%s profile=%s",
            self.mode,
            json.dumps(actual, sort_keys=True),
        )

    @contextmanager
    def _semantic_scope(self, **values: Any):
        previous = self._semantic
        self._semantic = dict(values)
        try:
            yield
        finally:
            self._semantic = previous

    def _build_prompt_census(
        self,
        backend: Any,
    ) -> list[tuple[_PromptCensusRow, Any]]:
        if len(self._prompt_specs) != _CERTIFIED_PROMPT_WAV_COUNT:
            raise RuntimeError("runtime prompt allowlist was not validated")
        rows: list[tuple[_PromptCensusRow, Any]] = []
        try:
            for index, spec in enumerate(self._prompt_specs):
                cache_id = f"__minicpmo45_stage2_graph_census_{index}_{spec.sha256[:16]}"
                features = backend.prepare_prompt(cache_id, spec.path)
                prompt_frames = int(features.mels.shape[1])
                if prompt_frames <= 0:
                    backend.evict_prompt(cache_id, spec.path)
                    raise RuntimeError(
                        f"MiniCPM Stage2 runtime prompt produced invalid mel width "
                        f"path={spec.path} prompt_frames={prompt_frames}"
                    )
                expected_shape = list(spec.manifest_row.get("prompt_mel_shape", []))
                actual_shape = list(features.mels.shape)
                expected_mel_sha = str(spec.manifest_row.get("prompt_mel_sha256", ""))
                actual_mel_sha = tensor_sha256(features.mels)
                if (
                    prompt_frames != int(spec.manifest_row.get("prompt_frames", -1))
                    or actual_shape != expected_shape
                    or actual_mel_sha != expected_mel_sha
                ):
                    backend.evict_prompt(cache_id, spec.path)
                    raise RuntimeError(
                        f"MiniCPM Stage2 startup prompt features differ from sealed live manifest path={spec.path}"
                    )
                row = _PromptCensusRow(
                    path=spec.path,
                    sha256=spec.sha256,
                    prompt_frames=prompt_frames,
                    manifest_row=dict(spec.manifest_row),
                )
                rows.append((row, features))
        except Exception:
            for index, (row, _) in enumerate(rows):
                cache_id = f"__minicpmo45_stage2_graph_census_{index}_{row.sha256[:16]}"
                backend.evict_prompt(cache_id, row.path)
            raise
        self._prompt_census = tuple(row for row, _ in rows)
        self._allowed_prompt_frames = frozenset(row.prompt_frames for row in self._prompt_census)
        self._expected_roles = frozenset(
            role for prompt_frames in self._allowed_prompt_frames for role in _roles_for_prompt_frames(prompt_frames)
        )
        if len(self._expected_roles) != 4 * len(self._allowed_prompt_frames):
            raise RuntimeError("runtime prompt role construction is not one-to-one")
        return rows

    def _steady_role(
        self,
        prompt_frames: int,
        att_cache: torch.Tensor | None,
    ) -> str:
        if prompt_frames not in self._allowed_prompt_frames:
            return f"ineligible_prompt_pf{prompt_frames}"
        if att_cache is None:
            return f"ineligible_cache_pf{prompt_frames}_ac0"
        return _steady_role_for_lengths(
            prompt_frames,
            int(att_cache.shape[-2]),
        )

    @staticmethod
    def _evict_census(
        backend: Any,
        census: list[tuple[_PromptCensusRow, Any]],
    ) -> None:
        for index, (row, _) in enumerate(census):
            cache_id = f"__minicpmo45_stage2_graph_census_{index}_{row.sha256[:16]}"
            backend.evict_prompt(cache_id, row.path)

    def _current_semantic(
        self,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> EstimatorSemantic:
        scope = self._semantic or {
            "role": "ineligible_unscoped",
            "request_batch": 0,
            "prompt_frames": 0,
            "codec_token_count": 0,
            "last_chunk": False,
            "flush_encoder": False,
        }
        role = str(scope["role"])
        prompt_frames = int(scope["prompt_frames"])
        if role == "prompt_b1":
            role = (
                _prompt_role(prompt_frames)
                if prompt_frames in self._allowed_prompt_frames
                else f"ineligible_prompt_pf{prompt_frames}"
            )
        if role == "steady_b1":
            role = self._steady_role(prompt_frames, att_cache)
        return EstimatorSemantic(
            role=role,
            request_batch=int(scope["request_batch"]),
            prompt_frames=prompt_frames,
            codec_token_count=int(scope["codec_token_count"]),
            last_chunk=bool(scope["last_chunk"]),
            flush_encoder=bool(scope["flush_encoder"]),
            cache_present=cnn_cache is not None,
            attention_cache_length=(int(att_cache.shape[-2]) if att_cache is not None else 0),
            cfm_steps=self._cfm_steps,
            codec_chunk_frames=self._codec_chunk_frames,
            left_context_frames=self._left_context_frames,
        )

    def _run_fixed_cfm3_solve(
        self,
        bound_backend: Any,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.runner is None:
            raise RuntimeError("Stage2 graph hook installed without a runner")
        if (cnn_cache is None) != (att_cache is None):
            raise ValueError("CFM CNN and attention caches must both be present or absent")
        semantic = self._current_semantic(cnn_cache=cnn_cache, att_cache=att_cache)
        original = self._original_decode_cfm
        if not callable(original):
            raise RuntimeError("Stage2 graph hook lost its original CFM solve method")

        def stock_compute() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return original(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )

        # These paths remain byte-for-byte stock and do not pay graph-only
        # tensor preparation.
        if self.runner.mode == "runtime_only" or semantic.role == "tail_b1" or semantic.role.startswith("ineligible_"):
            stock_inputs = (mu, speakers, cond)
            if cnn_cache is not None:
                stock_inputs += (cnn_cache, att_cache)
            return self.runner.run(
                "fixed_cfm3_solve",
                semantic,
                stock_inputs,
                stock_compute=stock_compute,
                graph_compute=None,
            )

        decoder = bound_backend.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        offset = int(att_cache.shape[4]) if att_cache is not None else 0
        end = offset + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            return stock_compute()
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        timeline = torch.linspace(
            0,
            1,
            _CERTIFIED_CFM_STEPS + 1,
            device=mu.device,
            dtype=mu.dtype,
        )
        timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
        # Preserve the stock solver's exact floating-point update order. Direct
        # timeline slicing can differ from cumulative ``time += dt`` by one ULP.
        time = timeline[0].expand(batch_size)
        dt = timeline[1] - timeline[0]
        time_embedding_rows: list[torch.Tensor] = []
        dt_rows: list[torch.Tensor] = []
        for step in range(_CERTIFIED_CFM_STEPS):
            time_embedding_rows.append(estimator.t_embedder(torch.cat((time, time), dim=0)).unsqueeze(1))
            dt_rows.append(dt)
            time = time + dt
            if step + 1 < _CERTIFIED_CFM_STEPS:
                dt = timeline[step + 2] - time[0]
        time_embeddings = torch.stack(time_embedding_rows).contiguous()
        dts = torch.stack(dt_rows).contiguous()
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)

        if cnn_cache is None:
            inputs = (x, mu_cfg, time_embeddings, speakers_cfg, cond_cfg, dts)

            def graph_compute(
                solve_x: torch.Tensor,
                solve_mu: torch.Tensor,
                solve_times: torch.Tensor,
                solve_speakers: torch.Tensor,
                solve_cond: torch.Tensor,
                solve_dts: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return _graphable_fixed_cfm3_solve(
                    bound_backend,
                    estimator,
                    decoder,
                    x=solve_x,
                    mu_cfg=solve_mu,
                    time_embeddings=solve_times,
                    speakers_cfg=solve_speakers,
                    cond_cfg=solve_cond,
                    dts=solve_dts,
                    cnn_cache=None,
                    att_cache=None,
                )

        else:
            inputs = (
                x,
                mu_cfg,
                time_embeddings,
                speakers_cfg,
                cond_cfg,
                dts,
                cnn_cache,
                att_cache,
            )

            def graph_compute(
                solve_x: torch.Tensor,
                solve_mu: torch.Tensor,
                solve_times: torch.Tensor,
                solve_speakers: torch.Tensor,
                solve_cond: torch.Tensor,
                solve_dts: torch.Tensor,
                solve_cnn: torch.Tensor,
                solve_att: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return _graphable_fixed_cfm3_solve(
                    bound_backend,
                    estimator,
                    decoder,
                    x=solve_x,
                    mu_cfg=solve_mu,
                    time_embeddings=solve_times,
                    speakers_cfg=solve_speakers,
                    cond_cfg=solve_cond,
                    dts=solve_dts,
                    cnn_cache=solve_cnn,
                    att_cache=solve_att,
                )

        if (
            self.runner.mode == "on"
            and self._resident_graph is not None
            and self._resident_graph.ready
            and not self._internal_warmup
        ):
            outputs = self._resident_graph.replay(
                semantic.role,
                inputs[:6],
                cnn_cache,
                att_cache,
            )
            if outputs is not None:
                self.runner.record_external_replay(
                    "fixed_cfm3_solve",
                    semantic,
                    inputs,
                )
                return outputs

        stock_outputs = self.runner.run(
            "fixed_cfm3_solve",
            semantic,
            inputs,
            stock_compute=stock_compute,
            graph_compute=graph_compute,
        )
        if (
            self.runner.mode == "on"
            and self._internal_warmup
            and self._resident_graph is not None
            and semantic.role in self._resident_graph.expected_roles
        ):
            return self._resident_graph.prepare_transition(
                semantic.role,
                inputs[:6],
                cnn_cache,
                att_cache,
                stock_outputs,
            )
        return stock_outputs

    def _run_setup(self, bound_backend: Any, features: Any, batch_size: int):
        if self._original_setup_batch is None:
            raise RuntimeError("Stage2 setup hook lost its original method")
        if batch_size == 1:
            role = "prompt_b1"
        else:
            role = "ineligible_batch"
        if not self._internal_warmup:
            self._request_ordinal += 1
            assert self.runner is not None
            self.runner.log_snapshot("request_start", self._request_ordinal)
        with (
            self._semantic_scope(
                role=role,
                request_batch=batch_size,
                prompt_frames=int(features.mels.shape[1]),
                codec_token_count=0,
                last_chunk=False,
                flush_encoder=False,
            ),
            _flow_context(enabled=True),
        ):
            return self._original_setup_batch(features, batch_size)

    def _run_decode(
        self,
        bound_backend: Any,
        tokens: torch.Tensor,
        features: Any,
        states: list[Any],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ):
        if self._original_decode_batch is None:
            raise RuntimeError("Stage2 decode hook lost its original method")
        batch_size = int(tokens.shape[0])
        token_count = int(tokens.shape[1])
        if batch_size != 1:
            role = "ineligible_batch"
        elif last_chunk or flush_encoder or token_count != _CERTIFIED_CODEC_INPUT_WIDTH:
            role = "tail_b1"
        else:
            role = "steady_b1"
        with (
            self._semantic_scope(
                role=role,
                request_batch=batch_size,
                prompt_frames=int(features.mels.shape[1]),
                codec_token_count=token_count,
                last_chunk=last_chunk,
                flush_encoder=flush_encoder,
            ),
            _flow_context(enabled=True),
        ):
            outputs = self._original_decode_batch(
                tokens,
                features,
                states,
                last_chunk=last_chunk,
                flush_encoder=flush_encoder,
            )
        if last_chunk and not self._internal_warmup:
            assert self.runner is not None
            self.runner.log_snapshot("request_end", self._request_ordinal)
            if self._resident_graph is not None:
                logger.info(
                    "MiniCPMO45Stage2ResidentCFM3Graph %s",
                    json.dumps(self._resident_graph.snapshot(), sort_keys=True),
                )
        return outputs

    def _install_instance_hooks(self, backend: Any) -> None:
        installed = getattr(backend, _BOOTSTRAP_ATTR, None)
        if installed is not None:
            if installed is self:
                return
            raise RuntimeError("MiniCPM Stage2 backend already has a different graph bootstrap")
        self._original_decode_cfm = backend._decode_cfm
        self._original_setup_batch = backend.setup_batch
        self._original_decode_batch = backend.decode_batch
        self._original_stack_flow_cache = backend._stack_flow_cache
        self._original_split_flow_cache = backend._split_flow_cache

        bootstrap = self

        def cfm_hook(
            bound_backend: Any,
            mu: torch.Tensor,
            speakers: torch.Tensor,
            cond: torch.Tensor,
            *,
            cnn_cache: torch.Tensor | None,
            att_cache: torch.Tensor | None,
        ):
            return bootstrap._run_fixed_cfm3_solve(
                bound_backend,
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )

        def setup_hook(bound_backend: Any, features: Any, batch_size: int):
            return bootstrap._run_setup(bound_backend, features, batch_size)

        def decode_hook(
            bound_backend: Any,
            tokens: torch.Tensor,
            features: Any,
            states: list[Any],
            *,
            last_chunk: bool,
            flush_encoder: bool = False,
        ):
            return bootstrap._run_decode(
                bound_backend,
                tokens,
                features,
                states,
                last_chunk=last_chunk,
                flush_encoder=flush_encoder,
            )

        def stack_flow_cache_hook(bound_backend: Any, states: list[Any]):
            original = bootstrap._original_stack_flow_cache
            if not callable(original):
                raise RuntimeError("Stage2 graph hook lost original cache stack")
            resident = bootstrap._resident_graph
            if resident is None or len(states) != 1:
                return original(states)
            flow = states[0].flow_cache
            if not resident.owns(
                flow["estimator_cnn_cache"],
                flow["estimator_att_cache"],
            ):
                return original(states)
            # Keep the stock conformer materialization contract. Only the two
            # large estimator CFG-row cats disappear after cache residency.
            return {
                "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"]], dim=0),
                "conformer_att_cache": torch.cat([flow["conformer_att_cache"]], dim=1),
                "estimator_cnn_cache": flow["estimator_cnn_cache"],
                "estimator_att_cache": flow["estimator_att_cache"],
            }

        def split_flow_cache_hook(
            bound_backend: Any,
            cache: dict[str, torch.Tensor],
            batch_size: int,
        ):
            original = bootstrap._original_split_flow_cache
            if not callable(original):
                raise RuntimeError("Stage2 graph hook lost original cache split")
            resident = bootstrap._resident_graph
            if resident is None or batch_size != 1:
                return original(cache, batch_size)
            resolved = resident.resolve_compacted_output(cache["estimator_cnn_cache"], cache["estimator_att_cache"])
            if resolved is None:
                return original(cache, batch_size)
            estimator_cnn, estimator_att = resolved
            return [
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"][0:1].detach().clone(),
                    "conformer_att_cache": cache["conformer_att_cache"][:, 0:1].detach().clone(),
                    "estimator_cnn_cache": estimator_cnn.detach(),
                    "estimator_att_cache": estimator_att.detach(),
                }
            ]

        def precompacted_estimator_cache_hook(
            bound_backend: Any,
            cnn_cache: torch.Tensor,
            full_att_cache: torch.Tensor,
            prompt_len: int,
        ):
            resident = bootstrap._resident_graph
            if resident is None:
                return None
            return resident.resolve_precompacted_output(cnn_cache, full_att_cache, prompt_len)

        backend._decode_cfm = MethodType(cfm_hook, backend)
        backend.setup_batch = MethodType(setup_hook, backend)
        backend.decode_batch = MethodType(decode_hook, backend)
        backend._stack_flow_cache = MethodType(stack_flow_cache_hook, backend)
        backend._split_flow_cache = MethodType(split_flow_cache_hook, backend)
        backend._resolve_precompacted_estimator_cache = MethodType(precompacted_estimator_cache_hook, backend)
        setattr(backend, _BOOTSTRAP_ATTR, self)
        self._installed = True

    def _advance_flow_only(
        self,
        tokens: torch.Tensor,
        features: Any,
        states: list[Any],
    ) -> list[Any]:
        """Advance the real flow state once without touching HiFT or audio RNG."""
        assert self.backend is not None
        backend = self.backend
        flow_cache = backend._stack_flow_cache(states)
        speakers = features.speaker_embedding.expand(1, -1)
        with (
            self._semantic_scope(
                role="steady_b1",
                request_batch=1,
                prompt_frames=int(features.mels.shape[1]),
                codec_token_count=int(tokens.shape[1]),
                last_chunk=False,
                flush_encoder=False,
            ),
            _flow_context(enabled=True),
            backend._autocast(tokens.device),
        ):
            hidden, conformer_cnn, conformer_att = backend._encode_chunk(
                tokens,
                last_chunk=False,
                cnn_cache=flow_cache["conformer_cnn_cache"],
                att_cache=flow_cache["conformer_att_cache"],
            )
            projected_speakers = backend.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            chunk_mel, estimator_cnn, estimator_att = backend._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
            )
        if int(chunk_mel.shape[2]) != _CERTIFIED_ESTIMATOR_WIDTH:
            raise RuntimeError(
                "MiniCPM Stage2 warmup estimator width mismatch "
                f"expected={_CERTIFIED_ESTIMATOR_WIDTH} actual={int(chunk_mel.shape[2])}"
            )
        prompt_len = int(features.mels.shape[1])
        if estimator_att.shape[4] > prompt_len + 100:
            estimator_att = torch.cat(
                (estimator_att[..., :prompt_len, :], estimator_att[..., -100:, :]),
                dim=4,
            )
        if conformer_att.shape[3] > prompt_len + 100:
            conformer_att = torch.cat(
                (conformer_att[..., :prompt_len, :], conformer_att[..., -100:, :]),
                dim=3,
            )
        split = backend._split_flow_cache(
            {
                "conformer_cnn_cache": conformer_cnn,
                "conformer_att_cache": conformer_att,
                "estimator_cnn_cache": estimator_cnn,
                "estimator_att_cache": estimator_att,
            },
            1,
        )
        return [replace(states[0], flow_cache=split[0])]

    def _warm_and_seal(
        self,
        census: list[tuple[_PromptCensusRow, Any]],
    ) -> None:
        assert self.backend is not None
        assert self.runner is not None
        backend = self.backend
        self._internal_warmup = True
        try:
            with torch.inference_mode():
                for _, features in census:
                    states = backend.setup_batch(features, 1)
                    tokens = torch.full(
                        (1, _CERTIFIED_CODEC_INPUT_WIDTH),
                        _SILENCE_TOKEN,
                        device=features.speech_tokens.device,
                        dtype=torch.long,
                    )
                    for _ in _CERTIFIED_STEADY_CACHE_OFFSETS:
                        states = self._advance_flow_only(tokens, features, states)
        finally:
            self._internal_warmup = False
            self._evict_census(backend, census)

        unique_prompt_lengths = len(self._allowed_prompt_frames)
        expected_graphs = 4 * unique_prompt_lengths
        expected_warm_solve_calls = _CERTIFIED_PROMPT_WAV_COUNT * 4
        expected = {
            "capture_requests": expected_graphs,
            "captures": expected_graphs,
            "capture_failures": 0,
            # One fixed solve graph owns all three CFM steps for each role.
            # Startup state stays stock-owned; replay is validation-only.
            "shadow_replay_successes": expected_warm_solve_calls,
            "shadow_replay_failures": 0,
            "replay_successes": 0,
            "replay_failures": 0,
            "eager_warm_calls": expected_warm_solve_calls,
            "resident_graphs": expected_graphs,
            "sealed_misses": 0,
            "tail_eager_calls": 0,
        }
        snapshot = self.runner.snapshot("pre_seal")
        actual = {key: snapshot[key] for key in expected}
        if actual != expected:
            raise RuntimeError(
                "MiniCPM Stage2 warmup counter mismatch "
                f"expected={json.dumps(expected, sort_keys=True)} "
                f"actual={json.dumps(actual, sort_keys=True)}"
            )
        per_role = {entry["role"]: entry for entry in snapshot["per_key"]}
        if frozenset(per_role) != self._expected_roles:
            raise RuntimeError(
                f"MiniCPM Stage2 warmup role mismatch expected={sorted(self._expected_roles)} actual={sorted(per_role)}"
            )
        for role, entry in per_role.items():
            role_counts = {
                "captures": entry["captures"],
                "shadow_replays": entry["shadow_replays"],
                "replay_successes": entry["replay_successes"],
            }
            prompt_frames = next(
                frames for frames in self._allowed_prompt_frames if role in _roles_for_prompt_frames(frames)
            )
            prompt_multiplicity = sum(row.prompt_frames == prompt_frames for row in self._prompt_census)
            expected_counts = {
                "captures": 1,
                "shadow_replays": prompt_multiplicity,
                "replay_successes": 0,
            }
            if role_counts != expected_counts:
                raise RuntimeError(
                    "MiniCPM Stage2 warmup per-role mismatch "
                    f"role={role} expected={expected_counts} actual={role_counts}"
                )
        self.runner.seal()
        self.runner.log_snapshot("warmup_sealed")

    def install_instance_hooks_and_warm(self, backend: Any) -> None:
        if self.mode == "off":
            logger.info("MiniCPM-o Stage2 estimator NPUGraph mode=off")
            return
        self._validate_runtime(backend)
        self.backend = backend
        self._prompt_specs = _parse_prompt_wav_specs(self._extra())
        census = self._build_prompt_census(backend)
        prompt_contract = {
            "profile": self.profile,
            "prompt_manifest_path": str(Path(str(self._extra()[_PROMPT_MANIFEST_KEY])).resolve(strict=True)),
            "prompt_manifest_sha256": str(self._extra()[_PROMPT_MANIFEST_SHA_KEY]),
            "prompt_wav_count": len(self._prompt_census),
            "unique_prompt_lengths": sorted(self._allowed_prompt_frames),
            "expected_roles": sorted(self._expected_roles),
            "prompt_census": [
                {
                    **row.manifest_row,
                    "path": row.path,
                    "sha256": row.sha256,
                    "prompt_frames": row.prompt_frames,
                }
                for row in self._prompt_census
            ],
        }
        try:
            self.runner = SealedNPUExactGraphRunner(
                mode=self.mode,
                expected_roles=self._expected_roles,
                component_name="MiniCPM-o Stage2 fixed CFM3 solve",
                contract_metadata=prompt_contract,
                # Final mel is consumed by HiFT on the same stream before the
                # next solve replay. CNN/attention caches remain detached from
                # graph-owned buffers because request state retains them.
                ephemeral_output_indices=(0,),
            )
            if self.mode == "on":
                if len(self._allowed_prompt_frames) != 1:
                    raise RuntimeError("resident CFM3 state pipeline requires one prompt length")
                self._resident_graph = _ResidentCFM3GraphPair(
                    backend,
                    backend.flow.decoder.estimator,
                    backend.flow.decoder,
                    prompt_frames=next(iter(self._allowed_prompt_frames)),
                )
            self._install_instance_hooks(backend)
            if self.mode == "on":
                self._warm_and_seal(census)
            else:
                self._evict_census(backend, census)
                self.runner.log_snapshot("runtime_only_ready")
        except Exception:
            self._evict_census(backend, census)
            raise


__all__ = [
    "Stage2EstimatorGraphBootstrap",
]
