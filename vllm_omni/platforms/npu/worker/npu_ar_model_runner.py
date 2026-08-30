# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, PerLayerAttnMetadata
from vllm.v1.worker.mamba_utils import preprocess_mamba
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import enable_sp, global_stream, lmhead_tp_enable
from vllm_ascend.worker.model_runner_v1 import graph_capture

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.distributed.omni_connectors.utils.config import get_stage_connector_role, stage_sends_async_output
from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRunnerMixin
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.platforms.npu.worker.npu_model_runner import OmniNPUModelRunner
from vllm_omni.utils.mm_outputs import build_mm_cpu, partition_payload_list, to_payload_element
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin
from vllm_omni.worker.sampling_utils import sanitize_min_tokens_stop_ids


def _ensure_tensor_values(payload: dict[str, object]) -> dict[str, torch.Tensor]:
    """Convert a flattened payload to strictly ``dict[str, torch.Tensor]``.

    Non-tensor scalars (int, float, bool) are wrapped with ``torch.tensor()``.
    Values that cannot be safely converted are dropped with a warning.
    This enforces the tensor-only invariant required by the
    ``OmniEngineCoreOutput.multimodal_output`` wire field and msgspec
    serialization. Mirrors ``gpu_ar_model_runner._ensure_tensor_values``.
    """
    result: dict[str, torch.Tensor] = {}
    for key, val in payload.items():
        if isinstance(val, torch.Tensor):
            result[key] = val
        elif isinstance(val, (int, float, bool)):
            result[key] = torch.tensor(val)
        elif isinstance(val, (list, tuple)):
            try:
                result[key] = torch.tensor(val)
            except (ValueError, TypeError, RuntimeError):
                logger.warning(
                    "Dropping non-tensorizable multimodal output key '%s' (type=%s) from wire payload.",
                    key,
                    type(val).__name__,
                )
        else:
            logger.warning(
                "Dropping non-tensor multimodal output key '%s' (type=%s) from wire payload.",
                key,
                type(val).__name__,
            )
    return result


_MINICPMO45_C1_CPU_SLOT_ENV = "VLLM_OMNI_MINICPMO45_STAGE1_C1_CPU_SLOT_MAPPING"
_MINICPMO45_STAGE1_HOST_LEDGER_ENV = "VLLM_OMNI_MINICPMO45_STAGE1_HOST_LEDGER"
_MINICPMO45_C1_FAST_PREP_ENV = "VLLM_OMNI_MINICPMO45_STAGE1_C1_FAST_PREP"
_MINICPMO45_C1_HOST_FAST_ENV = "VLLM_OMNI_MINICPMO45_STAGE1_C1_HOST_FASTPATH"
_MINICPMO45_C1_EXECUTION_PLAN_ENV = (
    "VLLM_OMNI_MINICPMO45_STAGE1_C1_EXECUTION_PLAN"
)
_MINICPMO45_CODEC_EMBED_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_STAGE1_CODEC_EMBED_GRAPH"
_MINICPMO45_KNOWN_CONTROLLER_BYPASS_ENV = (
    "VLLM_OMNI_MINICPMO45_STAGE1_KNOWN_CONTROLLER_BYPASS"
)


class _AlreadyReadyEvent:
    """Event-compatible marker for host data requiring no device copy."""

    @staticmethod
    def synchronize() -> None:
        return None


_ALREADY_READY_EVENT = _AlreadyReadyEvent()
_KNOWN_CONTROLLER_CPU_TOKENS = {
    token: torch.tensor([[token]], dtype=torch.int32) for token in (0, 1)
}


class _KnownControllerAsyncModelRunnerOutput(AsyncModelRunnerOutput):
    """Async-shaped output for a controller token already known on host."""

    def __init__(
        self,
        model_runner_output: OmniModelRunnerOutput,
        token: int,
        invalid_req_indices: list[int],
    ) -> None:
        if token not in (0, 1):
            raise ValueError(f"Known controller token must be binary, got {token}")
        if len(model_runner_output.req_ids) != 1:
            raise ValueError("Known controller output requires exactly one request")
        self._model_runner_output = model_runner_output
        self._token = token
        self._invalid_req_indices = tuple(invalid_req_indices)
        self._consumed = False
        self.sampled_token_ids_cpu = _KNOWN_CONTROLLER_CPU_TOKENS[token]
        self.async_copy_ready_event = _ALREADY_READY_EVENT

    def get_output(self) -> OmniModelRunnerOutput:
        if self._consumed:
            raise RuntimeError("Known controller async output was consumed twice")
        self._consumed = True
        sampled_token_ids = [[self._token]]
        for index in self._invalid_req_indices:
            sampled_token_ids[index].clear()
        self._model_runner_output.sampled_token_ids = sampled_token_ids
        self._model_runner_output.logprobs = None
        return self._model_runner_output


@dataclass(frozen=True, slots=True)
class _C1DecodeFastStep:
    """One strict single-request Talker decode step owned by the fast lane."""

    req_id: str
    position: int


@dataclass(slots=True)
class _C1ExecutionPlan:
    """Reusable host-only plan for one uninterrupted strict C=1 decode run.

    It owns no KV values.  It retains only views into persistent metadata
    buffers and one immutable FULL graph dispatch result.  Request, position,
    object identity, and storage identity are checked before every replay.
    """

    req_id: str
    last_position: int
    table_obj_id: int
    block_storage_ptr: int
    slot_storage_ptr: int
    seq_storage_ptr: int
    determine_result: tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]
    attn_metadata: PerLayerAttnMetadata | None = None
    unique_metadata: tuple[AscendMetadata, ...] = ()



def _stage1_host_ledger(bucket: str):
    """Time one runner method when the opt-in Stage1 ledger is active."""

    def decorate(fn):
        @wraps(fn)
        def timed(self, *args, **kwargs):
            if not getattr(self, "_stage1_host_ledger_enabled", False):
                return fn(self, *args, **kwargs)
            before = time.perf_counter_ns()
            before_cpu = time.thread_time_ns()
            try:
                return fn(self, *args, **kwargs)
            finally:
                self._record_stage1_host_ledger(
                    bucket,
                    time.perf_counter_ns() - before,
                    time.thread_time_ns() - before_cpu,
                )

        return timed

    return decorate


def _parse_bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean string, got {raw!r}")


def _c1_cpu_slot_from_table(block_table: Any, position: int) -> int | None:
    """Resolve one decode slot from the authoritative CPU block-table row.

    The block-table manager updates this row before input preparation.  Read it
    at each step so physical block reuse remains visible, and preserve the
    physical/logical split used by hybrid block tables.  Returning ``None`` is
    always a request to use the stock NPU slot-mapping implementation.
    """

    try:
        position = int(position)
        physical_block_size = int(block_table.physical_block_size)
        block_size = int(block_table.block_size)
        blocks_per_phys_block = int(block_table.blocks_per_phys_block)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if (
        position < 0
        or physical_block_size <= 0
        or block_size <= 0
        or blocks_per_phys_block <= 0
    ):
        return None

    physical_index, physical_offset = divmod(position, physical_block_size)
    logical_index = (
        physical_index * blocks_per_phys_block
        + physical_offset // block_size
    )
    try:
        num_blocks_per_row = block_table.num_blocks_per_row
        num_blocks = int(num_blocks_per_row[0])
        block_row = block_table.block_table.np
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if logical_index >= num_blocks:
        return None

    try:
        invalid_row = (
            block_row.ndim != 2
            or block_row.shape[0] < 1
            or logical_index >= block_row.shape[1]
        )
    except (AttributeError, IndexError, TypeError):
        return None
    if invalid_row:
        return None
    try:
        block_id = int(block_row[0, logical_index])
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if block_id < 0:
        return None
    return block_id * block_size + physical_offset % block_size


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: PerLayerAttnMetadata
    positions: torch.Tensor
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    batch_desc: BatchDescriptor
    multimodal_outputs: Any # Omni-Specific

class NPUARModelRunner(OmniNPUModelRunner, OmniConnectorModelRunnerMixin, DuplexSamplingRunnerMixin):
    """Autoregressive NPU model runner that returns hidden states per request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        # Initialize KV cache manager (preserve vllm_config fallback behavior)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
        self._async_chunk = getattr(self.model_config, "async_chunk", False)
        _OMNI_CONNECTOR_INIT_ARCHS = {
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen2_5OmniForConditionalGeneration",
            "CovoAudioForConditionalGeneration",
            "MiMoAudioModel",
            "Qwen3TTSTalkerForConditionalGeneration",
            "Qwen3TTSCode2Wav",
            "CosyVoice3Model",
            "DyninOmniForConditionalGeneration",
            "IndexTTS2TalkerForConditionalGeneration",
        }
        # Mirrors gpu_ar_model_runner: an arch missing from the hardcoded allowlist
        # still needs connectors when the deploy config hands the stage a
        # sender/receiver role (e.g. MiniCPM-o 4.5, whose archs are not listed but
        # whose YAML wires stage 1 -> stage 2). Without the role check the
        # full-payload (``--no-async-chunk``) handoff never initializes: nothing
        # accumulates, nothing flushes, and the downstream stage starves silently.
        if (
            getattr(self.model_config, "model_arch", None) in _OMNI_CONNECTOR_INIT_ARCHS
            or get_stage_connector_role(self.model_config) is not None
        ):
            self.init_omni_connectors(
                model_config=self.model_config,
                kv_transfer_manager=self.kv_transfer_manager,
            )
        self._downstream_payload_cache: dict[str, bool] = {}
        self._init_duplex_sampling_state()
        hf_config = getattr(self.model_config, "hf_config", None)
        self._c1_cpu_slot_enabled = bool(
            _parse_bool_env(_MINICPMO45_C1_CPU_SLOT_ENV, default=True)
            and getattr(self.model_config, "model_stage", None) == "tts"
            and str(getattr(hf_config, "version", "")) == "4.5"
        )
        self._c1_cpu_slot_hits = 0
        self._c1_cpu_slot_boundaries = 0
        self._c1_cpu_slot_fallbacks: Counter[str] = Counter()
        self._c1_fast_prep_enabled = bool(
            _parse_bool_env(_MINICPMO45_C1_FAST_PREP_ENV, default=True)
            and getattr(self.model_config, "model_stage", None) == "tts"
            and str(getattr(hf_config, "version", "")) == "4.5"
        )
        self._c1_fast_step: _C1DecodeFastStep | None = None
        self._c1_fast_static_ready = False
        self._c1_fast_hits = 0
        self._c1_fast_rejects: Counter[str] = Counter()
        self._c1_fast_one_np = np.ones(1, dtype=np.int32)
        self._c1_fast_query_lens = torch.from_numpy(self._c1_fast_one_np)
        self._c1_fast_logits_indices: torch.Tensor | None = None

        self._c1_execution_plan_enabled = bool(
            _parse_bool_env(_MINICPMO45_C1_EXECUTION_PLAN_ENV, default=True)
            and self._c1_fast_prep_enabled
        )
        self._c1_execution_plan: _C1ExecutionPlan | None = None
        self._c1_execution_plan_captures = 0
        self._c1_execution_plan_hits = 0
        self._c1_execution_plan_rejects: Counter[str] = Counter()
        if self._c1_execution_plan_enabled:
            logger.info(
                "MINICPMO45_STAGE1_C1_EXECUTION_PLAN event=enabled "
                "default_off=true"
            )
        self._c1_host_fast_enabled = bool(
            _parse_bool_env(_MINICPMO45_C1_HOST_FAST_ENV, default=False)
            and self._c1_fast_prep_enabled
        )
        self._c1_host_fast_info: dict[str, Any] | None = None
        self._c1_host_fast_infos: list[dict[str, Any]] = [{}]
        self._c1_host_fast_spans = [(0, 1)]
        self._c1_host_fast_sample_eligible = [True]
        self._c1_host_fast_preprocess_hits = 0
        self._c1_host_fast_kwargs_hits = 0
        self._c1_host_fast_sampler_history_skips = 0
        self._known_controller_bypass_enabled = bool(
            _parse_bool_env(
                _MINICPMO45_KNOWN_CONTROLLER_BYPASS_ENV,
                default=False,
            )
            and getattr(self.model_config, "model_stage", None) == "tts"
            and str(getattr(hf_config, "version", "")) == "4.5"
        )
        self._known_controller_device_tokens: dict[
            tuple[int, torch.device], torch.Tensor
        ] = {}
        self._known_controller_token_for_output: int | None = None
        self._known_controller_bypass_hits = 0
        self._codec_embed_graph_enabled = bool(
            _parse_bool_env(
                _MINICPMO45_CODEC_EMBED_GRAPH_ENV, default=True
            )
            and self._c1_fast_prep_enabled
        )
        self._codec_embed_graph_hits = 0
        self._stage1_host_ledger_enabled = bool(
            os.environ.get(_MINICPMO45_STAGE1_HOST_LEDGER_ENV, "0") == "1"
            and getattr(self.model_config, "model_stage", None) == "tts"
            and str(getattr(hf_config, "version", "")) == "4.5"
        )
        self._stage1_host_ledger_ns: Counter[str] = Counter()
        self._stage1_host_ledger_cpu_ns: Counter[str] = Counter()
        self._stage1_host_ledger_counts: Counter[str] = Counter()
        if self._stage1_host_ledger_enabled:
            for name, bucket in (
                ("_preprocess", "preprocess"),
                ("_model_forward", "model_forward"),
                ("extract_multimodal_outputs", "extract_multimodal"),
                ("_bookkeeping_sync", "bookkeeping_sync"),
                ("_sample", "sample_core"),
            ):
                self._wrap_stage1_host_ledger_method(self, name, bucket)

    def _wrap_stage1_host_ledger_method(
        self,
        owner: Any,
        name: str,
        bucket: str,
    ) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            return

        @wraps(original)
        def timed(*args, **kwargs):
            before = time.perf_counter_ns()
            before_cpu = time.thread_time_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self._record_stage1_host_ledger(
                    bucket,
                    time.perf_counter_ns() - before,
                    time.thread_time_ns() - before_cpu,
                )

        setattr(owner, name, timed)

    def _record_stage1_host_ledger(
        self,
        bucket: str,
        elapsed_ns: int,
        cpu_ns: int,
    ) -> None:
        self._stage1_host_ledger_ns[bucket] += elapsed_ns
        self._stage1_host_ledger_cpu_ns[bucket] += cpu_ns
        self._stage1_host_ledger_counts[bucket] += 1
        if bucket != "sample_tokens":
            return
        calls = self._stage1_host_ledger_counts[bucket]
        if calls % 128 != 0:
            return
        means_wall_us = {
            name: self._stage1_host_ledger_ns[name]
            / max(self._stage1_host_ledger_counts[name], 1)
            / 1_000
            for name in sorted(self._stage1_host_ledger_ns)
        }
        means_cpu_us = {
            name: self._stage1_host_ledger_cpu_ns[name]
            / max(self._stage1_host_ledger_counts[name], 1)
            / 1_000
            for name in sorted(self._stage1_host_ledger_cpu_ns)
        }
        logger.info(
            "MINICPMO45_STAGE1_RUNNER_HOST_LEDGER calls=%s "
            "means_wall_us=%s means_cpu_us=%s",
            dict(self._stage1_host_ledger_counts),
            means_wall_us,
            means_cpu_us,
        )

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        if self._stage1_host_ledger_enabled:
            for name, bucket in (
                ("forward", "model_module_forward"),
                ("make_omni_output", "make_omni_output"),
                ("compute_logits", "compute_logits"),
            ):
                self._wrap_stage1_host_ledger_method(self.model, name, bucket)
        self._resolve_duplex_sampling_hook(force=True)

    @_stage1_host_ledger("update_states")
    def _update_states(self, scheduler_output: SchedulerOutput):
        deferred_state_corrections_fn = super()._update_states(scheduler_output)
        self._update_duplex_sampling_states(scheduler_output)
        return deferred_state_corrections_fn

    def _c1_fast_reject_reason(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> str | None:
        """Return why the strict C=1 steady Talker lane cannot own this step."""

        if not self._c1_fast_prep_enabled:
            return "disabled"
        if not self.use_async_scheduling:
            return "not_async"
        if int(self.input_batch.num_reqs) != 1:
            return "not_c1"
        if (
            int(scheduler_output.total_num_scheduled_tokens) != 1
            or len(num_scheduled_tokens) != 1
            or int(num_scheduled_tokens[0]) != 1
        ):
            return "not_one_token"
        if scheduler_output.scheduled_new_reqs:
            return "new_request"
        if scheduler_output.finished_req_ids:
            return "finished_request"
        if scheduler_output.scheduled_spec_decode_tokens:
            return "scheduled_spec_decode"
        if self.speculative_config is not None or int(self.num_spec_tokens) != 0:
            return "speculative_config"
        if self.use_cp or int(self.pcp_size) != 1 or int(self.dcp_size) != 1:
            return "context_parallel"
        if get_pp_group().world_size != 1 or get_tp_group().world_size != 1:
            return "model_parallel"
        if int(self.vllm_config.parallel_config.data_parallel_size) != 1:
            return "data_parallel"
        if has_kv_transfer_group() or has_ec_transfer():
            return "kv_or_ec_transfer"
        if self.lora_config:
            return "lora"
        if self.uses_mrope or int(self.uses_xdrope_dim) > 0:
            return "special_rope"
        if self.enable_prompt_embeds or self.omni_prefix_cache is not None:
            return "prompt_or_prefix_embeds"
        if self.model_config.is_encoder_decoder:
            return "encoder_decoder"
        if scheduler_output.scheduled_encoder_inputs:
            return "encoder_input"
        if getattr(self, "use_compress", False):
            return "compressed_kv"
        if self.is_pooling_model:
            return "pooling"
        if self.has_talker_mtp:
            return "talker_mtp"
        if getattr(self, "_has_gdn", False):
            return "gdn"
        if self.cache_config.kv_sharing_fast_prefill:
            return "kv_sharing_fast_prefill"
        if lmhead_tp_enable():
            return "lmhead_tp"
        if self._c1_host_fast_enabled and (
            self._omni_query_start_loc_model_kwarg
            or getattr(self.model_config, "has_sampling_extra_args", False)
        ):
            return "extra_model_kwargs"
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if capture_sizes and 1 not in capture_sizes:
            return "missing_graph_bucket_one"

        # Model-local preprocessing hooks are exposed through the runner's
        # graph wrapper.  ``unwrap()`` returns the compiled forward module and
        # is not required to retain non-forward Python helper methods.
        fast_model = self.model
        if self._codec_embed_graph_enabled and (
            not bool(
                getattr(fast_model, "supports_codec_embed_graph", False)
            )
            or not callable(
                getattr(fast_model, "set_codec_embed_graph_active", None)
            )
        ):
            return "missing_codec_embed_graph_model"
        if callable(getattr(fast_model, "preprocess_batch", None)):
            return "batch_preprocess"
        if callable(getattr(fast_model, "preprocess_decode_batch", None)):
            return "decode_batch_preprocess"

        req_id = self.input_batch.req_ids[0]
        if self.input_batch.req_id_to_index.get(req_id) != 0:
            return "request_reordered"
        position = int(self.input_batch.num_computed_tokens_cpu[0])
        prompt_tokens = int(self.input_batch.num_prompt_tokens[0])
        req_state = self.requests.get(req_id)
        if req_state is None:
            return "missing_request"
        prompt_token_ids = getattr(req_state, "prompt_token_ids", ())
        prompt_len = len(prompt_token_ids or ())
        if prompt_tokens != prompt_len:
            return "prompt_length_mismatch"
        if position < prompt_tokens or position < prompt_len:
            return "prefill"
        if position + 1 < int(req_state.num_tokens):
            return "discarded_output"

        cached_infos = getattr(
            scheduler_output.scheduled_cached_reqs,
            "additional_information",
            None,
        )
        if isinstance(cached_infos, dict) and cached_infos.get(req_id):
            return "cached_additional_information"
        local_payloads = getattr(self, "_local_stage_payload_cache", None)
        if isinstance(local_payloads, dict) and req_id in local_payloads:
            return "local_stage_payload"
        pending_payloads = getattr(
            self,
            "_full_payload_pending_broadcast_req_ids",
            None,
        )
        if pending_payloads and req_id in pending_payloads:
            return "pending_stage_payload"

        prev_positions = self.input_batch.prev_req_id_to_index
        if prev_positions.get(req_id) != 0:
            return "missing_previous_request"
        prev_tokens = self.input_batch.prev_sampled_token_ids
        if (
            not isinstance(prev_tokens, torch.Tensor)
            or prev_tokens.ndim != 2
            or prev_tokens.shape[0] < 1
            or prev_tokens.shape[1] < 1
        ):
            return "missing_previous_token"

        block_tables = self.input_batch.block_table.block_tables
        if len(block_tables) != 1:
            return "multiple_kv_groups"
        table = block_tables[0]
        if table.is_mamba_group:
            return "mamba"
        if int(table.pcp_world_size) != 1 or int(table.dcp_world_size) != 1:
            return "kv_context_parallel"

        info = self.model_intermediate_buffer.get(req_id)
        if not isinstance(info, dict):
            return "missing_audio_state"
        if info.get("native_duplex") or info.get("resumable"):
            return "streaming_session"
        state = info.get("audio_state")
        current = (info.get("audio_codes", {}) or {}).get("current")
        if (
            not isinstance(state, dict)
            or state.get("finished")
            or not isinstance(current, torch.Tensor)
            or current.shape != (1,)
            or current.dtype != torch.long
            or current.device != self.device
        ):
            return "invalid_audio_state"
        return None


    def _c1_execution_plan_table_state(
        self,
    ) -> tuple[int, int, int, int] | None:
        """Return identities for buffers whose views a plan may retain."""

        try:
            tables = self.input_batch.block_table.block_tables
            if len(tables) != 1:
                return None
            table = tables[0]
            block_tensor = table.get_device_tensor()
            slot_tensor = table.slot_mapping.gpu
            seq_tensor = self.optimistic_seq_lens_cpu
            return (
                id(table),
                int(block_tensor.data_ptr()),
                int(slot_tensor.data_ptr()),
                int(seq_tensor.data_ptr()),
            )
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return None

    def _discard_c1_execution_plan(self, reason: str) -> None:
        plan = self._c1_execution_plan
        if plan is None:
            return
        self._c1_execution_plan = None
        self._c1_execution_plan_rejects[reason] += 1
        rejects = sum(self._c1_execution_plan_rejects.values())
        if rejects == 1 or rejects % 64 == 0 or reason.startswith("contract:"):
            logger.info(
                "MINICPMO45_STAGE1_C1_EXECUTION_PLAN event=discard "
                "kind=%s reason=%s hits=%d captures=%d plan_request_id=%s "
                "last_position=%d rejects=%s",
                "contract" if reason.startswith("contract:") else "boundary",
                reason,
                self._c1_execution_plan_hits,
                self._c1_execution_plan_captures,
                plan.req_id,
                plan.last_position,
                dict(self._c1_execution_plan_rejects),
            )

    def _c1_execution_plan_matches(
        self,
        step: _C1DecodeFastStep,
    ) -> bool:
        plan = self._c1_execution_plan
        state = self._c1_execution_plan_table_state()
        if plan is None or state is None:
            return False
        return bool(
            plan.req_id == step.req_id
            and step.position == plan.last_position + 1
            and state
            == (
                plan.table_obj_id,
                plan.block_storage_ptr,
                plan.slot_storage_ptr,
                plan.seq_storage_ptr,
            )
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = False,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        step = self._c1_fast_step
        exact = bool(
            self._c1_execution_plan_enabled
            and step is not None
            and num_tokens == 1
            and num_reqs == 1
            and num_scheduled_tokens_np.shape == (1,)
            and int(num_scheduled_tokens_np[0]) == 1
            and max_num_scheduled_tokens == 1
            and not use_cascade_attn
            and not allow_microbatching
            and not force_eager
            and force_uniform_decode is None
            and force_has_lora is None
            and force_num_active_loras is None
            and num_encoder_reqs == 0
        )
        if exact and step is not None and self._c1_execution_plan_matches(step):
            assert self._c1_execution_plan is not None
            return self._c1_execution_plan.determine_result

        if self._c1_execution_plan_enabled and step is not None:
            old = self._c1_execution_plan
            boundary = old is not None and old.req_id != step.req_id
            self._discard_c1_execution_plan(
                "boundary:new_request" if boundary else "contract:dispatch"
            )
        result = super()._determine_batch_execution_and_padding(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=use_cascade_attn,
            allow_microbatching=allow_microbatching,
            force_eager=force_eager,
            force_uniform_decode=force_uniform_decode,
            force_has_lora=force_has_lora,
            force_num_active_loras=force_num_active_loras,
            num_encoder_reqs=num_encoder_reqs,
        )
        if not exact or step is None:
            return result

        cudagraph_mode, batch_desc, should_ubatch, across_dp, stats = result
        state = self._c1_execution_plan_table_state()
        valid = bool(
            cudagraph_mode == CUDAGraphMode.FULL
            and int(batch_desc.num_tokens) == 1
            and batch_desc.num_reqs in (None, 1)
            and not should_ubatch
            and across_dp is None
            and stats is None
            and state is not None
        )
        if not valid or state is None:
            return result
        self._c1_execution_plan = _C1ExecutionPlan(
            req_id=step.req_id,
            last_position=step.position - 1,
            table_obj_id=state[0],
            block_storage_ptr=state[1],
            slot_storage_ptr=state[2],
            seq_storage_ptr=state[3],
            determine_result=result,
        )
        return result

    def _pad_query_start_loc_for_fia(
        self,
        query_start_loc: Any,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> int:
        step = self._c1_fast_step
        exact = bool(
            self._c1_execution_plan_enabled
            and step is not None
            and self._c1_execution_plan_matches(step)
            and query_start_loc is self.query_start_loc
            and num_tokens_padded == 1
            and num_reqs_padded == 1
            and num_reqs == 1
            and cudagraph_runtime_mode == CUDAGraphMode.FULL
            and batch_desc_num_reqs in (None, 1)
        )
        if exact:
            # Fast preparation sealed [0, 1] once for this uninterrupted run.
            return 1
        if self._c1_execution_plan_enabled and step is not None:
            self._discard_c1_execution_plan("contract:fia_padding")
        return super()._pad_query_start_loc_for_fia(
            query_start_loc,
            num_tokens_padded,
            num_reqs_padded,
            num_reqs,
            cudagraph_runtime_mode,
            batch_desc_num_reqs,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: Any | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
    ) -> tuple[PerLayerAttnMetadata, Any | None]:
        step = self._c1_fast_step
        exact = bool(
            self._c1_execution_plan_enabled
            and step is not None
            and self._c1_execution_plan_matches(step)
            and num_tokens == 1
            and num_reqs == 1
            and max_query_len == 1
            and num_tokens_padded == 1
            and num_reqs_padded == 1
            and ubatch_slices is None
            and not use_spec_decode
            and not for_cudagraph_capture
            and cascade_attn_prefix_lens is None
            and num_scheduled_tokens_np is not None
            and num_scheduled_tokens_np.shape == (1,)
            and int(num_scheduled_tokens_np[0]) == 1
        )
        plan = self._c1_execution_plan if exact else None
        if plan is not None and plan.attn_metadata is not None:
            next_seq_len = step.position + 1
            for metadata in plan.unique_metadata:
                storage_ok = bool(
                    len(metadata.seq_lens_list) == 1
                    and metadata.seq_lens is not None
                    and int(metadata.seq_lens.data_ptr()) == plan.seq_storage_ptr
                    and metadata.block_tables is not None
                    and int(metadata.block_tables.data_ptr())
                    == plan.block_storage_ptr
                    and metadata.slot_mapping is not None
                    and int(metadata.slot_mapping.data_ptr())
                    == plan.slot_storage_ptr
                )
                if not storage_ok:
                    self._discard_c1_execution_plan(
                        "contract:cached_metadata_storage"
                    )
                    plan = None
                    break
                metadata.seq_lens_list[0] = next_seq_len
            if plan is not None:
                plan.last_position = step.position
                self._c1_execution_plan_hits += 1
                if (
                    self._c1_execution_plan_hits == 1
                    or self._c1_execution_plan_hits % 64 == 0
                ):
                    logger.info(
                        "MINICPMO45_STAGE1_C1_EXECUTION_PLAN event=replay "
                        "hits=%d captures=%d request_id=%s position=%d",
                        self._c1_execution_plan_hits,
                        self._c1_execution_plan_captures,
                        step.req_id,
                        step.position,
                    )
                return plan.attn_metadata, None

        if self._c1_execution_plan_enabled and step is not None and not exact:
            self._discard_c1_execution_plan("contract:attention")
        result = super()._build_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            ubatch_slices=ubatch_slices,
            logits_indices=logits_indices,
            use_spec_decode=use_spec_decode,
            for_cudagraph_capture=for_cudagraph_capture,
            num_scheduled_tokens=num_scheduled_tokens,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            cascade_attn_prefix_lens=cascade_attn_prefix_lens,
        )
        if plan is None:
            return result

        attn_metadata, spec_common = result
        if spec_common is not None or not isinstance(attn_metadata, dict):
            self._discard_c1_execution_plan("contract:metadata_container")
            return result
        unique_metadata = tuple(
            {id(metadata): metadata for metadata in attn_metadata.values()}.values()
        )
        valid = bool(unique_metadata)
        for metadata in unique_metadata:
            valid = bool(
                valid
                and isinstance(metadata, AscendMetadata)
                and metadata.attn_state == AscendAttentionState.DecodeOnly
                and metadata.num_actual_tokens == 1
                and metadata.num_decode_tokens == 1
                and metadata.num_decodes == 1
                and metadata.num_prefills == 0
                and metadata.actual_seq_lengths_q == [1]
                and isinstance(metadata.seq_lens_list, list)
                and len(metadata.seq_lens_list) == 1
                and metadata.seq_lens is not None
                and int(metadata.seq_lens.data_ptr()) == plan.seq_storage_ptr
                and metadata.block_tables is not None
                and int(metadata.block_tables.data_ptr())
                == plan.block_storage_ptr
                and metadata.slot_mapping is not None
                and int(metadata.slot_mapping.data_ptr())
                == plan.slot_storage_ptr
                and metadata.query_start_loc is not None
                and metadata.query_start_loc.numel() == 2
            )
            if not valid:
                break
        if not valid:
            self._discard_c1_execution_plan(
                "contract:metadata_shape_or_storage"
            )
            return result

        plan.attn_metadata = attn_metadata
        plan.unique_metadata = unique_metadata
        plan.last_position = step.position
        self._c1_execution_plan_captures += 1
        logger.info(
            "MINICPMO45_STAGE1_C1_EXECUTION_PLAN event=capture "
            "captures=%d request_id=%s position=%d layers=%d metadata=%d",
            self._c1_execution_plan_captures,
            step.req_id,
            step.position,
            len(attn_metadata),
            len(unique_metadata),
        )
        return result


    def _initialize_c1_fast_static_buffers(self) -> None:
        """Initialize metadata that is invariant for strict C=1 K=1 decode."""

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1] = 1
        self.query_start_loc.np[2:].fill(1)
        self.query_start_loc.copy_to_gpu()
        self.query_start_loc.gpu[2:].fill_(-1)
        self.req_indices.np[0] = 0
        self.req_indices.copy_to_gpu(1)
        self.query_pos.np[0] = 0
        self.query_pos.copy_to_gpu(1)
        self.num_scheduled_tokens.np[0] = 1
        self.num_scheduled_tokens.copy_to_gpu(1)
        self.num_accepted_tokens.np.fill(1)
        self.num_accepted_tokens.gpu.fill_(1)
        self.discard_request_mask.np[0] = False
        self.discard_request_mask.copy_to_gpu(1)
        self.optimistic_seq_lens_cpu[1:].fill_(0)
        self.seq_lens[1:].fill_(0)
        self._c1_fast_logits_indices = torch.zeros_like(
            self.query_start_loc.gpu[:1],
        )
        self._c1_fast_static_ready = True

    def _prepare_inputs_c1_decode(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[torch.Tensor, None, int]:
        """Prepare only the mutable metadata for one steady Talker token."""

        if not self._c1_fast_static_ready:
            self._initialize_c1_fast_static_buffers()
        assert self._c1_fast_logits_indices is not None

        req_id = self.input_batch.req_ids[0]
        position = int(self.input_batch.num_computed_tokens_cpu[0])
        table_group = self.input_batch.block_table
        table_group.commit_block_table(1)

        # Reuse the copy that stock input preparation already performs.
        # The ordinary runner token is the binary continue/stop controller;
        # the graph-embedding lane instead needs the request-local codec id.
        if self._codec_embed_graph_enabled:
            info = self.model_intermediate_buffer.get(req_id)
            current = (info.get("audio_codes", {}) or {}).get("current")
            if (
                not isinstance(current, torch.Tensor)
                or current.shape != (1,)
                or current.dtype != torch.long
                or current.device != self.device
            ):
                raise RuntimeError(
                    "MiniCPM-o codec-embedding graph lost the authoritative codec token"
                )
            self.input_ids.gpu[:1].copy_(current, non_blocking=True)
        else:
            # Async scheduling keeps the authoritative runner sample on
            # device; token_ids_cpu is a placeholder until scheduler commit.
            self.input_ids.gpu[:1].copy_(
                self.input_batch.prev_sampled_token_ids[:1, 0],
                non_blocking=True,
            )

        self.optimistic_seq_lens_cpu[0] = position + 1
        computed_cpu = self.input_batch.num_computed_tokens_cpu_tensor[:1]
        self.num_computed_tokens[:1].copy_(computed_cpu, non_blocking=True)
        self.positions[:1].copy_(computed_cpu, non_blocking=True)
        self.seq_lens[:1].copy_(
            self.optimistic_seq_lens_cpu[:1],
            non_blocking=True,
        )

        # Keep the stock paged-KV writer and its 4-D cache contract.  For the
        # strict C=1 lane the authoritative CPU block-table row already names
        # the physical block, so resolve the one scalar slot without launching
        # the general NPU slot-mapping kernel.  Any inconsistent table state
        # falls back to the established implementation.
        table = table_group.block_tables[0]
        slot = (
            _c1_cpu_slot_from_table(table, position)
            if self._c1_cpu_slot_enabled
            else None
        )
        if slot is None:
            table_group.compute_slot_mapping(
                1,
                self.query_start_loc.gpu[:2],
                self.positions[:1],
            )
            if self._c1_cpu_slot_enabled:
                self._c1_cpu_slot_fallbacks["invalid_block_index"] += 1
        else:
            table.slot_mapping.np[0] = slot
            table.slot_mapping.copy_to_gpu(1)
            self._c1_cpu_slot_hits += 1
            if position > 0 and position % int(table.physical_block_size) == 0:
                self._c1_cpu_slot_boundaries += 1
            if self._c1_cpu_slot_hits == 1 or self._c1_cpu_slot_hits % 512 == 0:
                logger.info(
                    "MiniCPMO45Stage1C1FastCPUSlot hits=%d boundaries=%d "
                    "fallbacks=%s position=%d slot=%d",
                    self._c1_cpu_slot_hits,
                    self._c1_cpu_slot_boundaries,
                    dict(self._c1_cpu_slot_fallbacks),
                    position,
                    slot,
                )
        self.num_discarded_requests = 0
        self.query_lens = self._c1_fast_query_lens
        self.attn_state = AscendAttentionState.DecodeOnly
        self.with_prefill = False
        self.logits_indices = self._c1_fast_logits_indices
        self._c1_fast_step = _C1DecodeFastStep(req_id=req_id, position=position)
        self._c1_fast_hits += 1
        if self._c1_fast_hits == 1 or self._c1_fast_hits % 512 == 0:
            logger.info(
                "MINICPMO45_STAGE1_C1_FAST_PREP hits=%d rejects=%s "
                "request_id=%s position=%d",
                self._c1_fast_hits,
                dict(self._c1_fast_rejects),
                req_id,
                position,
            )
        return self._c1_fast_logits_indices, None, 1

    def _c1_cpu_slot_reject_reason(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> str | None:
        if not self._c1_cpu_slot_enabled:
            return "disabled"
        if int(self.input_batch.num_reqs) != 1:
            return "not_c1"
        if int(scheduler_output.total_num_scheduled_tokens) != 1:
            return "not_one_token"
        if len(num_scheduled_tokens) != 1 or int(num_scheduled_tokens[0]) != 1:
            return "not_one_token"
        if scheduler_output.scheduled_spec_decode_tokens:
            return "scheduled_spec_decode"
        if self.speculative_config is not None:
            return "speculative_config"
        if getattr(self, "use_compress", False):
            return "compressed_kv"
        if int(self.pcp_size) != 1:
            return "pcp"

        position = int(self.input_batch.num_computed_tokens_cpu[0])
        prompt_tokens = int(self.input_batch.num_prompt_tokens[0])
        if position < prompt_tokens:
            return "prefill"

        block_tables = self.input_batch.block_table.block_tables
        if len(block_tables) != 1:
            return "multiple_kv_groups"
        table = block_tables[0]
        if table.is_mamba_group:
            return "mamba"
        if int(table.pcp_world_size) != 1 or int(table.dcp_world_size) != 1:
            return "context_parallel"
        if _c1_cpu_slot_from_table(table, position) is None:
            return "invalid_block_index"
        return None

    @_stage1_host_ledger("prepare_inputs")
    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, int]:
        """Bypass the general NPU slot mapper for strict C=1 Talker decode.

        The temporary wrapper is visible only while the upstream Ascend input
        preparation runs.  Every rejected mode and every runtime inconsistency
        calls the original mapper with the original arguments.  The instance
        method is restored even if upstream preparation raises.
        """

        self._c1_fast_step = None
        self._c1_host_fast_info = None
        try:
            fast_reject = self._c1_fast_reject_reason(
                scheduler_output,
                num_scheduled_tokens,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            fast_reject = "invalid_metadata"
        if fast_reject is None:
            return self._prepare_inputs_c1_decode(scheduler_output)
        if self._c1_fast_prep_enabled:
            # The generic path is allowed to mutate every reusable metadata
            # buffer (for example a 25-token prefill writes
            # query_start_loc=[0, 25]). Static here means static only within
            # one uninterrupted C1 decode run, never process-lifetime static.
            # Force the next eligible step to reseal those buffers.
            self._c1_fast_static_ready = False
            self._discard_c1_execution_plan(f"boundary:{fast_reject}")
            self._c1_fast_rejects[fast_reject] += 1
            rejects = sum(self._c1_fast_rejects.values())
            if rejects == 1 or rejects % 512 == 0:
                logger.info(
                    "MINICPMO45_STAGE1_C1_FAST_PREP event=reject "
                    "rejects=%s",
                    dict(self._c1_fast_rejects),
                )

        if not self._c1_cpu_slot_enabled:
            return super()._prepare_inputs(scheduler_output, num_scheduled_tokens)

        multi_table = self.input_batch.block_table
        had_instance_compute = "compute_slot_mapping" in multi_table.__dict__
        old_instance_compute = multi_table.__dict__.get("compute_slot_mapping")
        stock_compute = multi_table.compute_slot_mapping
        try:
            reject_reason = self._c1_cpu_slot_reject_reason(
                scheduler_output,
                num_scheduled_tokens,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            reject_reason = "invalid_metadata"

        def compute_slot_mapping(
            num_reqs: int,
            query_start_loc: torch.Tensor,
            positions: torch.Tensor,
            positions_compressed_list: list[np.ndarray] | None = None,
            req_indices_compressed_list: list[np.ndarray] | None = None,
        ) -> None:
            dynamic_reject = reject_reason
            if (
                positions_compressed_list is not None
                or req_indices_compressed_list is not None
            ):
                dynamic_reject = "compressed_kv"
            if dynamic_reject is not None:
                self._c1_cpu_slot_fallbacks[dynamic_reject] += 1
                stock_compute(
                    num_reqs,
                    query_start_loc,
                    positions,
                    positions_compressed_list,
                    req_indices_compressed_list,
                )
                return

            table = multi_table.block_tables[0]
            position = int(self.input_batch.num_computed_tokens_cpu[0])
            slot = _c1_cpu_slot_from_table(table, position)
            if slot is None:
                self._c1_cpu_slot_fallbacks["invalid_block_index"] += 1
                stock_compute(
                    num_reqs,
                    query_start_loc,
                    positions,
                    positions_compressed_list,
                    req_indices_compressed_list,
                )
                return

            table.slot_mapping.np[0] = slot
            table.slot_mapping.copy_to_gpu(1)
            self._c1_cpu_slot_hits += 1
            if position > 0 and position % int(table.physical_block_size) == 0:
                self._c1_cpu_slot_boundaries += 1
            if self._c1_cpu_slot_hits == 1 or self._c1_cpu_slot_hits % 512 == 0:
                logger.info(
                    "MiniCPMO45Stage1CPUSlotFastPath hits=%d boundaries=%d "
                    "fallbacks=%s position=%d slot=%d",
                    self._c1_cpu_slot_hits,
                    self._c1_cpu_slot_boundaries,
                    dict(self._c1_cpu_slot_fallbacks),
                    position,
                    slot,
                )

        multi_table.compute_slot_mapping = compute_slot_mapping
        try:
            return super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        finally:
            if had_instance_compute:
                multi_table.__dict__["compute_slot_mapping"] = old_instance_compute
            else:
                del multi_table.__dict__["compute_slot_mapping"]

    def _preprocess(
        self,
        scheduler_output: SchedulerOutput,
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        """Use a sealed MiniCPM Talker decode embedding lane when prepared."""

        step = self._c1_fast_step
        self._c1_fast_step = None
        set_codec_embed_active = getattr(
            self.model, "set_codec_embed_graph_active", None
        )
        if step is None:
            if self._codec_embed_graph_enabled:
                if not callable(set_codec_embed_active):
                    raise RuntimeError(
                        "MiniCPM-o codec-embedding graph lost its model selector"
                    )
                set_codec_embed_active(False)
            return super()._preprocess(
                scheduler_output,
                num_input_tokens,
                intermediate_tensors,
            )
        if (
            num_input_tokens != 1
            or intermediate_tensors is not None
            or self.input_batch.req_ids[0] != step.req_id
        ):
            raise RuntimeError(
                "MiniCPM-o C1 fast preprocess contract changed after input preparation"
            )

        # Preserve dynamic request metadata refreshes; the expensive generic
        # modality/request loop is unnecessary for one established decode row.
        self._update_additional_information(scheduler_output)
        if not self.vllm_config.model_config.async_chunk:
            self._sync_local_stage_payloads()
        info = self.model_intermediate_buffer.get(step.req_id)
        preprocess = getattr(self.model, "preprocess", None)
        if not isinstance(info, dict) or not callable(preprocess):
            raise RuntimeError(
                "MiniCPM-o C1 fast preprocess eligibility changed within one step"
            )

        # Match the request-local metadata refresh performed by the generic
        # Omni preprocessing loop.  These values are part of the model
        # preprocess contract, not optional diagnostics: without
        # ``_omni_is_prefill=False`` an established Talker decode row is
        # interpreted as a first/prefill call and rebuilds its audio state.
        req_state = self.requests.get(step.req_id)
        if req_state is None:
            raise RuntimeError(
                "MiniCPM-o C1 fast preprocess lost its request state"
            )
        prompt_token_ids = getattr(req_state, "prompt_token_ids", ())
        prompt_len = len(prompt_token_ids or ())
        info["request_id"] = step.req_id
        info["duplex_token_offset"] = step.position
        info["duplex_prompt_len"] = prompt_len
        info["_omni_prompt_len"] = prompt_len
        info["_omni_num_computed_tokens"] = step.position
        info["_omni_is_prefill"] = False
        if self._codec_embed_graph_enabled:
            # Keep the same tensor/tensor model-call signature used by
            # startup FULL_DECODE capture.  The model-owned selector
            # makes the graph consume ``input_ids`` and ignore this stale
            # embedding row only for the strict C=1 codec lane.
            if not callable(set_codec_embed_active):
                raise RuntimeError(
                    "MiniCPM-o codec-embedding graph lost its model selector"
                )
            set_codec_embed_active(True)
            self._omni_num_scheduled_tokens_np = self._c1_fast_one_np
            self._codec_embed_graph_hits += 1
            if (
                self._codec_embed_graph_hits == 1
                or self._codec_embed_graph_hits % 512 == 0
            ):
                logger.info(
                    "MINICPMO45_STAGE1_CODEC_EMBED_GRAPH event=use hits=%d "
                    "request_id=%s position=%d signature=tensor_tensor "
                    "input_ids_ptr=%d inputs_embeds_ptr=%d",
                    self._codec_embed_graph_hits,
                    step.req_id,
                    step.position,
                    int(self.input_ids.gpu.data_ptr()),
                    int(self.inputs_embeds.gpu.data_ptr()),
                )
            return (
                self.input_ids.gpu[:1],
                self.inputs_embeds.gpu[:1],
                self.positions[:1],
                None,
                self._init_model_kwargs(),
                None,
            )

        out = self.inputs_embeds.gpu[:1]
        preprocess_into = getattr(self.model, "preprocess_c1_decode_into", None)
        used_host_fast = bool(
            self._c1_host_fast_enabled
            and callable(preprocess_into)
            and preprocess_into(info, out)
        )
        if used_host_fast:
            req_input_ids = self.input_ids.gpu[:1]
            self._c1_host_fast_info = info
            self._c1_host_fast_preprocess_hits += 1
            if (
                self._c1_host_fast_preprocess_hits == 1
                or self._c1_host_fast_preprocess_hits % 512 == 0
            ):
                logger.info(
                    "MINICPMO45_STAGE1_C1_HOST_FAST event=preprocess "
                    "hits=%d request_id=%s position=%d",
                    self._c1_host_fast_preprocess_hits,
                    step.req_id,
                    step.position,
                )
        else:
            req_input_ids, req_embeds, update_dict = preprocess(
                input_ids=self.input_ids.gpu[:1],
                input_embeds=None,
                **info,
            )
            if update_dict:
                raise RuntimeError(
                    "MiniCPM-o C1 decode unexpectedly produced intermediate updates: "
                    f"{tuple(sorted(update_dict))}"
                )
            out.copy_(req_embeds)

        self._omni_num_scheduled_tokens_np = self._c1_fast_one_np
        return (
            req_input_ids,
            out,
            self.positions[:1],
            None,
            self._init_model_kwargs(),
            None,
        )

    def _make_buffer(self, *size, dtype, numpy=True):
        # Prevent ray from pinning the buffer due to large size
        from vllm_omni.distributed.ray_utils.utils import (
            calculate_total_bytes,
            maybe_disable_pin_memory_for_ray,
        )

        total_bytes = calculate_total_bytes(size, dtype)

        # Use the context manager to temporarily disable pinning if needed
        with maybe_disable_pin_memory_for_ray(self, total_bytes):
            return super()._make_buffer(*size, dtype=dtype, numpy=numpy)

    @_stage1_host_ledger("build_model_kwargs")
    def _build_model_kwargs_extra(self) -> dict:
        known_controller_bypass = self._can_use_known_controller_bypass()
        info = self._c1_host_fast_info
        if self._c1_host_fast_enabled and info is not None:
            req_id = self.input_batch.req_ids[0]
            req_state = self.requests.get(req_id)
            if req_state is None or self.model_intermediate_buffer.get(req_id) is not info:
                raise RuntimeError(
                    "MiniCPM-o C1 host fast kwargs lost its request metadata"
                )
            info["generated_len"] = len(req_state.output_token_ids)
            self._c1_host_fast_infos[0] = info
            self._c1_host_fast_kwargs_hits += 1
            if (
                self._c1_host_fast_kwargs_hits == 1
                or self._c1_host_fast_kwargs_hits % 512 == 0
            ):
                logger.info(
                    "MINICPMO45_STAGE1_C1_HOST_FAST event=model_kwargs "
                    "hits=%d request_id=%s",
                    self._c1_host_fast_kwargs_hits,
                    req_id,
                )
            return {
                "model_intermediate_buffer": self._c1_host_fast_infos,
                "runtime_additional_information": self._c1_host_fast_infos,
                "request_token_spans": self._c1_host_fast_spans,
                "request_sample_eligible": self._c1_host_fast_sample_eligible,
                "defer_codec_commit": False,
                "known_controller_bypass": known_controller_bypass,
            }
        model_kwargs_extra = super()._build_model_kwargs_extra()
        model_kwargs_extra["known_controller_bypass"] = known_controller_bypass
        # K=1 MiniCPM Talker fast path: postpone only the request-visible codec
        # commit until _bookkeeping_sync has performed the normal sampled-token
        # D2H.  Keep every less constrained mode on the established immediate
        # path: async scheduling has no CPU tokens here, speculative decoding
        # can return multiple tokens, prefix caching consumes multimodal output
        # before bookkeeping, and discarded rows intentionally do not have a
        # valid runner sample to commit against.
        can_defer_codec_commit = (
            getattr(self.model, "model_stage", None) == "tts"
            and callable(getattr(self.model, "finalize_deferred_codec_output", None))
            and not self.use_async_scheduling
            and self.speculative_config is None
            and self.omni_prefix_cache is None
            and int(getattr(self, "num_discarded_requests", 0)) == 0
            and not getattr(self.vllm_config.model_config, "logits_processors", None)
            and get_pp_group().world_size == 1
        )
        can_defer_request = False
        if can_defer_codec_commit and int(getattr(self.input_batch, "num_reqs", 0)) == 1:
            req_id = self.input_batch.req_ids[0]
            req_state = self.requests.get(req_id)
            sampling_params = getattr(req_state, "sampling_params", None)
            model_eligibility = getattr(self.model, "can_defer_codec_commit", None)
            can_defer_request = bool(
                callable(model_eligibility)
                and model_eligibility(req_id, sampling_params)
            )
        model_kwargs_extra["defer_codec_commit"] = (
            can_defer_codec_commit and can_defer_request
        )
        return model_kwargs_extra

    def _can_use_known_controller_bypass(self) -> bool:
        """Gate the exact C=1 async non-speculative controller sideband."""

        return bool(
            self._known_controller_bypass_enabled
            and self.use_async_scheduling
            and getattr(self.model, "model_stage", None) == "tts"
            and int(getattr(self.input_batch, "num_reqs", 0)) == 1
            and self.speculative_config is None
            and self.omni_prefix_cache is None
            and int(getattr(self, "num_discarded_requests", 0)) == 0
            and not getattr(
                self.vllm_config.model_config,
                "logits_processors",
                None,
            )
            and get_pp_group().world_size == 1
        )

    def _known_controller_sampler_output(
        self,
        logits: torch.Tensor,
        token: int,
    ) -> SamplerOutput:
        key = (token, logits.device)
        sampled = self._known_controller_device_tokens.get(key)
        if sampled is None:
            sampled = torch.full(
                (1, 1),
                token,
                dtype=torch.int32,
                device=logits.device,
            )
            self._known_controller_device_tokens[key] = sampled
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)

    #  -------------------------------------- Omni-new -------------------------------------------------
    def capture_model(self) -> int:
        npugraph_memory_bytes = super().capture_model()
        self._capture_talker_mtp_graphs()
        return npugraph_memory_bytes

    def _capture_talker_mtp_graphs(self) -> None:
        if not self.has_talker_mtp or not isinstance(self.talker_mtp, ACLGraphWrapper):
            return

        from vllm.compilation.monitor import set_cudagraph_capturing_enabled

        capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes, reverse=True)
        num_warmups = self.compilation_config.cudagraph_num_of_warmups
        logger.info("Capturing talker_mtp graphs for sizes %s", capture_sizes)

        set_cudagraph_capturing_enabled(True)
        try:
            with torch.inference_mode(), graph_capture(device=self.device):
                for bsz in capture_sizes:
                    _, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
                        num_tokens=bsz,
                        num_reqs=bsz,
                        num_scheduled_tokens_np=np.ones(bsz, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                    )
                    n = batch_desc.num_tokens
                    ids = self.talker_mtp_input_ids.gpu[:n]
                    emb = self.talker_mtp_inputs_embeds.gpu[:n]
                    hid = self.last_talker_hidden.gpu[:n]
                    ts = self.text_step.gpu[:n]

                    for _ in range(num_warmups):
                        with set_ascend_forward_context(
                            None,
                            self.vllm_config,
                            aclgraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=batch_desc,
                        ):
                            self.talker_mtp(ids, emb, hid, ts)

                    with set_ascend_forward_context(
                        None,
                        self.vllm_config,
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=batch_desc,
                    ):
                        self.talker_mtp(ids, emb, hid, ts)
                    torch.npu.synchronize()

            logger.info("Captured talker_mtp graphs for %d sizes", len(capture_sizes))
        except RuntimeError as e:
            raise RuntimeError(
                f"talker_mtp graph capture failed for a model that declared talker_mtp_graph_safe=True: {e}"
            ) from e
        finally:
            set_cudagraph_capturing_enabled(False)

    def _model_needs_full_prefix_hidden_states(self) -> bool:
        """See gpu_ar_model_runner._model_needs_full_prefix_hidden_states."""
        model = getattr(self, "model", None)
        return bool(getattr(model, "requires_full_prefix_cached_hidden_states", True))

    def _maybe_update_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ):
        if self.omni_prefix_cache is not None and get_pp_group().is_last_rank:
            if multimodal_outputs is not None and not isinstance(multimodal_outputs, Mapping):
                logger.warning_once(
                    "prefix caching expects mm outputs to be a dict, but got %s",
                    type(multimodal_outputs),
                )

            hs_for_cache = hidden_states if self._model_needs_full_prefix_hidden_states() else None
            self.omni_prefix_cache.update_omni_tensor_prefix_cache(
                hidden_states=hs_for_cache,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                slot_mapping=self.input_batch.block_table[0].slot_mapping.cpu,
                num_tokens_padded=num_tokens_padded,
            )

    def _maybe_get_combined_prefix_cache_tensors(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ) -> tuple[dict[str, torch.Tensor] | None, dict | None]:
        combined_hidden_states, combined_multimodal_outputs = None, None
        if self.omni_prefix_cache is not None:
            if self._model_needs_full_prefix_hidden_states():
                combined_hidden_states = self.omni_prefix_cache.get_merged_hidden_states(
                    query_start_loc=self.query_start_loc.cpu,
                    input_batch=self.input_batch,
                    hidden_states=hidden_states,
                    num_scheduled_tokens=num_scheduled_tokens,
                )
            combined_multimodal_outputs = self.omni_prefix_cache.get_merged_multimodal_states(
                query_start_loc=self.query_start_loc.cpu,
                input_batch=self.input_batch,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_scheduled_tokens=num_scheduled_tokens,
            )
        return combined_hidden_states, combined_multimodal_outputs

    @staticmethod
    def _resolve_req_hidden_states(
        hidden_states_cpu: torch.Tensor,
        combined_hidden_states: dict[str, torch.Tensor] | None,
        rid: str,
        start: int,
        end: int,
    ):
        if combined_hidden_states is not None:
            if rid not in combined_hidden_states:
                raise RuntimeError("Request IDs in the batch are missing from the merged states!")
            return combined_hidden_states[rid]
        return hidden_states_cpu[start:end]


    def _build_multimodal_outputs(
        self,
        per_req_payloads: list[dict[str, object] | None] | None,
    ) -> list[dict[str, torch.Tensor] | None] | None:
        if self.vllm_config.model_config.engine_output_type == "text":
            return None
        if per_req_payloads is None:
            return None
        wire_payloads: list[dict[str, torch.Tensor] | None] = []
        for payload in per_req_payloads:
            if not payload:
                wire_payloads.append(None)
            else:
                wire_payloads.append(_ensure_tensor_values(payload))
        if all(item is None for item in wire_payloads):
            return None
        return wire_payloads


    def _request_final_stage_id(self, req_id: str) -> int | None:
        info = self.model_intermediate_buffer.get(req_id)
        if not isinstance(info, dict):
            req_state = self.requests.get(req_id)
            info = getattr(req_state, "additional_information_cpu", None)
        if not isinstance(info, dict):
            return None
        val = info.get("omni_final_stage_id")
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _request_needs_downstream_stage_payload(self, req_id: str) -> bool:
        cached = self._downstream_payload_cache.get(req_id)
        if cached is not None:
            return cached
        final_stage_id = self._request_final_stage_id(req_id)
        needs_payload = final_stage_id is None or final_stage_id > 0
        self._downstream_payload_cache[req_id] = needs_payload
        return needs_payload

    def _resolve_pooler_payload_req_ids(self, req_ids_output_copy: list[str]) -> tuple[str, list[str]]:
        downstream_req_ids = [rid for rid in req_ids_output_copy if self._request_needs_downstream_stage_payload(rid)]
        engine_output_type = (self.vllm_config.model_config.engine_output_type or "").lower()
        # Single-stage AR TTS models (e.g. VoxCPM2) finish on this stage but still
        # need multimodal payloads for final audio postprocess/output.
        if engine_output_type == "audio" and not downstream_req_ids:
            downstream_req_ids = req_ids_output_copy
        return engine_output_type, downstream_req_ids

    @staticmethod
    def _sparse_mm_req_ids(multimodal_outputs: Any) -> list[str] | None:
        if not isinstance(multimodal_outputs, dict):
            return None
        meta = multimodal_outputs.get("meta")
        req_ids = None
        sparse_audio = False
        if isinstance(meta, dict):
            req_ids = meta.get("req_id")
            sparse_audio = NPUARModelRunner._is_sparse_audio_marker(meta.get("sparse_audio"))
        if req_ids is None:
            req_ids = multimodal_outputs.get("meta.req_id")
            sparse_audio = NPUARModelRunner._is_sparse_audio_marker(multimodal_outputs.get("meta.sparse_audio"))
        if not sparse_audio:
            return None
        if not isinstance(req_ids, list):
            return None
        return [rid for rid in req_ids if isinstance(rid, str)]

    @staticmethod
    def _is_sparse_audio_marker(value: Any) -> bool:
        if isinstance(value, list):
            return any(str(item).lower() in ("1", "true", "yes", "on") for item in value)
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    #  -------------------------------------- Omni-new -------------------------------------------------

    @torch.inference_mode()
    @_stage1_host_ledger("execute_model")
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | IntermediateTensors | None:
        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = self.routed_experts_capturer
            if capturer is not None and hasattr(capturer, "finalize_pending_copy"):
                capturer.finalize_pending_copy()
        if self.ascend_config.profiling_chunk_config.enabled:
            self._sync_device()
            self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        #  -------------------------------------- Omni-new -------------------------------------------------
        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        if not getattr(self, "_warmup_state_cleared", False):
            self._warmup_state_cleared = True
            if hasattr(self.model, "_clear_warmup_state"):
                self.model._clear_warmup_state()

        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        finished_reqs = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if finished_reqs and hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished_reqs.items():
                try:
                    req_idx = self.input_batch.req_id_to_index.get(req_id)
                    num_computed = (
                        int(self.input_batch.num_computed_tokens_cpu[req_idx]) if req_idx is not None else None
                    )
                    model_meta = self.model.get_kv_transfer_metadata(
                        req_id,
                        num_computed_tokens=num_computed,
                    )
                    if model_meta:
                        existing = data.get("custom_metadata") or {}
                        existing.update(model_meta)
                        data["custom_metadata"] = existing
                except Exception as e:
                    logger.warning(f"Failed to get custom metadata from model for {req_id}: {e}")
        self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished_reqs,
            kv_caches=self.kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
            request_id_resolver=self._resolve_global_request_id,
        )
        #  -------------------------------------- Omni-new -------------------------------------------------
        if hasattr(self, "_omni_connector"):
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
            if self._pending_full_payload_send:
                flush_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
                flush_ids.update({rid for rid in self._pending_full_payload_send if rid not in self.requests})
                if flush_ids:
                    self.flush_full_payload_outputs(flush_ids)
        # self._draft_token_ids is None when `input_fits_in_drafter=False`
        # and there is no draft tokens scheduled. so it need to update the
        # spec_decoding info in scheduler_output with async_scheduling.
        # use deepcopy to avoid the modification has influence on the
        # scheduler_output in engine core process.
        # TODO(Ronald1995): deepcopy is expensive when there is a large
        # number of requests, optimize it later.
        if ((
            self.use_async_scheduling
            and self.num_spec_tokens
            and self._draft_token_ids is None  # type: ignore[has-type]
        ) or (
            # NOTE: This branch specifically triggers a deepcopy during the prefill phase
            # only for PCP (Parallel Context Processing) + Multi-Modal (MM) scenarios.
            # It does not affect other use cases. This is a temporary workaround and
            # will be removed once upstream vLLM provides native support for PCP + MM.
            self.pcp_size > 1
            and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        )):
            scheduler_output = deepcopy(scheduler_output)

        #  -------------------------------------- Omni-new -------------------------------------------------
        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            if kv_connector_metadata is not None:
                get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
        #  -------------------------------------- Omni-new -------------------------------------------------

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                deferred_state_corrections_fn = self._update_states(scheduler_output)

                #  -------------------------------------- Omni-new -------------------------------------------------
                if scheduler_output.finished_req_ids and hasattr(self.model, "on_requests_finished"):
                    self.model.on_requests_finished(scheduler_output.finished_req_ids)
                #  -------------------------------------- Omni-new -------------------------------------------------

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)

                        kv_ids = self.kv_extracted_req_ids
                        self.kv_extracted_req_ids = None

                        output = make_empty_encoder_model_runner_output(scheduler_output)
                        if kv_ids:
                            output = copy(output)
                            output.kv_extracted_req_ids = kv_ids
                        return self.attach_omni_connector_output(output)

                # `<= 0`: upstream can schedule a negative span, which is truthy (#5196).
                if num_scheduled_tokens <= 0:
                    if (
                        self.parallel_config.distributed_executor_backend == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        # this is a corner case when both external launcher
                        # and DP are enabled, num_scheduled_tokens could be
                        # 0, and has_unfinished_requests in the outer loop
                        # returns True. before returning early here we call
                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(1)

                    kv_ids = self.kv_extracted_req_ids
                    self.kv_extracted_req_ids = None

                    if not has_kv_transfer_group():
                        output = EMPTY_MODEL_RUNNER_OUTPUT
                    else:
                        output = self.kv_connector_no_forward(scheduler_output, self.vllm_config)

                    if kv_ids:
                        output = copy(output)
                        output.kv_extracted_req_ids = kv_ids

                    return self.attach_omni_connector_output(output)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                (
                    logits_indices,
                    spec_decode_metadata,
                    total_num_scheduled_tokens,
                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
                if self.pcp_size > 1:
                    num_tokens_unpadded = self.pcp_manager.total_num_sampled_tokens_pcp
                cascade_attn_prefix_lens = None
                # Disable cascade attention when using microbatching (DBO)
                if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
                    # Pre-compute cascade attention prefix lengths
                    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                        num_scheduled_tokens_np,
                        self.input_batch.num_computed_tokens_cpu[:num_reqs],
                        scheduler_output.num_common_prefix_blocks,
                    )

                (
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                    cudagraph_stats,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=cascade_attn_prefix_lens is not None,
                    force_eager=self.model_config.enforce_eager,
                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )

                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

                num_tokens_padded = batch_desc.num_tokens
                num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
                ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                    should_ubatch,
                    num_scheduled_tokens_np,
                    num_tokens_padded,
                    num_reqs_padded,
                    self.parallel_config.num_ubatches,
                )

                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                # NOTE(Angazenn): According to https://github.com/vllm-project/vllm/pull/30877,
                # there should be a corresponding 'postprocess_mamba'. However, it is called inside
                # '_update_states_after_model_execute', which is not overridden in vLLM-Ascend.
                # We simply utilize the implementation in vLLM.
                if self.cache_config.mamba_cache_mode == "align":
                    # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                    # to decide copy operations, so we must apply deferred
                    # corrections before it runs.
                    if deferred_state_corrections_fn:
                        deferred_state_corrections_fn()
                        deferred_state_corrections_fn = None
                    preprocess_mamba(
                        scheduler_output,
                        self.kv_cache_config,
                        self.cache_config,
                        self.mamba_state_idx,
                        self.input_batch,
                        self.requests,
                        self.compilation_config.static_forward_context,
                        self.model.get_mamba_state_copy_func(),
                        self._get_mamba_copy_bufs(),
                    )
                    # preprocess_mamba resets num_accepted_tokens_cpu to 1
                    # for requests whose state was copied to a new block.
                    # Re-sync to GPU so the mamba kernel reads from the
                    # correct initial state slot (init_token_idx = 0).
                    self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                    self.num_accepted_tokens.copy_to_gpu(num_reqs)

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

                if (
                    cudagraph_mode == CUDAGraphMode.FULL
                    or (enable_sp() and not self.model_config.use_mla)
                    and self.pcp_size * self.dcp_size == 1
                ):
                    # Currently, Graph Mode and SP will both pad num_tokens,
                    # Another possible condition is num_tokens_padded != num_tokens_unpadded
                    # but this scope is way too big and the consequences are unpredictable
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        self.query_start_loc,
                        num_tokens_padded,
                        num_reqs_padded,
                        num_reqs,
                        cudagraph_mode,
                        batch_desc.num_reqs,
                    )

                (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded
                    if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                    else total_num_scheduled_tokens,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded
                if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                else total_num_scheduled_tokens,
                intermediate_tensors,
            )

            #  -------------------------------------- Omni-new -------------------------------------------------
            if hasattr(self.model, "prepare_runner_inputs"):
                input_ids, positions = self.model.prepare_runner_inputs(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    req_ids=req_ids[:num_reqs],
                    num_computed_tokens=self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    num_scheduled_tokens=num_scheduled_tokens_np[:num_reqs],
                    input_ids_buffer=self.input_ids.gpu[:num_tokens_padded],
                )
            #  -------------------------------------- Omni-new -------------------------------------------------

            # update global cos, sin
            update_cos_sin(positions)

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB weight D2D"):
                self.eplb_updator.forward_before()

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:  # type: ignore[has-type]
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False  # type: ignore[has-type]
        # prevent debugger is None
        if self.debugger is not None:
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(getattr(dbg_cfg, "level", "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()
        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # Run forward pass
        clear_kv_metadata = self.speculative_config is None
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                model_instance=self.model,
                max_tokens_across_pcp=0 if self.pcp_size == 1 else self.pcp_manager.max_num_tokens_across_pcp,
                skip_compiled=has_encoder_input,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **(
                    {"defer_finalize": not clear_kv_metadata}
                ),
            ) as kv_connector_output,
        ):
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )
        with record_function_or_nullcontext("post process"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            # [Omni] Map pending ropes metadata to req_ids.
            flush_pending_metadata = getattr(self.model, "flush_pending_metadata", None)
            if callable(flush_pending_metadata):
                flush_pending_metadata(req_ids[:num_reqs])

            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)

            if multimodal_outputs is not None:
                keys_or_type = (
                    list(multimodal_outputs.keys())
                    if isinstance(multimodal_outputs, Mapping)
                    else type(multimodal_outputs)
                )
                logger.debug(f"[AR] execute_model: multimodal_outputs keys = {keys_or_type}")
            else:
                logger.debug("[AR] execute_model: multimodal_outputs is None")
            #  -------------------------------------- Omni-new -------------------------------------------------
            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states
            if self.pcp_size > 1:
                # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
                # ignores the padding from CUDA Graph.
                hidden_states = self.pcp_manager.get_restore_hidden_states(hidden_states)
                if aux_hidden_states is not None:
                    aux_hidden_states = [
                        self.pcp_manager.get_restore_hidden_states(aux_hidden_states_pcp)
                        for aux_hidden_states_pcp in aux_hidden_states
                    ]

            #  -------------------------------------- Omni-new -------------------------------------------------
            self._maybe_update_prefix_cache(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
            )
            #  -------------------------------------- Omni-new -------------------------------------------------

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                    )
                    output.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                #  -------------------------------------- Omni-new -------------------------------------------------
                # Try with sampling_metadata first; fall back to without for models that don't support it
                try:
                    logits = self.model.compute_logits(
                        sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                    )
                except TypeError:
                    logits = self.model.compute_logits(sample_hidden_states)
                #  -------------------------------------- Omni-new -------------------------------------------------
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    sample_hidden_states = hidden_states[logits_indices]
                    get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    #  -------------------------------------- Omni-new -------------------------------------------------
                    # Try with sampling_metadata first; fall back to without for models that don't support it
                    try:
                        logits = self.model.compute_logits(
                            sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                        )
                    except TypeError:
                        logits = self.model.compute_logits(sample_hidden_states)
                    #  -------------------------------------- Omni-new -------------------------------------------------

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()
                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
                ec_connector_output,
                cudagraph_stats,
                batch_desc,
                multimodal_outputs, # Omni-specific
            )
            self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        if self.vllm_config.model_config.enable_return_routed_experts and hasattr(self, "_positions_cpu"):
            self._omni_routed_experts_d2h(scheduler_output)

        return None

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: Any,
    ):
        self._known_controller_token_for_output = None
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            model_sample = getattr(self.model, "sample", None)
            self.input_batch.update_async_output_token_ids()
            take_known_controller = getattr(
                self.model,
                "take_known_controller_token",
                None,
            )
            if (
                logits is not None
                and self._can_use_known_controller_bypass()
                and callable(take_known_controller)
            ):
                req_id = self.input_batch.req_ids[0]
                req_state = self.requests.get(req_id)
                sampling_params = getattr(req_state, "sampling_params", None)
                known_token = take_known_controller(req_id, sampling_params)
                if known_token is not None:
                    self._known_controller_token_for_output = int(known_token)
                    self._known_controller_bypass_hits += 1
                    if (
                        self._known_controller_bypass_hits == 1
                        or self._known_controller_bypass_hits % 512 == 0
                    ):
                        logger.info(
                            "MINICPMO45_KNOWN_CONTROLLER event=bypass "
                            "hits=%d request_id=%s token=%d",
                            self._known_controller_bypass_hits,
                            req_id,
                            known_token,
                        )
                    return self._known_controller_sampler_output(
                        logits,
                        int(known_token),
                    )
            if logits is not None and callable(model_sample) and getattr(self.model, "prefer_model_sampler", False):
                # Apply logit bias (min_tokens, allowed_token_ids) before
                # the custom model sampler — the standard GPU sampler does
                # this internally, but prefer_model_sampler bypasses it.
                if hasattr(self.sampler, "logit_bias_state"):
                    self.sampler.logit_bias_state.apply_logit_bias(
                        logits,
                        self.input_batch.expanded_idx_mapping,
                        self.input_batch.idx_mapping_np,
                        self.input_batch.positions[self.input_batch.logits_indices],
                    )
                can_skip_history = getattr(
                    self.model,
                    "can_skip_model_sampler_output_token_history",
                    None,
                )
                skip_history = bool(
                    self._c1_host_fast_enabled
                    and callable(can_skip_history)
                    and can_skip_history(sampling_metadata)
                )
                if skip_history:
                    prepared_sampling_metadata = sampling_metadata
                    self._c1_host_fast_sampler_history_skips += 1
                    if (
                        self._c1_host_fast_sampler_history_skips == 1
                        or self._c1_host_fast_sampler_history_skips % 512 == 0
                    ):
                        logger.info(
                            "MINICPMO45_STAGE1_C1_HOST_FAST event=sampler_history "
                            "skips=%d",
                            self._c1_host_fast_sampler_history_skips,
                        )
                else:
                    prepared_sampling_metadata = self._sampling_metadata_for_model_sampler(
                        sampling_metadata
                    )
                self._apply_duplex_sampling(logits, prepared_sampling_metadata)
                sampler_output = model_sample(logits, prepared_sampling_metadata)
                if sampler_output is not None:
                    return sampler_output
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        return super()._sample(logits, spec_decode_metadata)

    @torch.inference_mode()
    @_stage1_host_ledger("sample_tokens")
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        #  -------------------------------------- Omni-new -------------------------------------------------
        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None
        combined_hidden_states = None
        combined_multimodal_outputs = None
        mm_cpu = {}
        #  -------------------------------------- Omni-new -------------------------------------------------


        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            # receive sampled token ids from the last PP rank when using
            # async scheduling + pipeline parallelism so downstream code
            # (e.g., PCP input preparation) can access them.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return self.attach_omni_connector_output(EMPTY_MODEL_RUNNER_OUTPUT)

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return self.attach_omni_connector_output(output)

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
            multimodal_outputs, # Omni-Specific
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None
        hidden_seq_len = int(hidden_states.shape[0])
        scheduled_seq_len = int(scheduler_output.total_num_scheduled_tokens)

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        #  -------------------------------------- Omni-new -------------------------------------------------
        # Correct padding values of prompt_token_ids to match the logits vocabulary size.
        if logits is not None and not self.input_batch.sampling_metadata.no_penalties:
            smd = self.input_batch.sampling_metadata
            if smd.prompt_token_ids is not None:
                logits_vocab = logits.shape[-1]
                if self.input_batch.vocab_size > logits_vocab:
                    smd.prompt_token_ids = smd.prompt_token_ids.clamp(max=logits_vocab)

        # Drop min-tokens stop ids the head cannot emit (e.g. the text
        # tokenizer EOS folded into all_stop_token_ids on a narrow codec
        # talker head); they would index_put_ out of bounds (#4962).
        if logits is not None:
            sanitize_min_tokens_stop_ids(
                self.input_batch.sampling_metadata.logitsprocs,
                logits.shape[-1],
            )
        #  -------------------------------------- Omni-new -------------------------------------------------


        with record_function_or_nullcontext("sample_token"):
            sampler_output = self._sample(logits, spec_decode_metadata)
        known_controller_token = self._known_controller_token_for_output
        self._known_controller_token_for_output = None

        if self.need_accepted_tokens:
            if self.sampling_done_event is None:
                self.sampling_done_event = torch.npu.Event()

            assert self.sampling_done_event is not None
            self.sampling_done_event.record()

        self.valid_sampled_token_count_gpu: torch.Tensor | None = None # type: ignore[no-redef]

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                aux_hidden_states,
                sample_hidden_states,
                batch_desc,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        finalize_deferred_codec_output = getattr(
            self.model,
            "finalize_deferred_codec_output",
            None,
        )
        if (
            getattr(self.model, "model_stage", None) == "tts"
            and callable(finalize_deferred_codec_output)
        ):
            multimodal_outputs = finalize_deferred_codec_output(
                multimodal_outputs,
                valid_sampled_token_ids,
            )

        with record_function_or_nullcontext("draft_token"):
            if self.speculative_config:
                use_padded_batch = (
                    self.speculative_config
                    and (self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model())
                    and not self.speculative_config.disable_padded_drafter_batch
                )
                if use_padded_batch:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            # vLLM v0.18 defers KV connector finalization during target-model
            # forward when speculative decoding is enabled. Finalize here after
            # draft model runs so KV pool save/put can complete.
            if self.speculative_config is not None:
                self.finalize_kv_connector()

        routed_experts_lists = None
        if self.model_config.enable_return_routed_experts:
            capturer = self.routed_experts_capturer
            if capturer is not None and hasattr(self.input_batch, "num_tokens_no_spec"):
                routed_experts_lists = self._omni_extract_routed_experts(scheduler_output)

        #  -------------------------------------- Omni-new -------------------------------------------------
        engine_output_type, downstream_req_ids = self._resolve_pooler_payload_req_ids(req_ids_output_copy)
        sparse_mm_req_ids = self._sparse_mm_req_ids(multimodal_outputs)
        sparse_mm_index = {rid: i for i, rid in enumerate(sparse_mm_req_ids or [])}
        if sparse_mm_req_ids is not None:
            sparse_req_id_set = set(sparse_mm_req_ids)
            downstream_req_ids = [rid for rid in req_ids_output_copy if rid in sparse_req_id_set]
        needs_pooler_payload = len(downstream_req_ids) > 0
        downstream_req_id_set = set(downstream_req_ids)
        hidden_states_cpu = None
        req_hidden_states_cpu: dict[str, torch.Tensor] | None = None
        audio_sparse_output = sparse_mm_req_ids is not None
        needs_scheduled_hidden_payload = needs_pooler_payload and (
            self.omni_prefix_cache is None or not self._model_needs_full_prefix_hidden_states()
        )
        if needs_scheduled_hidden_payload:
            num_valid_tokens = min(
                int(scheduler_output.total_num_scheduled_tokens),
                int(hidden_states.shape[0]),
            )
            if audio_sparse_output:
                pass
            elif len(downstream_req_ids) == len(req_ids_output_copy):
                hidden_states_cpu = hidden_states[:num_valid_tokens].detach().to("cpu").contiguous()
            else:
                req_hidden_states_cpu = {}
        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )
        query_start_loc_cpu = self.query_start_loc.cpu
        if callable(query_start_loc_cpu):
            query_start_loc_cpu = query_start_loc_cpu()

        pooler_output: list[dict[str, object]] | None = None
        if needs_pooler_payload:
            combined_hidden_states = None
            combined_multimodal_outputs = None
            mm_cpu = None
            if self.omni_prefix_cache is not None:
                (
                    combined_hidden_states,
                    combined_multimodal_outputs,
                ) = self._maybe_get_combined_prefix_cache_tensors(
                    hidden_states,
                    multimodal_outputs,
                    scheduler_output.num_scheduled_tokens,
                )
            if self.omni_prefix_cache is None or combined_multimodal_outputs is None:
                mm_cpu = build_mm_cpu(
                    flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs
                )

            self._process_additional_information_updates(
                hidden_states,
                multimodal_outputs,
                num_scheduled_tokens_np,
                scheduler_output,
                combined_hidden_states,
                combined_multimodal_outputs,
                req_ids_filter=downstream_req_id_set,
            )

            if req_hidden_states_cpu is not None and combined_hidden_states is None:
                for rid in downstream_req_ids:
                    idx = req_id_to_index_output_copy[rid]
                    start = int(query_start_loc_cpu[idx])
                    sched = int(num_scheduled_tokens_np[idx])
                    end = start + sched
                    req_hidden_states_cpu[rid] = hidden_states[start:end].detach().to("cpu").contiguous()

            pooler_output = []
            for rid in req_ids_output_copy:
                if rid not in downstream_req_id_set:
                    pooler_output.append({})
                    continue
                idx = req_id_to_index_output_copy[rid]
                start = int(query_start_loc_cpu[idx])
                sched = int(num_scheduled_tokens_np[idx])
                end = start + sched
                payload: dict[str, object] = {}
                if not audio_sparse_output:
                    if req_hidden_states_cpu is not None and combined_hidden_states is None:
                        req_hidden_states = req_hidden_states_cpu[rid]
                    else:
                        req_hidden_states = self._resolve_req_hidden_states(
                            hidden_states_cpu,
                            combined_hidden_states,
                            rid,
                            start,
                            end,
                        )
                    payload["hidden"] = req_hidden_states

                mm_payload: dict[str, object] = {}
                if combined_multimodal_outputs or mm_cpu:
                    if combined_multimodal_outputs:
                        # Prefix cache enabled; all items have already been processed
                        # and split apart for each request as needed, and all tensors
                        # have already been detached to the CPU.  Lists are kept as
                        # passthrough data for consistent behavior in postprocess.
                        # Recurse into nested dicts so list-valued sub-keys (e.g.
                        # embed.tts_bos = [tensor]) are unwrapped to bare tensors
                        # at the leaves; downstream flatten_payload then yields a
                        # wire-clean dict[str, torch.Tensor].
                        def _unwrap_lists(v):
                            if isinstance(v, list):
                                return v[idx] if idx < len(v) else v[0]
                            if isinstance(v, dict):
                                return {k: _unwrap_lists(sv) for k, sv in v.items()}
                            return v

                        for mm_key in combined_multimodal_outputs.keys():
                            mm_payload[mm_key] = _unwrap_lists(combined_multimodal_outputs[mm_key][rid])
                    else:
                        for mm_key, mm_val in mm_cpu.items():
                            if mm_key in {"meta.req_id", "meta.sparse_audio"}:
                                continue
                            if audio_sparse_output and isinstance(mm_val, list):
                                sparse_idx = sparse_mm_index.get(rid)
                                if sparse_idx is None:
                                    continue
                                if sparse_idx >= len(mm_val):
                                    logger.warning(
                                        "Sparse multimodal payload mismatch for request %s: index %d >= %d.",
                                        rid,
                                        sparse_idx,
                                        len(mm_val),
                                    )
                                    continue
                                sparse_val = mm_val[sparse_idx]
                                mm_payload[mm_key] = (
                                    sparse_val.clone() if isinstance(sparse_val, torch.Tensor) else sparse_val
                                )
                                continue
                            mm_payload[mm_key] = to_payload_element(
                                element=mm_val,
                                idx=idx,
                                start=start,
                                end=end,
                                pass_lists_through=False,
                                seq_len=hidden_seq_len,
                                scheduled_seq_len=scheduled_seq_len,
                            )
                    payload.update(mm_payload)
                pooler_output.append(flatten_payload(payload))

        pooler_output = pooler_output or []
        if self._async_chunk and stage_sends_async_output(self.model_config):
            pooler_inter, pooler_client = partition_payload_list(pooler_output)
        else:
            # Non-async-chunk ships the full payload to the next stage via
            # inter_stage_outputs (the NPU runner has no separate full-payload
            # accumulate). #4527's (None, pooler_output) starved it. (PR #4792)
            pooler_inter, pooler_client = pooler_output, pooler_output

        # [Omni] Full-payload send-side accumulation. Mirrors gpu_ar_model_runner.py.
        if pooler_inter and self._should_accumulate_full_payload_output():
            with record_function_or_nullcontext("omni_output_builder:accumulate_full_payload_output"):
                for i, rid in enumerate(req_ids_output_copy):
                    req_state = self.requests.get(rid)
                    if req_state is not None and pooler_inter[i]:
                        self.accumulate_full_payload_output(rid, pooler_inter[i], req_state)

        inter_stage_outputs = self._build_multimodal_outputs(pooler_inter)
        multimodal_outputs = (
            inter_stage_outputs if pooler_client is pooler_inter else self._build_multimodal_outputs(pooler_client)
        )
        model_runner_output = OmniModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=None,
            multimodal_outputs=multimodal_outputs,
            inter_stage_outputs=inter_stage_outputs,
            kv_connector_output=kv_connector_output,
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
            cudagraph_stats=cudagraph_stats,
        )
        model_runner_output.kv_extracted_req_ids = kv_extracted_req_ids
        model_runner_output.routed_experts = routed_experts_lists
        with record_function_or_nullcontext("omni_output_builder:get_omni_connector_output"):
            model_runner_output.omni_connector_output = self.get_omni_connector_output()
        #  -------------------------------------- Omni-new -------------------------------------------------

        if self.ascend_config.profiling_chunk_config.enabled and hasattr(self, "_execution_start_time"):
            self._sync_device()
            model_runner_output.execution_time_ms = (time.perf_counter() - self._execution_start_time) * 1000.0

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB update"):
                self.eplb_updator.forward_end()

        if self.debugger is not None:
            self.debugger.stop()
            self.debugger.step()

        if self.need_accepted_tokens:
            assert self.sampling_done_event is not None
            with (
                record_function_or_nullcontext("async_state_update"),
                torch.npu.stream(global_stream()),
            ):
                global_stream().wait_event(self.sampling_done_event)
                self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        # In async scheduling + PP, broadcast sampled token ids from the
        # last PP rank so other PP ranks can receive them without going
        # through the scheduler/engine IPC path.
        if self.use_async_scheduling:
            pp = get_pp_group()
            if pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

        if not self.use_async_scheduling:
            return model_runner_output
        if known_controller_token is not None:
            if sampler_output.logprobs_tensors is not None:
                raise RuntimeError(
                    "Known controller bypass cannot return logprobs"
                )
            async_output = _KnownControllerAsyncModelRunnerOutput(
                model_runner_output=model_runner_output,
                token=known_controller_token,
                invalid_req_indices=invalid_req_indices,
            )
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )
            return async_output
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )
        return async_output

    #  -------------------------------------- Omni-new -------------------------------------------------
    def _resolve_global_request_id(self, req_id: str) -> str:
        """Resolve global request ID from request state."""
        req_state = self.requests.get(req_id)
        if not req_state:
            return req_id

        add_info = self.model_intermediate_buffer.get(req_id, {})
        global_id = add_info.get("global_request_id")
        if global_id:
            if isinstance(global_id, list) and global_id:
                global_id = global_id[0]
            if isinstance(global_id, bytes):
                return global_id.decode("utf-8")
            return str(global_id)
        return req_id
    #  -------------------------------------- Omni-new -------------------------------------------------
