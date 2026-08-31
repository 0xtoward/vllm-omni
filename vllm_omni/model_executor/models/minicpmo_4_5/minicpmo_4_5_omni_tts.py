# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 native autoregressive Talker.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Continuously generate request-aligned discrete audio-code deltas
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.sampler import Sampler

from vllm_omni.experimental.fullduplex.engine.intermediate import (
    get_tts_handoff,
    normalize_handoff_tensor,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_REPETITION_WINDOW = 16
_MIN_AUDIO_TOKENS = 64
_MAX_AUDIO_TOKENS = 2048
_AUDIO_TOKENS_PER_TEXT_TOKEN = 10
# Codec-token sampling happens inside the model; vLLM sampling parameters
# only choose the Talker's binary continue/stop row.
_CODEC_SEED = 42
_CODEC_TEMPERATURE = 0.8
_CODEC_TOP_K = 25
_CODEC_TOP_P = 0.85
_CODEC_REPETITION_PENALTY = 1.05
_CODEC_MIN_TOKENS = 50
_DUPLEX_CODEC_TOKENS_PER_CHUNK = 26


@dataclass(frozen=True, slots=True)
class _CodecProposal:
    """One sampled codec token before any request-visible state commit."""

    request_id: str
    sampled: torch.Tensor
    step_before: int
    min_tokens: int
    max_tokens: int
    codes_before: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CodecCommitState:
    """Resolved one-token prefix that may be committed exactly once."""

    request_id: str
    step_before: int
    step_after: int
    steps_advanced: int
    current_code: torch.Tensor
    codes_after: torch.Tensor
    codec_delta: torch.Tensor
    reached_limit: bool
    is_eos: bool
    finished: bool
    terminal: torch.Tensor


@dataclass(frozen=True, slots=True)
class _DeferredCodecEntry:
    """One model proposal waiting for the runner's sampled stop token."""

    output_index: int
    info: dict[str, Any]
    state: dict[str, Any]
    proposal: _CodecProposal


@dataclass(frozen=True, slots=True)
class _DeferredCodecBatch:
    """Single in-flight K=1 codec commit owned by ``sample_tokens``.

    The model still samples exactly one codec token during forward.  Only the
    CPU-visible state mutation is delayed until the runner has copied its
    binary continue/stop samples to CPU for normal scheduler bookkeeping.
    """

    multimodal_outputs: dict[str, Any]
    entries: tuple[_DeferredCodecEntry, ...]
    empty_delta: torch.Tensor
    sparse_output: bool


@dataclass(frozen=True, slots=True)
class _KnownControllerHint:
    """One normally committed nonterminal step with a known runner token.

    Codec sampling, RNG advance, repetition history, sparse handoff and model
    state all use the stock path before this sideband exists.  A terminal step
    is never represented by the hint and therefore always uses stock sampling,
    D2H, finish ownership and cleanup.
    """

    request_id: str
    step_before: int
    step_after: int
    token: int


@dataclass(slots=True)
class _CodecEosBatch:
    """One private K-row EOS-resolution transaction.

    The first K-1 codec rows are committed optimistically as nonterminal so the
    next ordinary q_len=1 forward can consume them.  Nothing is sent to Stage2
    while the transaction is open.  At row K-1 one D2H resolves all sampled
    tokens; an early EOS rolls back only the private suffix starting at that
    EOS before publishing the terminal payload.
    """

    k: int
    request_id: str
    step_before: int
    history_version_before: int
    sparse_pending_len_before: int
    sparse_suppressed_before: int
    sparse_suppressed_total_before: int
    codes_before: torch.Tensor
    current_code_before: torch.Tensor
    proposals: list[_CodecProposal]


def _max_audio_tokens(condition_tokens: int) -> int:
    """Bound codec generation with a conservative text-length estimate.

    EOS is masked for the first 50 steps, so a direct ``text_tokens * 10``
    limit can terminate short responses before EOS is eligible. The 2048
    ceiling matches the checkpoint's native generation default and keeps the
    sequence within the Talker's 4096-position context.
    """
    return max(
        _MIN_AUDIO_TOKENS,
        min(_MAX_AUDIO_TOKENS, condition_tokens * _AUDIO_TOKENS_PER_TEXT_TOKEN),
    )


def _restore_weight_norm_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    return torch._weight_norm(weight_v, weight_g, dim=0)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    history: torch.Tensor,
    *,
    penalty: float,
    window_size: int,
    inplace: bool = False,
    penalty_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match MiniCPMTTS' frequency-aware repetition penalty.

    Only ids in the short history window can be changed. Counting the window
    directly avoids the full-vocabulary AI-CPU bincount and full-vocabulary
    scale passes on every decode step.
    """
    if penalty == 1.0 or history.numel() == 0:
        return logits
    recent = history.reshape(-1)[-window_size:].to(device=logits.device, dtype=torch.long)
    counts = (recent.unsqueeze(1) == recent.unsqueeze(0)).sum(dim=1).to(dtype=logits.dtype)
    penalty_base = (
        penalty_tensor
        if penalty_tensor is not None
        else torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype)
    )
    alpha = torch.pow(penalty_base, counts)
    scored = logits if inplace else logits.clone()
    selected = scored[..., recent]
    scored[..., recent] = torch.where(selected < 0, selected * alpha, selected / alpha)
    return scored


def _restore_top_p_mask(
    remove: torch.Tensor,
    sorted_indices: torch.Tensor,
    *,
    use_npu_scatter_nd_update: bool,
    scatter_nd_update_op: Any | None = None,
) -> torch.Tensor:
    """Restore a sorted Top-P mask without an in-place graph side effect.

    MiniCPM-o's C=1 sampler writes every vocabulary position exactly once, so
    a functional ScatterNdUpdate is equivalent to ``Tensor.scatter``.  Keep a
    strict shape/device gate: flattening indices is only valid for one row.
    """
    if (
        use_npu_scatter_nd_update
        and remove.device.type == "npu"
        and remove.ndim == 2
        and remove.shape[0] == 1
        and sorted_indices.shape == remove.shape
    ):
        op = scatter_nd_update_op
        if op is None:
            npu_ops = getattr(torch.ops, "npu", None)
            op = getattr(npu_ops, "npu_scatter_nd_update", None)
        if op is not None:
            flat_remove = remove.reshape(-1)
            return op(
                torch.empty_like(flat_remove),
                sorted_indices.reshape(-1, 1),
                flat_remove,
            ).reshape_as(remove)
    return remove.scatter(-1, sorted_indices, remove)


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
    min_tokens_to_keep: int = 3,
    inplace: bool = False,
    use_npu_scatter_nd_update: bool = False,
) -> torch.Tensor:
    """Apply the same candidate floors as the upstream Transformers warpers."""
    filtered = logits if inplace else logits.clone()
    vocab_size = filtered.shape[-1]
    # MiniCPM-o's gen_logits() appends TopPLogitsWarper before
    # TopKLogitsWarper. The order is observable for fixed-seed sampling.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = _restore_top_p_mask(
            remove,
            sorted_indices,
            use_npu_scatter_nd_update=use_npu_scatter_nd_update,
        )
        filtered.masked_fill_(remove, float("-inf"))
    if top_k is not None and top_k > 0:
        keep = min(vocab_size, max(int(top_k), min_tokens_to_keep))
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    return filtered


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible hidden-state projector used by MiniCPMTTS."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class _CodecSamplerCapturedGraph:
    def __init__(
        self,
        graph: Any,
        pool: object,
        static_inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        self.graph = graph
        self.pool = pool
        self.static_inputs = static_inputs
        self.output = output

    def replay(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        for static, current in zip(self.static_inputs, inputs, strict=True):
            static.copy_(current)
        self.graph.replay()
        # Multinomial consumes this persistent output immediately on the same
        # stream. The following replay is enqueued after that consume, so a
        # per-token clone/copy is unnecessary.
        return self.output


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """Runner-owned MiniCPM-o 4.5 Talker that emits codec tokens only."""

    requires_request_sample_eligibility = True
    supports_codec_embed_graph = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._batch_stop_logits: torch.Tensor | None = None
        self._continue_stop_row: torch.Tensor | None = None
        self._finished_stop_row: torch.Tensor | None = None
        self._fallback_stop_sampler = Sampler()
        self._request_generators: dict[str, torch.Generator] = {}
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        self._deferred_cleanup_ids: set[str] = set()
        self._deferred_codec_batch: _DeferredCodecBatch | None = None
        self._deferred_codec_commit_total = 0
        self._deferred_codec_reject_reasons_logged: set[str] = set()
        self._known_controller_hint: _KnownControllerHint | None = None
        self._known_controller_hint_hits = 0
        raw_eos_batch_k = os.environ.get(
            "VLLM_OMNI_MINICPMO45_STAGE1_EOS_BATCH_K",
            "0",
        )
        try:
            self._codec_eos_batch_k = int(raw_eos_batch_k)
        except ValueError:
            raise ValueError(
                "VLLM_OMNI_MINICPMO45_STAGE1_EOS_BATCH_K must be 0, 2, or 4"
            ) from None
        if self._codec_eos_batch_k not in (0, 2, 4):
            raise ValueError(
                "VLLM_OMNI_MINICPMO45_STAGE1_EOS_BATCH_K must be 0, 2, or 4"
            )
        self._codec_eos_batch: _CodecEosBatch | None = None
        self._codec_eos_batch_steps = 0
        self._codec_eos_batch_boundaries = 0
        self._codec_eos_batch_no_eos = 0
        self._codec_eos_batch_terminal_rows = [0, 0, 0, 0]
        self._codec_eos_batch_rollbacks = 0
        self._codec_eos_batch_rolled_back_rows = 0
        self._codec_eos_batch_wasted_suffix_rows = 0
        self._codec_eos_batch_aborts = 0
        self.register_buffer(
            "_codec_eos_batch_samples",
            torch.full((4,), -1, dtype=torch.long),
            persistent=False,
        )
        npu_default = "1" if current_omni_platform.is_npu() else "0"
        self._codec_sampler_graph_enabled = os.environ.get(
            "VLLM_OMNI_MINICPMO45_STAGE1_CODEC_SAMPLER_NPUGRAPH", npu_default
        ).lower() in ("1", "true", "yes", "on")
        self._codec_sampler_graphs: dict[tuple[Any, ...], _CodecSamplerCapturedGraph] = {}
        self._codec_sampler_graph_replayed_keys: set[tuple[Any, ...]] = set()
        self._codec_sampler_penalty_tensor: torch.Tensor | None = None
        self._codec_sampler_graph_failed = False
        self._codec_sampler_graph_hits = 0
        self._codec_sampler_graph_captures = 0
        self._codec_sampler_scatternd_enabled = os.environ.get(
            "VLLM_OMNI_MINICPMO45_STAGE1_CODEC_SAMPLER_SCATTERND",
            npu_default,
        ).lower() in ("1", "true", "yes", "on")
        self._codec_greedy_enabled = os.environ.get(
            "VLLM_OMNI_MINICPMO45_STAGE1_CODEC_GREEDY",
            "0",
        ).lower() in ("1", "true", "yes", "on")
        self._codec_embed_graph_enabled = os.environ.get(
            "VLLM_OMNI_MINICPMO45_STAGE1_CODEC_EMBED_GRAPH",
            "1",
        ).lower() in ("1", "true", "yes", "on")
        if self._codec_sampler_graph_enabled:
            logger.info("MINICPMO45_CODEC_SAMPLER_NPUGRAPH event=enabled")
        if self._codec_sampler_scatternd_enabled:
            logger.info("MINICPMO45_CODEC_SAMPLER_SCATTERND event=enabled")
        if self._codec_greedy_enabled:
            logger.info("MINICPMO45_CODEC_GREEDY event=enabled")
        if self._codec_embed_graph_enabled:
            logger.info("MINICPMO45_STAGE1_CODEC_EMBED_GRAPH event=enabled")
        if self._codec_eos_batch_k:
            logger.warning(
                "MINICPMO45_STAGE1_EOS_BATCH event=enabled k=%d "
                "judge_default=true explicit_opt_out=0",
                self._codec_eos_batch_k,
            )
        raw_sparse_chunk = os.environ.get(
            "VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES",
            "0",
        )
        try:
            self._sparse_chunk_frames = max(0, int(raw_sparse_chunk))
        except ValueError:
            raise ValueError(
                "VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES must be an integer"
            ) from None
        self._sparse_emit_total = 0
        self._sparse_suppressed_total = 0
        if self._sparse_chunk_frames:
            logger.info(
                "MINICPMO45_SPARSE_CHUNK event=enabled frames=%d",
                self._sparse_chunk_frames,
            )

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
            self._codec_seed = int(getattr(tts_config, "seed", _CODEC_SEED))
            self._codec_temperature = float(getattr(tts_config, "temperature", _CODEC_TEMPERATURE))
            self._codec_top_k = int(getattr(tts_config, "top_k", _CODEC_TOP_K))
            self._codec_top_p = float(getattr(tts_config, "top_p", _CODEC_TOP_P))
            self._codec_repetition_penalty = float(getattr(tts_config, "repetition_penalty", _CODEC_REPETITION_PENALTY))
            self._codec_min_tokens = int(getattr(tts_config, "min_new_tokens", _CODEC_MIN_TOKENS))
        else:
            self._tts_config = None

        self.has_preprocess = True
        self.has_postprocess = False
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("audio_codes", "accumulated"),
        }
        self._init_native_talker(prefix)

    def _init_native_talker(self, prefix: str) -> None:
        if self._tts_config is None:
            raise ValueError("MiniCPM-o continuous Talker requires tts_config")
        cfg = self._tts_config
        if int(getattr(cfg, "num_vq", 1)) != 1:
            raise ValueError(
                "MiniCPM-o continuous Talker currently requires num_vq=1; "
                f"checkpoint reports {getattr(cfg, 'num_vq', None)}"
            )
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            hidden_act=getattr(cfg, "hidden_act", "silu"),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=False,
        )
        talker_config = self.vllm_config.with_hf_config(llama_config, architectures=["LlamaForCausalLM"])
        talker_config.model_config.hf_text_config = llama_config
        self.tts_model = LlamaModel(
            vllm_config=talker_config,
            prefix=maybe_prefix(prefix, "tts_obj.model"),
        )
        self.emb_text = nn.Embedding(int(cfg.num_text_tokens), int(cfg.hidden_size))
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg.llm_dim), int(cfg.hidden_size))
        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(cfg.num_audio_tokens), int(cfg.hidden_size)) for _ in range(int(cfg.num_vq))]
        )
        self.register_buffer(
            "_codec_embed_graph_active",
            torch.zeros(
                1, dtype=torch.bool, device=self.emb_code[0].weight.device
            ),
            persistent=False,
        )
        self._codec_embed_graph_active_host = False
        self.head_code = nn.ModuleList(
            [nn.Linear(int(cfg.hidden_size), int(cfg.num_audio_tokens), bias=False) for _ in range(int(cfg.num_vq))]
        )
        self.make_empty_intermediate_tensors = self.tts_model.make_empty_intermediate_tensors

    def set_codec_embed_graph_active(self, active: bool) -> None:
        """Select graph-local codec lookup without changing its signature."""

        active = bool(active)
        if not self._codec_embed_graph_enabled or (
            active == self._codec_embed_graph_active_host
        ):
            return
        self._codec_embed_graph_active.fill_(active)
        self._codec_embed_graph_active_host = active

    def _boundary_embeddings(self) -> torch.Tensor:
        """Embed the ``<text_eos><audio_bos>`` tail every condition ends with."""
        ids = torch.tensor(
            [self._text_eos_id, self._tts_bos_id],
            device=self.emb_text.weight.device,
            dtype=torch.long,
        )
        return self.emb_text(ids)

    def _build_condition_embeddings(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
        *,
        native_duplex: bool = False,
    ) -> torch.Tensor:
        if tts_token_ids.numel() == 0 or tts_hidden_states.numel() == 0:
            # The thinker can legally emit an empty speech segment (<|tts_bos|>
            # immediately followed by a boundary token) when it decides not to
            # speak. Condition on the boundary tokens alone, which matches the
            # 2-token scheduler prompt the stage bridge builds for an empty
            # handoff.
            return self._boundary_embeddings()
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        token_ids = tts_token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = tts_hidden_states.to(device=device, dtype=dtype)
        if hidden.shape[0] != token_ids.shape[0] and token_ids.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o Talker condition length mismatch: "
                f"token_ids={token_ids.shape[0]} hidden_states={hidden.shape[0]}"
            )
        text_embeds = self.emb_text(token_ids)
        hidden_embeds = self.projector_semantic(hidden)
        if self._normalize:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        audio_bos = self.emb_text(torch.tensor([self._tts_bos_id], device=device, dtype=torch.long))
        condition = text_embeds + hidden_embeds
        if native_duplex:
            # Match MiniCPMTTS.generate_chunk's streaming condition.
            return torch.cat([condition, audio_bos], dim=0)
        return torch.cat([condition, self._boundary_embeddings()], dim=0)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build request-local prefill/decode embeddings for the vLLM runner."""
        del input_embeds
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        state = info_dict.get("audio_state")
        first_call = not isinstance(state, dict)

        if is_prefill or first_call:
            token_ids, hidden_states = get_tts_handoff(info_dict)
            hidden_states = normalize_handoff_tensor(hidden_states)
            # Cross-process stage transport serializes CPU tensors as lists.
            # Normalize both local tensor handoffs and transported payloads
            # before validating/building the Talker condition.
            if isinstance(token_ids, (list, tuple)):
                token_ids = torch.as_tensor(token_ids, dtype=torch.long)
            if isinstance(hidden_states, (list, tuple)):
                hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
            if not isinstance(token_ids, torch.Tensor) or not isinstance(hidden_states, torch.Tensor):
                available = sorted(key for key in info_dict if not key.startswith("_"))
                raise ValueError(
                    "MiniCPM-o Talker requires tensor tts_token_ids and "
                    "tts_hidden_states conditioning; "
                    f"received token_ids={type(token_ids).__name__}, "
                    f"hidden_states={type(hidden_states).__name__}, "
                    f"available_keys={available}"
                )
            # An empty condition means the thinker chose not to speak: finish the
            # request up front so it emits zero audio codes instead of killing
            # the stage engine.
            empty_condition = token_ids.numel() == 0 or hidden_states.numel() == 0
            if empty_condition:
                logger.warning_once(
                    "MiniCPM-o Talker received an empty condition (request %s); this request produces no audio.",
                    info_dict.get("request_id"),
                )
            native_duplex = bool(info_dict.get("native_duplex", False))
            full_embeds = self._build_condition_embeddings(
                token_ids,
                hidden_states,
                native_duplex=native_duplex,
            )
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            request_id = str(info_dict.get("request_id", "0"))
            meta = info_dict.get("meta")
            # The handoff rebuilds only the tail-aligned Talker condition.
            # Materialize zero-token embeddings for any scheduler prompt
            # prefix so chunked prefill can slice from a non-zero offset.
            prompt_len = info_dict.get("_omni_prompt_len")
            target_len = int(prompt_len) if prompt_len is not None else offset + span_len
            prefix_len = target_len - full_embeds.shape[0]
            if prefix_len > 0:
                placeholder_ids = torch.zeros(
                    prefix_len,
                    dtype=torch.long,
                    device=self.emb_text.weight.device,
                )
                full_embeds = torch.cat([self.emb_text(placeholder_ids), full_embeds], dim=0)
            embeds = full_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o Talker prefill span exceeds condition: "
                    f"request_id={info_dict.get('request_id')} offset={offset} "
                    f"span={span_len} condition={full_embeds.shape[0]} "
                    f"tts_ids={token_ids.shape[0]} tts_hidden={hidden_states.shape[0]} "
                    f"prompt_len={info_dict.get('_omni_prompt_len')}"
                )
            duplex_boundary = isinstance(meta, dict) and (
                bool(meta.get("turn_start", False)) or bool(meta.get("turn_end", False))
            )
            if native_duplex:
                max_tokens = _DUPLEX_CODEC_TOKENS_PER_CHUNK
                min_tokens = 0 if duplex_boundary else _DUPLEX_CODEC_TOKENS_PER_CHUNK
            else:
                max_tokens = _max_audio_tokens(int(token_ids.numel()))
                min_tokens = self._codec_min_tokens
            state = {
                "step": 0,
                # Monotonic request-local commit generation used by the EOS
                # K-batch rollback contract. Every optimistic row advances
                # this counter exactly once.
                "history_version": 0,
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "finished": empty_condition,
            }
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            request_states[request_id] = state
            empty_codes = torch.empty(0, dtype=torch.long, device=embeds.device)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    "audio_codes": {
                        "current": empty_codes,
                        "accumulated": empty_codes,
                    },
                },
            )

        current = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(current, torch.Tensor) or current.numel() != 1:
            if state.get("finished"):
                # A request that finished before sampling any code can still be
                # scheduled for decode steps while sampling min_tokens masks the
                # stop token. make_omni_output ignores its hidden states, so any
                # shape-correct embedding will do.
                weight = self.emb_code[0].weight
                return input_ids, weight.new_zeros((span_len, weight.shape[1])), {}
            raise RuntimeError("MiniCPM-o Talker decode is missing the previous request-local audio code")
        code = current.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(1)
        embeds = self.emb_code[0](code)
        return input_ids, embeds, {}

    def can_preprocess_c1_decode(
        self,
        info_dict: dict[str, Any],
        out: torch.Tensor,
    ) -> bool:
        """Whether one established codec row can use the sealed runner lane."""

        state = info_dict.get("audio_state")
        current = (info_dict.get("audio_codes", {}) or {}).get("current")
        return bool(
            isinstance(state, dict)
            and not state.get("finished")
            and isinstance(current, torch.Tensor)
            and current.shape == (1,)
            and current.dtype == torch.long
            and current.device == out.device
            and out.shape == (1, self.emb_code[0].weight.shape[1])
            and out.dtype == self.emb_code[0].weight.dtype
        )

    def preprocess_c1_decode_into(
        self,
        info_dict: dict[str, Any],
        out: torch.Tensor,
    ) -> bool:
        """Write the exact stock decode embedding into the persistent buffer."""

        if not self.can_preprocess_c1_decode(info_dict, out):
            return False
        current = info_dict["audio_codes"]["current"]
        out.copy_(self.emb_code[0](current))
        return True

    def _request_generator(self, request_id: str, device: torch.device) -> torch.Generator:
        generator = self._request_generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._codec_seed)
            self._request_generators[request_id] = generator
        return generator

    def can_defer_codec_commit(self, request_id: str, sampling_params: Any) -> bool:
        """Whether runner stop sampling is exactly the model's stop decision.

        Deferral is deliberately restricted to the official non-duplex binary
        controller contract.  Client logit constraints or mismatched token
        limits could change the runner's sampled stop token after the model has
        constructed it, so those requests retain the established immediate
        codec commit path.
        """
        def reject(reason: str) -> bool:
            # Keep diagnostics bounded by rejection reason so a missed fast
            # path is observable without adding per-token log traffic.
            if reason not in self._deferred_codec_reject_reasons_logged:
                self._deferred_codec_reject_reasons_logged.add(reason)
                logger.info(
                    "MINICPMO45_DEFERRED_CODEC_COMMIT event=reject reason=%s "
                    "request_id=%s state_keys=%s min_tokens=%r max_tokens=%r "
                    "stop_ids=%r",
                    reason,
                    request_id,
                    tuple(self._request_audio_states),
                    getattr(sampling_params, "min_tokens", None),
                    getattr(sampling_params, "max_tokens", None),
                    getattr(sampling_params, "all_stop_token_ids", None),
                )
            return False

        state = self._request_audio_states.get(str(request_id))
        if not isinstance(state, dict) or state.get("finished"):
            return reject("missing_or_finished_state")
        try:
            min_tokens = int(getattr(sampling_params, "min_tokens"))
            max_tokens = int(getattr(sampling_params, "max_tokens"))
            stop_token_ids = {
                int(token_id)
                for token_id in getattr(sampling_params, "all_stop_token_ids")
                if 0 <= int(token_id) < 2
            }
        except (AttributeError, TypeError, ValueError):
            return reject("invalid_sampling_contract")
        if min_tokens != int(state.get("min_tokens", self._codec_min_tokens)):
            return reject("min_tokens_mismatch")
        if max_tokens < int(state.get("max_tokens", 2048)):
            return reject("max_tokens_too_small")
        if stop_token_ids != {1}:
            return reject("stop_token_ids_mismatch")
        if any(
            (
                getattr(sampling_params, "allowed_token_ids", None),
                getattr(sampling_params, "logit_bias", None),
                getattr(sampling_params, "bad_words", None),
                getattr(sampling_params, "structured_outputs", None),
                getattr(sampling_params, "ignore_eos", False),
                getattr(sampling_params, "logprobs", None),
                getattr(sampling_params, "prompt_logprobs", None),
            )
        ):
            return reject("request_constraints")
        return True

    def take_known_controller_token(
        self,
        request_id: str,
        sampling_params: Any,
    ) -> int | None:
        """Consume one committed nonterminal hint after exact revalidation.

        Clear-before-check makes the hint one-shot even when validation fails
        or the request is aborted.  Request identity and the committed state
        step must both match; no codec tensor or RNG state is changed here.
        """

        hint = self._known_controller_hint
        self._known_controller_hint = None
        if hint is None or hint.request_id != str(request_id):
            return None
        state = self._request_audio_states.get(hint.request_id)
        if not isinstance(state, dict) or state.get("finished"):
            return None
        if int(state.get("step", -1)) != hint.step_after:
            return None
        if hint.step_after != hint.step_before + 1 or hint.token != 0:
            return None

        # This bypass returns no logprobs and does not apply client-side logit
        # processors.  Require absence, rather than truthiness, so values such
        # as logprobs=0 cannot silently enter the fast lane.
        if sampling_params is None:
            return None
        try:
            raw_stop_ids = {
                int(token_id)
                for token_id in sampling_params.all_stop_token_ids
            }
        except (AttributeError, TypeError, ValueError):
            return None
        if raw_stop_ids != {1}:
            return None
        # vLLM materializes the default ``bad_words`` as ``[]``.
        # The empty list is a no-op, while every real request processor stays
        # fail-closed on the stock sampler path.
        if (
            any(
                getattr(sampling_params, name, None) is not None
                for name in (
                    "allowed_token_ids",
                    "logit_bias",
                    "structured_outputs",
                    "logprobs",
                    "prompt_logprobs",
                )
            )
            or bool(getattr(sampling_params, "bad_words", None))
            or bool(getattr(sampling_params, "ignore_eos", False))
        ):
            return None
        if not self.can_defer_codec_commit(hint.request_id, sampling_params):
            return None

        self._known_controller_hint_hits += 1
        if (
            self._known_controller_hint_hits == 1
            or self._known_controller_hint_hits % 512 == 0
        ):
            logger.info(
                "MINICPMO45_KNOWN_CONTROLLER event=consume hits=%d "
                "request_id=%s step=%d token=%d",
                self._known_controller_hint_hits,
                hint.request_id,
                hint.step_before,
                hint.token,
            )
        return hint.token

    def _codec_sampler_probabilities(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        *,
        allow_eos: bool,
    ) -> torch.Tensor:
        logits = self.head_code[0](hidden_state).float() / self._codec_temperature
        logits = _apply_repetition_penalty(
            logits,
            history,
            penalty=self._codec_repetition_penalty,
            window_size=_REPETITION_WINDOW,
            inplace=True,
            penalty_tensor=self._codec_sampler_penalty_tensor,
        )
        if not allow_eos:
            logits[..., self._num_audio_tokens - 1] = float("-inf")
        logits = _apply_top_k_top_p(
            logits,
            top_k=self._codec_top_k,
            top_p=self._codec_top_p,
            min_tokens_to_keep=3,
            inplace=True,
            use_npu_scatter_nd_update=getattr(
                self,
                "_codec_sampler_scatternd_enabled",
                False,
            ),
        )
        return torch.softmax(logits, dim=-1)

    def _ensure_codec_sampler_penalty_tensor(
        self, device: torch.device
    ) -> torch.Tensor:
        penalty_tensor = self._codec_sampler_penalty_tensor
        if (
            penalty_tensor is None
            or penalty_tensor.device != device
            or penalty_tensor.dtype != torch.float32
        ):
            penalty_tensor = torch.tensor(
                self._codec_repetition_penalty,
                device=device,
                dtype=torch.float32,
            )
            self._codec_sampler_penalty_tensor = penalty_tensor
        return penalty_tensor

    def _capture_codec_sampler_graph(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
        *,
        allow_eos: bool,
    ) -> _CodecSamplerCapturedGraph:
        npu = torch.npu
        static_inputs = tuple(value.detach().clone() for value in inputs)
        # Materialize the scalar before capture; torch.as_tensor(...,
        # device="npu") inside capture performs a forbidden synchronous H2D.
        self._ensure_codec_sampler_penalty_tensor(static_inputs[0].device)
        npu.synchronize()
        graph = npu.NPUGraph()
        # Give each alternating allow_eos graph an independent pool and never
        # share workspace lifetime with the outer FULL_DECODE graph.
        pool = npu.graph_pool_handle()
        with torch.inference_mode(), npu.graph(graph, pool=pool):
            output = self._codec_sampler_probabilities(
                static_inputs[0], static_inputs[1], allow_eos=allow_eos
            )
        npu.synchronize()
        return _CodecSamplerCapturedGraph(graph, pool, static_inputs, output)

    def _run_codec_sampler_probabilities(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        *,
        allow_eos: bool,
    ) -> torch.Tensor:
        if self._codec_sampler_graph_failed:
            return self._codec_sampler_probabilities(
                hidden_state, history, allow_eos=allow_eos
            )
        stream_capture_probe = getattr(torch.npu, "is_current_stream_capturing", None)
        eligible = (
            self._codec_sampler_graph_enabled
            and hidden_state.device.type == "npu"
            and history.device.type == "npu"
            and hidden_state.device == history.device
            and tuple(hidden_state.shape) == (1, int(self._hidden_size))
            and tuple(history.shape) == (_REPETITION_WINDOW,)
            and history.dtype == torch.long
            and hidden_state.is_contiguous()
            and history.is_contiguous()
            and hasattr(torch, "npu")
            and hasattr(torch.npu, "NPUGraph")
            and hasattr(torch.npu, "graph")
            and hasattr(torch.npu, "graph_pool_handle")
            and callable(stream_capture_probe)
            and not stream_capture_probe()
        )
        if not eligible:
            return self._codec_sampler_probabilities(
                hidden_state, history, allow_eos=allow_eos
            )

        key = (
            allow_eos,
            tuple(hidden_state.shape),
            hidden_state.dtype,
            hidden_state.device.index,
            tuple(hidden_state.stride()),
            tuple(history.shape),
            history.dtype,
            history.device.index,
            tuple(history.stride()),
        )
        captured = self._codec_sampler_graphs.get(key)
        inputs = (hidden_state, history)
        if captured is not None:
            self._codec_sampler_graph_hits += 1
            first_key_replay = key not in self._codec_sampler_graph_replayed_keys
            if first_key_replay:
                self._codec_sampler_graph_replayed_keys.add(key)
            if first_key_replay or self._codec_sampler_graph_hits % 1024 == 0:
                logger.info(
                    "MINICPMO45_CODEC_SAMPLER_NPUGRAPH event=replay "
                    "hits=%d captures=%d allow_eos=%s first_key_replay=%s",
                    self._codec_sampler_graph_hits,
                    self._codec_sampler_graph_captures,
                    allow_eos,
                    first_key_replay,
                )
            return captured.replay(inputs)

        # The current request remains eager.  Capture has no RNG operation, so
        # executing the probability program a second time cannot consume an
        # extra random draw.  Warmup absorbs the one-time synchronization.
        eager = self._codec_sampler_probabilities(
            hidden_state, history, allow_eos=allow_eos
        )
        try:
            self._codec_sampler_graphs[key] = self._capture_codec_sampler_graph(
                inputs, allow_eos=allow_eos
            )
        except Exception:
            self._codec_sampler_graph_failed = True
            self._codec_sampler_graph_enabled = False
            logger.exception(
                "MINICPMO45_CODEC_SAMPLER_NPUGRAPH event=capture_failed allow_eos=%s",
                allow_eos,
            )
            logger.warning(
                "Stage1 codec sampler NPU graph capture failed; continuing with "
                "the selected eager sampler path"
            )
            return eager
        self._codec_sampler_graph_captures += 1
        logger.info(
            "MINICPMO45_CODEC_SAMPLER_NPUGRAPH event=captured captures=%d allow_eos=%s",
            self._codec_sampler_graph_captures,
            allow_eos,
        )
        return eager

    def _sample_audio_code(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        request_id: str,
        step: int,
        *,
        greedy: bool = False,
    ) -> torch.Tensor:
        request_states = getattr(self, "_request_audio_states", {})
        state = request_states.get(request_id)
        min_tokens = (
            int(state.get("min_tokens", self._codec_min_tokens))
            if isinstance(state, dict)
            else self._codec_min_tokens
        )
        allow_eos = step >= min_tokens
        if greedy:
            # Positive temperature scaling and Top-P/Top-K filtering preserve
            # the maximum-logit token.  Keep the model-specific repetition and
            # EOS contracts, then bypass sort/filter/softmax/RNG entirely.
            logits = self.head_code[0](hidden_state).float()
            penalty_tensor = self._ensure_codec_sampler_penalty_tensor(
                logits.device
            )
            logits = _apply_repetition_penalty(
                logits,
                history,
                penalty=self._codec_repetition_penalty,
                window_size=_REPETITION_WINDOW,
                inplace=True,
                penalty_tensor=penalty_tensor,
            )
            if not allow_eos:
                logits[..., self._num_audio_tokens - 1] = float("-inf")
            return torch.argmax(logits, dim=-1).reshape(())

        probabilities = self._run_codec_sampler_probabilities(
            hidden_state, history, allow_eos=allow_eos
        )
        return torch.multinomial(
            probabilities,
            num_samples=1,
            generator=self._request_generator(request_id, probabilities.device),
        ).reshape(())

    def _propose_codec(
        self,
        hidden_state: torch.Tensor,
        codes: torch.Tensor,
        *,
        request_id: str,
        step: int,
        min_tokens: int,
        max_tokens: int,
        greedy: bool = False,
    ) -> _CodecProposal:
        """Consume one random draw without mutating request-visible state.

        Once sampling succeeds, this logical step must be committed exactly
        once or the whole request must be abandoned. Retrying would consume a
        second random draw and would no longer preserve the stock token stream.
        """
        if not request_id:
            raise RuntimeError("MiniCPM-o Talker codec proposal requires a request id")
        if step < 0:
            raise RuntimeError("MiniCPM-o Talker codec proposal has a negative step")
        if min_tokens < 0 or max_tokens <= 0:
            raise RuntimeError("MiniCPM-o Talker codec proposal has invalid token limits")
        if not isinstance(codes, torch.Tensor) or codes.ndim != 1:
            raise RuntimeError("MiniCPM-o Talker codec history must be a 1-D tensor")
        if codes.dtype != torch.long:
            raise RuntimeError("MiniCPM-o Talker codec history must be int64")
        if codes.device != hidden_state.device:
            raise RuntimeError("MiniCPM-o Talker codec history/device mismatch")
        if greedy:
            sampled = self._sample_audio_code(
                hidden_state,
                codes,
                request_id,
                step,
                greedy=True,
            )
        else:
            sampled = self._sample_audio_code(hidden_state, codes, request_id, step)
        return _CodecProposal(
            request_id=request_id,
            sampled=sampled,
            step_before=step,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            codes_before=codes,
        )

    def _resolve_codec_proposal(
        self,
        proposal: _CodecProposal,
        *,
        empty_delta: torch.Tensor,
        runner_stop_token: int | None = None,
    ) -> _CodecCommitState:
        """Resolve EOS/limit and construct the accepted prefix without mutation."""
        if proposal.sampled.numel() != 1 or proposal.sampled.dtype != torch.long:
            raise RuntimeError("MiniCPM-o Talker proposal must contain one int64 token")
        if proposal.sampled.device != proposal.codes_before.device:
            raise RuntimeError("MiniCPM-o Talker proposal token/history device mismatch")
        if (
            empty_delta.shape != (0, 1)
            or empty_delta.dtype != torch.long
            or empty_delta.device != proposal.codes_before.device
        ):
            raise RuntimeError("MiniCPM-o Talker empty codec delta contract is invalid")
        step_after = proposal.step_before + 1
        reached_limit = step_after >= proposal.max_tokens
        if proposal.step_before < proposal.min_tokens or reached_limit:
            is_eos = False
            expected_stop_token = int(reached_limit)
            if runner_stop_token is not None and runner_stop_token != expected_stop_token:
                raise RuntimeError(
                    "MiniCPM-o Talker runner stop token disagrees with the "
                    f"forced codec boundary: got={runner_stop_token} "
                    f"expected={expected_stop_token}"
                )
        else:
            if runner_stop_token is None:
                is_eos = int(proposal.sampled.item()) == self._num_audio_tokens - 1
            else:
                if runner_stop_token not in (0, 1):
                    raise RuntimeError(
                        "MiniCPM-o Talker runner stop token must be binary, "
                        f"got {runner_stop_token}"
                    )
                is_eos = runner_stop_token == 1
        finished = is_eos or reached_limit
        if finished:
            codes_after = proposal.codes_before
            codec_delta = empty_delta
        else:
            codes_after = torch.cat(
                [
                    proposal.codes_before[-(_REPETITION_WINDOW - 1) :],
                    proposal.sampled.reshape(1),
                ]
            )
            codec_delta = proposal.sampled.reshape(1, 1)
        return _CodecCommitState(
            request_id=proposal.request_id,
            step_before=proposal.step_before,
            step_after=step_after,
            steps_advanced=1,
            current_code=proposal.sampled.reshape(1),
            codes_after=codes_after,
            codec_delta=codec_delta,
            reached_limit=reached_limit,
            is_eos=is_eos,
            finished=finished,
            terminal=torch.tensor(finished, dtype=torch.bool),
        )

    def _cached_stop_row(self, hidden: torch.Tensor, *, finished: bool) -> torch.Tensor:
        """Return the immutable binary stop row for the current device/dtype."""
        row_attr = "_finished_stop_row" if finished else "_continue_stop_row"
        row = getattr(self, row_attr)
        if row is None or row.device != hidden.device or row.dtype != hidden.dtype:
            row = hidden.new_tensor(
                [float("-inf"), 0.0] if finished else [0.0, float("-inf")]
            )
            setattr(self, row_attr, row)
        return row

    def _proposal_stop_row(
        self,
        proposal: _CodecProposal,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Build the binary stop logits without reading the codec token on CPU."""
        step_after = proposal.step_before + 1
        if step_after >= proposal.max_tokens:
            return self._cached_stop_row(hidden, finished=True)
        if proposal.step_before < proposal.min_tokens:
            return self._cached_stop_row(hidden, finished=False)
        is_eos = proposal.sampled == self._num_audio_tokens - 1
        return torch.where(
            is_eos,
            self._cached_stop_row(hidden, finished=True),
            self._cached_stop_row(hidden, finished=False),
        )

    def finalize_deferred_codec_output(
        self,
        multimodal_outputs: dict[str, Any],
        sampled_stop_token_ids: list[list[int]],
    ) -> dict[str, Any]:
        """Commit pending codec proposals after the runner's normal D2H.

        ``sampled_stop_token_ids`` is the already synchronized result produced
        by ``NPUARModelRunner._bookkeeping_sync``.  No codec tensor is read on
        CPU here, so codec EOS and scheduler stop bookkeeping share the same
        synchronization point while codec/RNG/history semantics remain K=1.
        """
        pending = self._deferred_codec_batch
        if pending is None:
            return multimodal_outputs
        # Clear first: any failure abandons this request turn rather than
        # accidentally committing the same random draw on a later runner step.
        self._deferred_codec_batch = None
        if multimodal_outputs is not pending.multimodal_outputs:
            raise RuntimeError("MiniCPM-o Talker deferred output identity changed before commit")

        codes_output = multimodal_outputs["codes"]["audio"]
        meta_output = multimodal_outputs["meta"]
        terminal_output = meta_output["finished"]
        sparse_req_ids = meta_output.get("req_id") if pending.sparse_output else None

        for entry in pending.entries:
            if entry.output_index >= len(sampled_stop_token_ids):
                raise RuntimeError("MiniCPM-o Talker deferred stop batch is shorter than the request batch")
            sampled_ids = sampled_stop_token_ids[entry.output_index]
            if len(sampled_ids) != 1:
                raise RuntimeError(
                    "MiniCPM-o Talker deferred commit requires one valid runner "
                    f"sample, got {sampled_ids!r}"
                )
            stop_token = int(sampled_ids[0])
            commit = self._resolve_codec_proposal(
                entry.proposal,
                empty_delta=pending.empty_delta,
                runner_stop_token=stop_token,
            )
            if commit.finished != (stop_token == 1):
                raise RuntimeError("MiniCPM-o Talker runner/model terminal decisions diverged")
            delta, terminal, sparse_delta = self._commit_codec_prefix(
                info=entry.info,
                state=entry.state,
                commit=commit,
                sparse_output=pending.sparse_output,
                empty_delta=pending.empty_delta,
            )
            if pending.sparse_output:
                if sparse_delta is not None:
                    assert isinstance(sparse_req_ids, list)
                    sparse_req_ids.append(commit.request_id)
                    codes_output.append(sparse_delta)
                    terminal_output.append(terminal)
            else:
                codes_output[entry.output_index] = delta
                terminal_output[entry.output_index] = terminal
        self._deferred_codec_commit_total += len(pending.entries)
        if (
            self._deferred_codec_commit_total == len(pending.entries)
            or self._deferred_codec_commit_total % 1024 == 0
        ):
            logger.info(
                "MINICPMO45_DEFERRED_CODEC_COMMIT event=commit total=%d sparse=%s",
                self._deferred_codec_commit_total,
                pending.sparse_output,
            )
        return multimodal_outputs

    def _commit_codec_prefix(
        self,
        *,
        info: dict[str, Any],
        state: dict[str, Any],
        commit: _CodecCommitState,
        sparse_output: bool,
        empty_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Commit exactly once after all possibly failing payload work succeeds."""
        if str(info.get("request_id", commit.request_id)) != commit.request_id:
            raise RuntimeError("MiniCPM-o Talker codec commit request mismatch")
        if int(state.get("step", 0)) != commit.step_before or state.get("finished"):
            raise RuntimeError("MiniCPM-o Talker codec commit is stale or duplicated")
        if (
            commit.step_before < 0
            or commit.steps_advanced != 1
            or commit.step_after != commit.step_before + 1
        ):
            raise RuntimeError("MiniCPM-o Talker codec commit has invalid step accounting")
        if commit.finished != (commit.is_eos or commit.reached_limit):
            raise RuntimeError("MiniCPM-o Talker codec commit stop flags are inconsistent")
        if (
            commit.current_code.shape != (1,)
            or commit.current_code.dtype != torch.long
            or commit.codes_after.ndim != 1
            or commit.codes_after.dtype != torch.long
            or commit.codec_delta.ndim != 2
            or commit.codec_delta.shape[1:] != (1,)
            or commit.codec_delta.dtype != torch.long
            or commit.current_code.device != commit.codes_after.device
            or commit.codec_delta.device != commit.codes_after.device
            or empty_delta.shape != (0, 1)
            or empty_delta.dtype != torch.long
            or empty_delta.device != commit.codes_after.device
        ):
            raise RuntimeError("MiniCPM-o Talker codec commit has an invalid tensor contract")
        if commit.codec_delta.shape[0] != (0 if commit.finished else 1):
            raise RuntimeError("MiniCPM-o Talker codec commit prefix is invalid")
        if (
            commit.terminal.shape != ()
            or commit.terminal.dtype != torch.bool
            or commit.terminal.device.type != "cpu"
            or bool(commit.terminal.item()) != commit.finished
        ):
            raise RuntimeError("MiniCPM-o Talker codec terminal flag is invalid")

        pending_value = state.get("sparse_pending_codec_deltas")
        if pending_value is not None and not isinstance(pending_value, list):
            raise RuntimeError("MiniCPM-o Talker sparse pending state is not a list")
        pending = pending_value
        result_delta = commit.codec_delta
        sparse_delta: torch.Tensor | None = None
        sparse_emits_after: int | None = None
        sparse_suppressed_after: int | None = None
        total_emits_after: int | None = None
        total_suppressed_after: int | None = None

        if sparse_output:
            pending_len = len(pending) if pending is not None else 0
            delta_rows = int(commit.codec_delta.shape[0])
            if commit.finished or pending_len + delta_rows >= self._sparse_chunk_frames:
                pending_rows = list(pending) if pending is not None else []
                if delta_rows:
                    pending_rows.append(commit.codec_delta)
                sparse_delta = (
                    torch.cat(pending_rows, dim=0) if pending_rows else empty_delta
                )
                sparse_emits_after = int(state.get("sparse_emits", 0)) + 1
                total_emits_after = self._sparse_emit_total + 1
            else:
                sparse_suppressed_after = int(state.get("sparse_suppressed", 0)) + 1
                total_suppressed_after = self._sparse_suppressed_total + 1
        elif pending:
            pending_rows = [row for row in pending if isinstance(row, torch.Tensor)]
            if commit.codec_delta.numel():
                pending_rows.append(commit.codec_delta)
            result_delta = (
                torch.cat(pending_rows, dim=0) if pending_rows else empty_delta
            )

        state["step"] = commit.step_after
        state["history_version"] = int(state.get("history_version", 0)) + 1
        state["finished"] = commit.finished
        state["codes"] = commit.codes_after
        info["audio_state"] = state
        info["audio_codes"] = {
            "current": commit.current_code,
            "accumulated": commit.codes_after,
        }
        if commit.finished and getattr(self, "_codec_greedy_enabled", False):
            logger.info(
                "MINICPMO45_CODEC_GREEDY event=finish request_id=%s "
                "steps=%d reason=%s",
                commit.request_id,
                commit.step_after,
                "eos" if commit.is_eos else "max_tokens",
            )

        if sparse_output:
            if pending is None:
                pending = []
                state["sparse_pending_codec_deltas"] = pending
            if sparse_delta is not None:
                pending.clear()
                assert sparse_emits_after is not None and total_emits_after is not None
                state["sparse_emits"] = sparse_emits_after
                self._sparse_emit_total = total_emits_after
                if self._sparse_emit_total == 1 or self._sparse_emit_total % 64 == 0:
                    logger.info(
                        "MINICPMO45_SPARSE_CHUNK event=emit emits=%d suppressed=%d rows=%d terminal=%s",
                        self._sparse_emit_total,
                        self._sparse_suppressed_total,
                        int(sparse_delta.shape[0]),
                        commit.finished,
                    )
            else:
                if commit.codec_delta.numel():
                    pending.append(commit.codec_delta)
                assert (
                    sparse_suppressed_after is not None
                    and total_suppressed_after is not None
                )
                state["sparse_suppressed"] = sparse_suppressed_after
                self._sparse_suppressed_total = total_suppressed_after
                if (
                    self._sparse_suppressed_total == 1
                    or self._sparse_suppressed_total % 1024 == 0
                ):
                    logger.info(
                        "MINICPMO45_SPARSE_CHUNK event=suppress emits=%d suppressed=%d",
                        self._sparse_emit_total,
                        self._sparse_suppressed_total,
                    )
        elif pending:
            pending.clear()
        return result_delta, commit.terminal, sparse_delta

    def _can_start_codec_eos_batch(
        self,
        *,
        proposal: _CodecProposal,
        state: dict[str, Any],
        sparse_output: bool,
        known_controller_bypass: bool,
    ) -> bool:
        """Fail-closed gate for a K-row delayed-EOS transaction."""

        k = self._codec_eos_batch_k
        if (
            k not in (2, 4)
            or self._codec_eos_batch is not None
            or not sparse_output
            or not known_controller_bypass
            or proposal.step_before < proposal.min_tokens
            or proposal.step_before + k >= proposal.max_tokens
            # K=4 is deliberately competition-only and validated against the
            # real Stage1 sparse handoff contract.  Stage2 CCF50 is a separate
            # aggregation boundary; the Talker itself publishes every 25 rows.
            or (k == 4 and self._sparse_chunk_frames != 25)
        ):
            return False
        pending = state.get("sparse_pending_codec_deltas")
        if pending is None:
            pending_len = 0
        elif isinstance(pending, list):
            pending_len = len(pending)
        else:
            return False
        # No row may publish a payload before the joint EOS decision.
        return pending_len + k < self._sparse_chunk_frames

    def _rollback_codec_eos_batch_suffix(
        self,
        *,
        batch: _CodecEosBatch,
        first_eos: int,
        info: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        """Undo optimistic rows starting at the first EOS.

        Rows before ``first_eos`` remain accepted.  Their request-visible
        history already exists in ``batch.proposals[first_eos].codes_before``;
        retaining it avoids replaying Python commits or duplicating sparse
        telemetry.
        """

        optimistic_rows = batch.k - 1
        if not 0 <= first_eos < optimistic_rows:
            raise RuntimeError("MiniCPM-o EOS batch rollback index is invalid")
        if (
            int(state.get("step", -1)) != batch.step_before + optimistic_rows
            or int(state.get("history_version", -1))
            != batch.history_version_before + optimistic_rows
            or int(state.get("sparse_suppressed", -1))
            != batch.sparse_suppressed_before + optimistic_rows
            or self._sparse_suppressed_total
            != batch.sparse_suppressed_total_before + optimistic_rows
            or state.get("finished")
        ):
            raise RuntimeError(
                "MiniCPM-o EOS batch lost its optimistic private state"
            )
        pending = state.get("sparse_pending_codec_deltas")
        if not isinstance(pending, list) or len(pending) != (
            batch.sparse_pending_len_before + optimistic_rows
        ):
            raise RuntimeError(
                "MiniCPM-o EOS batch lost its private sparse payload"
            )
        del pending[batch.sparse_pending_len_before + first_eos :]
        if first_eos == 0:
            codes = batch.codes_before
            current = batch.current_code_before
        else:
            codes = batch.proposals[first_eos].codes_before
            if codes.numel() == 0:
                raise RuntimeError(
                    "MiniCPM-o EOS batch accepted prefix lost its current code"
                )
            current = codes[-1:]
        state["step"] = batch.step_before + first_eos
        state["history_version"] = batch.history_version_before + first_eos
        state["finished"] = False
        state["codes"] = codes
        state["sparse_suppressed"] = (
            batch.sparse_suppressed_before + first_eos
        )
        self._sparse_suppressed_total = (
            batch.sparse_suppressed_total_before + first_eos
        )
        info["audio_state"] = state
        info["audio_codes"] = {
            "current": current,
            "accumulated": codes,
        }

    def _try_codec_eos_batch_step(
        self,
        *,
        info: dict[str, Any],
        state: dict[str, Any],
        proposal: _CodecProposal,
        sparse_output: bool,
        known_controller_bypass: bool,
        empty_delta: torch.Tensor,
    ) -> tuple[_CodecCommitState, torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        """Commit one ordinary row while resolving EOS once per K rows.

        This is not a multi-token model graph: the scheduler still owns every
        q_len=1 row and therefore every physical KV slot.  Only the scalar EOS
        synchronization is batched.  Any contract loss after the first draw is
        fatal; there is no retry or eager fallback inside a transaction.
        """

        batch = self._codec_eos_batch
        if batch is None:
            if not self._can_start_codec_eos_batch(
                proposal=proposal,
                state=state,
                sparse_output=sparse_output,
                known_controller_bypass=known_controller_bypass,
            ):
                return None
            pending = state.get("sparse_pending_codec_deltas")
            pending_len = len(pending) if isinstance(pending, list) else 0
            current = (info.get("audio_codes", {}) or {}).get("current")
            if not isinstance(current, torch.Tensor) or current.shape != (1,):
                raise RuntimeError(
                    "MiniCPM-o EOS batch requires one request-local current code"
                )
            batch = _CodecEosBatch(
                k=self._codec_eos_batch_k,
                request_id=proposal.request_id,
                step_before=proposal.step_before,
                history_version_before=int(state.get("history_version", 0)),
                sparse_pending_len_before=pending_len,
                sparse_suppressed_before=int(state.get("sparse_suppressed", 0)),
                sparse_suppressed_total_before=self._sparse_suppressed_total,
                codes_before=proposal.codes_before,
                current_code_before=current,
                proposals=[],
            )
            self._codec_eos_batch = batch
        elif (
            batch.request_id != proposal.request_id
            or len(batch.proposals) >= batch.k
            or proposal.step_before
            != batch.step_before + len(batch.proposals)
            or proposal.min_tokens != batch.proposals[0].min_tokens
            or proposal.max_tokens != batch.proposals[0].max_tokens
            or not sparse_output
            or not known_controller_bypass
        ):
            raise RuntimeError(
                "MiniCPM-o EOS batch transaction lost its strict C=1 continuation"
            )

        row = len(batch.proposals)
        if not 0 <= row < batch.k:
            raise RuntimeError("MiniCPM-o EOS batch exceeded its K boundary")
        self._codec_eos_batch_samples[row].copy_(proposal.sampled.reshape(()))
        batch.proposals.append(proposal)
        self._codec_eos_batch_steps += 1

        if row < batch.k - 1:
            commit = self._resolve_codec_proposal(
                proposal,
                empty_delta=empty_delta,
                runner_stop_token=0,
            )
            delta, terminal, sparse_delta = self._commit_codec_prefix(
                info=info,
                state=state,
                commit=commit,
                sparse_output=True,
                empty_delta=empty_delta,
            )
            if commit.finished or sparse_delta is not None:
                raise RuntimeError(
                    "MiniCPM-o EOS batch optimistic row escaped its private boundary"
                )
            return commit, delta, terminal, sparse_delta

        sampled = [
            int(token)
            for token in self._codec_eos_batch_samples[: batch.k]
            .detach()
            .to(device="cpu")
            .tolist()
        ]
        eos_id = self._num_audio_tokens - 1
        first_eos = next(
            (index for index, token in enumerate(sampled) if token == eos_id),
            None,
        )
        self._codec_eos_batch = None
        self._codec_eos_batch_boundaries += 1

        if first_eos is None:
            self._codec_eos_batch_no_eos += 1
            commit_index = batch.k - 1
            stop_token = 0
        else:
            self._codec_eos_batch_terminal_rows[first_eos] += 1
            commit_index = first_eos
            stop_token = 1
        if first_eos is not None and first_eos < batch.k - 1:
            self._codec_eos_batch_rollbacks += 1
            rolled_back = batch.k - 1 - first_eos
            self._codec_eos_batch_rolled_back_rows += rolled_back
            self._codec_eos_batch_wasted_suffix_rows += batch.k - first_eos - 1
            self._rollback_codec_eos_batch_suffix(
                batch=batch,
                first_eos=first_eos,
                info=info,
                state=state,
            )

        if commit_index == batch.k - 1:
            # Preserve the known-good K=2 lifetime contract for the final row.
            # Its live proposal owns the codec delta that may remain queued in
            # sparse_pending_codec_deltas across later boundaries.  Returning
            # a view into _codec_eos_batch_samples here makes old pending rows
            # change when that fixed buffer is reused, corrupting the audio.
            commit_proposal = proposal
        else:
            base = batch.proposals[commit_index]
            codes_before = state.get("codes")
            if not isinstance(codes_before, torch.Tensor):
                raise RuntimeError("MiniCPM-o EOS batch lost its commit history")
            # An early EOS row's original graph output may have been reused by
            # a later optimistic row.  Copy the authoritative fixed sample to
            # request-owned storage before publishing terminal state.
            commit_proposal = _CodecProposal(
                request_id=base.request_id,
                sampled=self._codec_eos_batch_samples[commit_index]
                .reshape(())
                .clone(),
                step_before=base.step_before,
                min_tokens=base.min_tokens,
                max_tokens=base.max_tokens,
                codes_before=codes_before,
            )

        commit = self._resolve_codec_proposal(
            commit_proposal,
            empty_delta=empty_delta,
            runner_stop_token=stop_token,
        )
        delta, terminal, sparse_delta = self._commit_codec_prefix(
            info=info,
            state=state,
            commit=commit,
            sparse_output=True,
            empty_delta=empty_delta,
        )
        if (
            self._codec_eos_batch_boundaries == 1
            or self._codec_eos_batch_boundaries % 512 == 0
            or commit.finished
        ):
            logger.info(
                "MINICPMO45_STAGE1_EOS_BATCH event=resolve k=%d boundaries=%d "
                "steps=%d first_eos=%s terminal_rows=%s rollbacks=%d "
                "rolled_back_rows=%d wasted_suffix_rows=%d request_id=%s",
                batch.k,
                self._codec_eos_batch_boundaries,
                self._codec_eos_batch_steps,
                first_eos,
                ",".join(
                    str(value)
                    for value in self._codec_eos_batch_terminal_rows[: batch.k]
                ),
                self._codec_eos_batch_rollbacks,
                self._codec_eos_batch_rolled_back_rows,
                self._codec_eos_batch_wasted_suffix_rows,
                proposal.request_id,
            )
        return commit, delta, terminal, sparse_delta

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        # A hint belongs to exactly one forward/sample pair.  Clear before
        # passthrough, invalid rows and every normal forward so a stale hint
        # cannot be consumed by another request or step.
        self._known_controller_hint = None
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        spans = kwargs.get("request_token_spans")
        if spans is None or len(spans) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker requires one request_token_span per request")
        sample_eligible = kwargs.get("request_sample_eligible")
        if sample_eligible is None:
            sample_eligible = [True] * len(infos)
        if len(sample_eligible) != len(infos):
            raise RuntimeError(
                f"MiniCPM-o continuous Talker received {len(sample_eligible)} sampling flags for {len(infos)} requests"
            )
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)
        # Sparse handoff is a C=1 non-duplex fast path. Unsupported runtime
        # shapes must remain valid serving inputs, so fall back to dense output
        # and flush any rows accumulated while the request was sparse.
        sparse_output = (
            self._sparse_chunk_frames > 0
            and len(infos) == 1
            and not emit_duplex_metadata
        )
        # Native duplex changes the two-way controller logits after this model
        # hook.  It therefore cannot reuse the ordinary runner stop token as
        # the codec EOS decision; retain the immediate commit path.
        defer_codec_commit = bool(kwargs.get("defer_codec_commit", False)) and not emit_duplex_metadata
        known_controller_bypass = bool(
            kwargs.get("known_controller_bypass", False)
        ) and not emit_duplex_metadata
        if defer_codec_commit and self._deferred_codec_batch is not None:
            raise RuntimeError("MiniCPM-o Talker has an unconsumed deferred codec batch")
        deferred_entries: list[_DeferredCodecEntry] = []
        sparse_req_ids: list[str] = []
        sparse_codec_deltas: list[torch.Tensor] = []
        sparse_terminal_flags: list[torch.Tensor] = []

        stop_rows: list[torch.Tensor] = []
        codec_deltas: list[torch.Tensor] = []
        terminal_flags: list[torch.Tensor] = []
        native_duplex_flags: list[torch.Tensor] = []
        duplex_epochs: list[torch.Tensor] = []
        duplex_turn_ids: list[torch.Tensor] = []
        segment_texts_utf8: list[torch.Tensor] = []
        turn_end_flags: list[torch.Tensor] = []
        empty_delta = hidden.new_empty((0, 1), dtype=torch.long)
        for index, info in enumerate(infos):
            info_dict = info if isinstance(info, dict) else {}
            native_duplex = info_dict.get("native_duplex") is True
            if emit_duplex_metadata:
                duplex_info = info_dict.get("duplex")
                if not isinstance(duplex_info, dict):
                    duplex_info = {}
                epoch = duplex_info.get("epoch", -1)
                turn_id = duplex_info.get("turn_id", -1)
                if native_duplex and not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (epoch, turn_id)
                ):
                    raise RuntimeError(
                        "MiniCPM-o native duplex Talker requires non-negative integer "
                        f"epoch and turn_id, got epoch={epoch!r}, turn_id={turn_id!r}"
                    )
                meta_info = info_dict.get("meta")
                if not isinstance(meta_info, dict):
                    meta_info = {}
                segment_text = meta_info.get("native_duplex_segment_text", "") if native_duplex else ""
                if not isinstance(segment_text, str):
                    segment_text = ""
                turn_eos_id = meta_info.get("turn_eos_token_id")
                ids_info = info_dict.get("ids")
                tts_ids = ids_info.get("tts") if native_duplex and isinstance(ids_info, dict) else None
                if isinstance(tts_ids, torch.Tensor):
                    contains_turn_eos = isinstance(turn_eos_id, int) and bool(
                        torch.any(tts_ids.reshape(-1) == turn_eos_id).item()
                    )
                elif isinstance(tts_ids, (list, tuple)):
                    contains_turn_eos = isinstance(turn_eos_id, int) and turn_eos_id in tts_ids
                else:
                    contains_turn_eos = False
                native_duplex_flags.append(torch.tensor(native_duplex, dtype=torch.bool))
                duplex_epochs.append(torch.tensor(epoch if isinstance(epoch, int) else -1, dtype=torch.long))
                duplex_turn_ids.append(torch.tensor(turn_id if isinstance(turn_id, int) else -1, dtype=torch.long))
                segment_texts_utf8.append(
                    torch.tensor(
                        list(segment_text.encode("utf-8")),
                        dtype=torch.uint8,
                    )
                )
                turn_end_flags.append(torch.tensor(native_duplex and contains_turn_eos, dtype=torch.bool))

            if not isinstance(info, dict):
                stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                continue
            start, end = spans[index]
            end = min(int(end), int(hidden.shape[0]))
            if int(start) >= end:
                stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                continue
            request_id = str(info.get("request_id", index))
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            state = request_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                request_states[request_id] = state
            if state.get("finished"):
                stop_rows.append(hidden.new_tensor([float("-inf"), 0.0]))
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                continue
            if not sample_eligible[index]:
                # vLLM computes a logit row for incomplete chunked prefills but
                # discards its sampled token. Advancing codec/RNG state here
                # would make output depend on prefill chunking and compaction.
                stop_rows.append(hidden.new_tensor([0.0, float("-inf")]))
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                continue
            codes = state.get("codes")
            if not isinstance(codes, torch.Tensor):
                codes = (info.get("audio_codes", {}) or {}).get("accumulated")
            if not isinstance(codes, torch.Tensor):
                codes = torch.empty(0, dtype=torch.long, device=hidden.device)
            else:
                codes = codes.to(device=hidden.device, dtype=torch.long).reshape(-1)
            step = int(state.get("step", 0))
            proposal = self._propose_codec(
                hidden[end - 1 : end],
                codes,
                request_id=request_id,
                step=step,
                min_tokens=int(state.get("min_tokens", self._codec_min_tokens)),
                max_tokens=int(state.get("max_tokens", 2048)),
                # Sparse output is already the strict C=1, non-duplex serving
                # gate.  All other requests retain the stock randomized path.
                greedy=getattr(self, "_codec_greedy_enabled", False)
                and sparse_output,
            )
            if defer_codec_commit:
                deferred_entries.append(
                    _DeferredCodecEntry(
                        output_index=index,
                        info=info,
                        state=state,
                        proposal=proposal,
                    )
                )
                # Placeholders are finalized after the runner's standard
                # sampled-token D2H.  Dense output remains request aligned;
                # sparse output stays compact until an emit is committed.
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                stop_rows.append(self._proposal_stop_row(proposal, hidden))
                continue
            eos_batch_result = self._try_codec_eos_batch_step(
                info=info,
                state=state,
                proposal=proposal,
                sparse_output=sparse_output,
                known_controller_bypass=known_controller_bypass,
                empty_delta=empty_delta,
            )
            if eos_batch_result is None:
                commit = self._resolve_codec_proposal(
                    proposal,
                    empty_delta=empty_delta,
                )
                delta, terminal, sparse_delta = self._commit_codec_prefix(
                    info=info,
                    state=state,
                    commit=commit,
                    sparse_output=sparse_output,
                    empty_delta=empty_delta,
                )
            else:
                commit, delta, terminal, sparse_delta = eos_batch_result
            codec_deltas.append(delta)
            terminal_flags.append(terminal)
            if sparse_output and sparse_delta is not None:
                sparse_req_ids.append(request_id)
                sparse_codec_deltas.append(sparse_delta)
                sparse_terminal_flags.append(terminal)
            if known_controller_bypass and len(infos) == 1 and not commit.finished:
                # The stock commit is authoritative.  Every nonterminal row
                # maps to controller token 0 both before and after min_tokens;
                # terminal remains on the stock sampler and D2H path.
                self._known_controller_hint = _KnownControllerHint(
                    request_id=request_id,
                    step_before=proposal.step_before,
                    step_after=proposal.step_before + 1,
                    token=0,
                )
            stop_rows.append(self._cached_stop_row(hidden, finished=commit.finished))

        if len(stop_rows) == 1:
            # unsqueeze is a view; torch.stack would launch a copy for the
            # overwhelmingly common single-concurrency serving path.
            self._batch_stop_logits = stop_rows[0].unsqueeze(0)
        elif stop_rows:
            self._batch_stop_logits = torch.stack(stop_rows, dim=0)
        else:
            self._batch_stop_logits = hidden.new_empty((0, 2))
        # Lists are deliberate: the runner routes element i to request i,
        # preserving compaction alignment while emitting only this step's code.
        if sparse_output:
            codec_deltas = sparse_codec_deltas
            terminal_flags = sparse_terminal_flags
        meta_outputs = {"finished": terminal_flags}
        if sparse_output:
            # Reuse the existing VoxCPM2 sparse multimodal contract. req_id is
            # compact and aligned with the compact payload lists; an empty list
            # explicitly means this step has no downstream payload.
            meta_outputs.update(
                {
                    "req_id": sparse_req_ids,
                    "sparse_audio": ["1"],
                }
            )
        if emit_duplex_metadata:
            meta_outputs.update(
                {
                    "native_duplex": native_duplex_flags,
                    "duplex_epoch": duplex_epochs,
                    "duplex_turn_id": duplex_turn_ids,
                    "llm_output_text_utf8": segment_texts_utf8,
                    "turn_end": turn_end_flags,
                }
            )
        multimodal_outputs: dict[str, Any] = {
            "codes": {"audio": codec_deltas},
            "meta": meta_outputs,
        }
        if deferred_entries:
            self._deferred_codec_batch = _DeferredCodecBatch(
                multimodal_outputs=multimodal_outputs,
                entries=tuple(deferred_entries),
                empty_delta=empty_delta,
                sparse_output=sparse_output,
            )
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        finished_ids = {str(req_id) for req_id in finished_req_ids}
        hint = self._known_controller_hint
        if hint is not None and hint.request_id in finished_ids:
            self._known_controller_hint = None
        pending = self._deferred_codec_batch
        if pending is not None and any(
            entry.proposal.request_id in finished_ids for entry in pending.entries
        ):
            # Abort/error cleanup must not let a proposal from an earlier turn
            # be committed by a later request's sampled controller token.
            self._deferred_codec_batch = None
        eos_batch = self._codec_eos_batch
        if eos_batch is not None and eos_batch.request_id in finished_ids:
            # Abort/error owns the whole request.  No retry is legal after the
            # first random draw, so discard the private transaction with it.
            self._codec_eos_batch = None
            self._codec_eos_batch_aborts += 1
            logger.warning(
                "MINICPMO45_STAGE1_EOS_BATCH event=abort_open k=%d "
                "rows=%d aborts=%d request_id=%s",
                eos_batch.k,
                len(eos_batch.proposals),
                self._codec_eos_batch_aborts,
                eos_batch.request_id,
            )
        if self._sparse_chunk_frames:
            pending_rows = sum(
                len(state.get("sparse_pending_codec_deltas", []))
                for req_id, state in self._request_audio_states.items()
                if req_id in finished_req_ids and isinstance(state, dict)
            )
            log_method = logger.error if pending_rows else logger.info
            log_method(
                "MINICPMO45_SPARSE_CHUNK event=finish emits=%d suppressed=%d finished=%d pending_rows=%d",
                self._sparse_emit_total,
                self._sparse_suppressed_total,
                len(finished_req_ids),
                pending_rows,
            )
        self._deferred_cleanup_ids.update(finished_ids)

    def _flush_deferred_cleanup(self) -> None:
        request_audio_states = getattr(self, "_request_audio_states", {})
        for request_id in self._deferred_cleanup_ids:
            self._request_generators.pop(request_id, None)
            request_audio_states.pop(request_id, None)
        self._deferred_cleanup_ids.clear()

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self._flush_deferred_cleanup()
        if input_ids is None and inputs_embeds is None:
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)
        if (
            self._codec_embed_graph_enabled
            and input_ids is not None
            and inputs_embeds is not None
            and input_ids.shape[0] == 1
            and inputs_embeds.shape[0] == 1
        ):
            # Startup capture and every replay keep the same two Tensor
            # inputs.  On a rejected/eager row the selector is false, so
            # use index zero for the otherwise-dead lookup and return the
            # exact stock embedding.  On the strict C=1 lane input_ids is
            # the authoritative codec id copied into stable int32 storage.
            safe_ids = torch.where(
                self._codec_embed_graph_active,
                input_ids,
                torch.zeros_like(input_ids),
            )
            codec_embeds = self.emb_code[0](safe_ids)
            inputs_embeds = torch.where(
                self._codec_embed_graph_active.reshape(1, 1),
                codec_embeds,
                inputs_embeds,
            )
            input_ids = None
        return self.tts_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states, *args, **kwargs):
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if self._batch_stop_logits is None:
            return torch.zeros(
                hidden_states.shape[0],
                2,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        logits = self._batch_stop_logits
        self._batch_stop_logits = None
        return logits

    def sample(self, logits, sampling_metadata):
        # The model emits a binary row containing exactly one finite value, so
        # the standard sampler's temperature/processor/top-k machinery cannot
        # change the selected token. The runner has already applied its logit
        # bias (including caller-specified min_tokens), hence a direct argmax
        # preserves serving semantics while avoiding the generic sampler path.
        if (
            sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logprob_token_ids
        ):
            sampled = logits.argmax(dim=-1).to(torch.int32).unsqueeze(-1)
            return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)
        # The C=1 row is a view of a process-lifetime cached constant. Generic
        # sampler processors may update logits in place, so isolate fallback
        # requests (for example logprobs) from that cache.
        return self._fallback_stop_sampler(logits.clone(), sampling_metadata)

    @staticmethod
    def can_skip_model_sampler_output_token_history(sampling_metadata: Any) -> bool:
        """Whether the binary controller fast sampler ignores token history."""

        return bool(
            sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logprob_token_ids
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return self._load_native_weights(weights)

    def _load_native_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g = head_v = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name[len("tts.") :]
            if stripped.startswith("model."):
                backbone_weights.append((stripped[len("model.") :], tensor))
                continue
            if stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            if stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            target = stripped
            parameter = direct_params.get(target)
            if parameter is None:
                continue
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(target)

        for name in self.tts_model.load_weights(backbone_weights):
            loaded.add(f"tts_model.{name}")

        if head_g is None or head_v is None:
            raise ValueError("MiniCPM-o checkpoint is missing weight-norm Talker head parameters")
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code[0].weight.data.copy_(
            restored.to(
                device=self.head_code[0].weight.device,
                dtype=self.head_code[0].weight.dtype,
            )
        )
        loaded.add("head_code.0.weight")
        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
