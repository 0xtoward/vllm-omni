# SPDX-License-Identifier: Apache-2.0
"""Opt-in MiniCPM-o 4.5 Stage1 fixed192 FIA task-update candidate.

This is conditional scaffolding, not a promoted optimization. It must remain
off unless the task-persistence Gate0 passes on the exact runtime. The plugin
keeps the stock KV writer and one existing FULL graph; it only changes how the
20 captured FIA task descriptors are prepared and released.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.platforms.npu.stage1_fia_fixed192_state import (
    FIXED_CAPACITY,
    Fixed192StateMachine,
    Observation,
    UpdateAction,
    build_mask_row,
)
from vllm_omni.platforms.npu.stage1_fia_prefix_barrier_state import (
    PrefixBarrierContractError,
    PrefixBarrierState,
)

_ENV = "VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192"
_PREFIX_BARRIER_ENV = "VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0"
_INSTALL_MARKER = "_vllm_omni_stage1_fia_fixed192_install_v1"
_CAPTURE_MARKER = "_vllm_omni_stage1_fia_fixed192_capture_v1"
_UPDATE_MARKER = "_vllm_omni_stage1_fia_fixed192_update_v1"
_RUNNER_MARKER = "_vllm_omni_stage1_fia_fixed192_runner_v1"
_EXPECTED_ARCH = "MiniCPMO45OmniForConditionalGeneration"
_EXPECTED_MAX_MODEL_LEN = 4096
_EXPECTED_TASKS = 20


def _enabled() -> bool:
    return os.environ.get(_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _mode_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _prefix_barrier_gate0_enabled() -> bool:
    return os.environ.get(_PREFIX_BARRIER_ENV, "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


@dataclass
class _Runtime:
    torch: Any
    device: Any
    mask_bank: Any
    target_mask: Any
    target_table: Any
    weak_ref_tensors: Any = None
    state: Fixed192StateMachine = field(default_factory=Fixed192StateMachine)
    counters: Counter[str] = field(default_factory=Counter)
    # FULL graph capture is driven by generic dummy decoder metadata.  Keep
    # those exact per-layer mask/sparse fields alive so a request that falls
    # outside the fixed192 contract can reconstruct the stock task update.
    stock_capture_descriptors: list[tuple[Any | None, int, int, int]] = field(
        default_factory=list
    )
    last_log_total: int = 0
    prefix_barrier_state: PrefixBarrierState = field(
        default_factory=PrefixBarrierState
    )
    prefix_barrier_event: Any | None = None
    prefix_barrier_proxies: list[Any] = field(default_factory=list)

    @classmethod
    def create(cls, torch: Any, device: Any) -> _Runtime:
        rows = [bytes([1]) * 256]
        rows.extend(
            build_mask_row(live_len) for live_len in range(1, FIXED_CAPACITY + 1)
        )
        mask_bank = torch.tensor(
            [list(row) for row in rows], dtype=torch.int8, device=device
        ).view(FIXED_CAPACITY + 1, 1, 1, 256)
        target_mask = torch.empty((1, 1, 256), dtype=torch.int8, device=device)
        target_table = torch.zeros((1, 2), dtype=torch.int32, device=device)
        target_mask.copy_(mask_bank[1])
        torch.npu.synchronize()
        return cls(torch, device, mask_bank, target_mask, target_table)


_RUNTIME: _Runtime | None = None


def _is_stage1_runner(runner: Any) -> tuple[bool, str]:
    model_config = getattr(runner, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    parallel = getattr(runner, "parallel_config", None)
    compilation = getattr(runner, "compilation_config", None)
    scheduler = getattr(getattr(runner, "vllm_config", None), "scheduler_config", None)
    if getattr(model_config, "_architecture", None) != _EXPECTED_ARCH:
        return False, "architecture"
    # The outer MiniCPM architecture is shared by Stage0 and Stage1.  The
    # model_stage field is the process-local pipeline identity used by the
    # existing NPU AR runner fast paths; without this guard the plugin could
    # allocate its runtime in the Stage0 worker as well.
    if getattr(model_config, "model_stage", None) != "tts":
        return False, "not_stage1_tts"
    if str(getattr(hf_config, "version", "")) != "4.5":
        return False, "model_version"
    if getattr(model_config, "max_model_len", None) != _EXPECTED_MAX_MODEL_LEN:
        return False, "max_model_len"
    if any(
        int(getattr(parallel, name, 1)) != 1
        for name in (
            "tensor_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        return False, "parallelism"
    if int(getattr(scheduler, "max_num_seqs", getattr(runner, "max_num_reqs", 0))) != 1:
        return False, "not_strict_c1"
    if getattr(runner, "enable_enpu", False):
        return False, "enpu"
    if getattr(runner, "speculative_config", None) is not None:
        return False, "speculative"
    if getattr(runner, "use_sparse", False) or getattr(runner, "use_compress", False):
        return False, "sparse_or_compress"
    mode = _mode_name(getattr(compilation, "cudagraph_mode", ""))
    if "FULL" not in mode:
        return False, "not_full_graph"
    device = getattr(runner, "device", None)
    if getattr(device, "type", None) != "npu":
        return False, "device"
    return True, "eligible"


def _unique_metadata(forward_context: Any) -> list[Any]:
    metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(metadata, dict) or not metadata:
        return []
    return list({id(value): value for value in metadata.values()}.values())


def _observe(
    forward_context: Any,
    num_tokens: int,
    graph_params: Any,
) -> tuple[Observation, list[Any], Any | None]:
    runtime_mode = _mode_name(getattr(forward_context, "cudagraph_runtime_mode", ""))
    if runtime_mode != "FULL":
        return Observation(False, None, "runtime_not_full"), [], None
    if getattr(forward_context, "capturing", False):
        return Observation(False, None, "capturing"), [], None
    if num_tokens != 1:
        return Observation(False, None, "graph_key_not_one"), [], None
    if graph_params is None:
        return Observation(False, None, "missing_graph_params"), [], None
    try:
        params = graph_params.attn_params[num_tokens]
        handles = graph_params.handles[num_tokens]
        events = graph_params.events[num_tokens]
    except (AttributeError, KeyError, TypeError):
        return Observation(False, None, "missing_task_registry"), [], None
    if not (len(params) == len(handles) == len(events) == _EXPECTED_TASKS):
        return Observation(False, None, "task_count"), [], None

    unique = _unique_metadata(forward_context)
    if not unique:
        return Observation(False, None, "metadata"), [], None
    live_lengths: set[int] = set()
    source_table = None
    for metadata in unique:
        seq_lens = getattr(metadata, "seq_lens_list", None)
        table = getattr(metadata, "block_tables", None)
        if not bool(getattr(metadata, "causal", False)):
            return Observation(False, None, "stock_not_causal"), [], None
        if getattr(metadata, "attn_mask", None) is None:
            return Observation(False, None, "stock_mask_missing"), [], None
        if _mode_name(getattr(metadata, "attn_state", "")) != "DECODEONLY":
            return Observation(False, None, "not_decode_only"), [], None
        if not isinstance(seq_lens, list) or len(seq_lens) != 1:
            return Observation(False, None, "c_gt_1"), [], None
        if getattr(metadata, "actual_seq_lengths_q", None) != [1]:
            return Observation(False, None, "query_len"), [], None
        if table is None or getattr(table, "ndim", 0) != 2:
            return Observation(False, None, "block_table_rank"), [], None
        if int(table.shape[0]) != 1:
            return Observation(False, None, "block_table_shape"), [], None
        live_len = int(seq_lens[0])
        required_width = 1 if live_len <= 128 else 2
        if int(table.shape[1]) < required_width:
            return Observation(False, None, "block_table_width"), [], None
        live_lengths.add(live_len)
        if source_table is None:
            source_table = table
        elif int(source_table.data_ptr()) != int(table.data_ptr()):
            return Observation(False, None, "block_table_alias"), [], None
    if len(live_lengths) != 1:
        return Observation(False, None, "layer_length_mismatch"), [], None
    live_len = live_lengths.pop()
    reason = "eligible" if 1 <= live_len <= FIXED_CAPACITY else "length_gt_256"
    return Observation(True, live_len, reason), unique, source_table


def _copy_fixed_descriptors(
    runtime: _Runtime,
    update_stream: Any,
    source_table: Any,
    live_len: int,
) -> None:
    torch = runtime.torch
    with torch.npu.stream(update_stream):
        runtime.target_mask.copy_(runtime.mask_bank[live_len])
        if live_len <= 128:
            runtime.target_table[:, 0].copy_(source_table[:1, 0])
            runtime.target_table[:, 1].copy_(source_table[:1, 0])
        else:
            runtime.target_table.copy_(source_table[:1, :2])
    runtime.counters["descriptor.mask_copy"] += 1
    runtime.counters["descriptor.table_copy"] += 1


@contextmanager
def _fixed_metadata(unique: list[Any], runtime: _Runtime):
    saved: list[tuple[Any, list[int], Any, Any, bool]] = []
    for metadata in unique:
        saved.append(
            (
                metadata,
                metadata.seq_lens_list,
                metadata.block_tables,
                getattr(metadata, "attn_mask", None),
                bool(getattr(metadata, "causal", True)),
            )
        )
        metadata.seq_lens_list = [FIXED_CAPACITY]
        metadata.block_tables = runtime.target_table
        metadata.attn_mask = runtime.target_mask
        metadata.causal = False
    try:
        yield
    finally:
        for metadata, seq_lens, table, mask, causal in saved:
            metadata.seq_lens_list = seq_lens
            metadata.block_tables = table
            metadata.attn_mask = mask
            metadata.causal = causal


class _PrefixBarrierEventProxy:
    """A per-FIA proxy; only site zero contributes a graph wait/reset."""

    def __init__(self, runtime: _Runtime, site: int) -> None:
        self._runtime = runtime
        self._site = site

    def wait(self, stream: Any) -> Any:
        state = self._runtime.prefix_barrier_state
        if state.capture_wait(self._site):
            self._runtime.counters["prefix.capture_real_wait"] += 1
            return self._runtime.prefix_barrier_event.wait(stream)
        return None

    def reset(self, stream: Any) -> Any:
        state = self._runtime.prefix_barrier_state
        if state.capture_reset(self._site):
            self._runtime.counters["prefix.capture_real_reset"] += 1
            return self._runtime.prefix_barrier_event.reset(stream)
        return None

    def record(self, stream: Any) -> None:
        # Never release the real event here.  original_update must return and
        # the wrapper must validate all twenty sites before the one real record.
        self._runtime.prefix_barrier_state.observe_task_record(self._site)
        self._runtime.counters["prefix.task_record_observed"] += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime.prefix_barrier_event, name)


@contextmanager
def _prefix_capture_event(runtime: _Runtime):
    if not _prefix_barrier_gate0_enabled():
        yield
        return
    state = runtime.prefix_barrier_state
    site = state.allocate_capture_site()
    torch = runtime.torch
    original_factory = torch.npu.ExternalEvent
    if runtime.prefix_barrier_event is None:
        runtime.prefix_barrier_event = original_factory()
    proxy = _PrefixBarrierEventProxy(runtime, site)
    runtime.prefix_barrier_proxies.append(proxy)
    factory_calls = 0

    def proxy_factory(*args: Any, **kwargs: Any) -> Any:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls != 1:
            state.abort_capture(f"site {site} requested multiple ExternalEvents")
        return proxy

    torch.npu.ExternalEvent = proxy_factory
    try:
        yield
    except BaseException as exc:
        try:
            state.abort_capture(f"site {site}: {type(exc).__name__}: {exc}")
        except PrefixBarrierContractError as fatal:
            raise fatal from exc
    finally:
        torch.npu.ExternalEvent = original_factory
    if factory_calls != 1:
        state.abort_capture(f"site {site} ExternalEvent calls={factory_calls}")
    runtime.counters["prefix.capture_site"] += 1


def _assert_prefix_registry(runtime: _Runtime, graph_params: Any) -> None:
    state = runtime.prefix_barrier_state
    state.assert_capture_complete()
    try:
        events = graph_params.events[1]
    except (AttributeError, KeyError, TypeError) as exc:
        state.abort_capture(f"missing event registry: {exc}")
    if len(events) != _EXPECTED_TASKS:
        state.abort_capture(f"event registry size={len(events)}")
    if len({id(event) for event in events}) != _EXPECTED_TASKS:
        state.abort_capture("event proxies are not twenty distinct objects")
    for site, event in enumerate(events):
        if not isinstance(event, _PrefixBarrierEventProxy):
            state.abort_capture(f"event {site} is not a prefix proxy")
        if event._runtime is not runtime or event._site != site:
            state.abort_capture(f"event {site} proxy ownership/order mismatch")
    if events != runtime.prefix_barrier_proxies:
        state.abort_capture("event registry differs from capture proxy ledger")


@contextmanager
def _prefix_task_epoch(
    runtime: _Runtime,
    graph_params: Any,
    num_tokens: int,
    kind: str,
    update_stream: Any,
):
    if not _prefix_barrier_gate0_enabled() or num_tokens != 1:
        yield
        return
    _assert_prefix_registry(runtime, graph_params)
    state = runtime.prefix_barrier_state
    state.begin_task_epoch(kind)
    try:
        yield
    except BaseException as exc:
        runtime.counters["prefix.task_epoch_fatal"] += 1
        try:
            state.abort_task_epoch(f"{kind}: {type(exc).__name__}: {exc}")
        except PrefixBarrierContractError as fatal:
            raise fatal from exc
    try:
        state.prepare_task_release()
        with runtime.torch.npu.stream(update_stream):
            runtime.prefix_barrier_event.record(update_stream)
        state.commit_task_release()
    except BaseException as exc:
        runtime.counters["prefix.task_epoch_fatal"] += 1
        if not state.poisoned:
            try:
                state.abort_task_epoch(f"{kind} release: {type(exc).__name__}: {exc}")
            except PrefixBarrierContractError as fatal:
                raise fatal from exc
        raise RuntimeError("prefix-barrier task epoch failed after validation") from exc
    runtime.counters["prefix.task_epoch_success"] += 1
    runtime.counters["prefix.real_record"] += 1


def _record_only(runtime: _Runtime, update_stream: Any, graph_params: Any) -> None:
    torch = runtime.torch
    events = graph_params.events[1]
    if _prefix_barrier_gate0_enabled():
        _assert_prefix_registry(runtime, graph_params)
        state = runtime.prefix_barrier_state
        state.begin_record_only_release()
        try:
            with torch.npu.stream(update_stream):
                runtime.prefix_barrier_event.record(update_stream)
        except BaseException as exc:
            runtime.counters["prefix.record_only_fatal"] += 1
            try:
                state.abort_record_only_release(f"{type(exc).__name__}: {exc}")
            except PrefixBarrierContractError as fatal:
                raise fatal from exc
        state.commit_record_only_release()
        runtime.counters["event.record_only"] += 1
        runtime.counters["prefix.record_only_calls"] += 1
        runtime.counters["prefix.real_record"] += 1
        return
    with torch.npu.stream(update_stream):
        for event in events:
            event.record(update_stream)
    runtime.counters["event.record_only"] += len(events)


@contextmanager
def _stock_task_params(
    runtime: _Runtime,
    graph_params: Any,
    num_tokens: int,
    source_table: Any | None = None,
):
    """Temporarily restore stock causal FIA fields for a dynamic fallback."""

    params = graph_params.attn_params.get(num_tokens, [])
    descriptors = runtime.stock_capture_descriptors
    if num_tokens != 1:
        yield
        return
    if len(params) != _EXPECTED_TASKS or len(descriptors) != _EXPECTED_TASKS:
        raise RuntimeError(
            "fixed192 stock fallback registry is incomplete: "
            f"params={len(params)} descriptors={len(descriptors)}"
        )
    saved = list(params)
    try:
        weak_ref_tensors = runtime.weak_ref_tensors
        for index, (param, descriptor) in enumerate(zip(saved, descriptors)):
            if not isinstance(param, tuple) or len(param) != 21:
                raise RuntimeError("captured FIA tuple contract changed")
            stock_mask, stock_sparse, stock_pre, stock_next = descriptor
            fields = list(param)
            if source_table is not None:
                fields[3] = weak_ref_tensors(source_table)
            fields[4] = weak_ref_tensors(stock_mask) if stock_mask is not None else None
            fields[13] = stock_sparse
            fields[14] = stock_pre
            fields[15] = stock_next
            params[index] = tuple(fields)
        yield
    finally:
        params[:] = saved


def _maybe_log(
    runtime: _Runtime, logger: Any, live_len: int, action: UpdateAction
) -> None:
    total = sum(
        runtime.counters[key]
        for key in (
            "action.FIXED_REBIND",
            "action.FIXED_RECORD_ONLY",
            "action.STOCK_DYNAMIC",
        )
    )
    if total != 1 and total % 64 != 0:
        return
    logger.info(
        "MINICPMO45_STAGE1_FIA_FIXED192 action=%s live_len=%d calls=%d "
        "rebinds=%d record_only=%d stock=%d events_recorded=%d capture_rewrites=%d",
        action.value,
        live_len,
        total,
        runtime.counters["action.FIXED_REBIND"],
        runtime.counters["action.FIXED_RECORD_ONLY"],
        runtime.counters["action.STOCK_DYNAMIC"],
        runtime.counters["event.record_only"],
        runtime.counters["capture.rewrite"],
    )


def install_stage1_fia_fixed192_candidate() -> None:
    """Install the default-off, exact-contract service candidate."""

    if not _enabled():
        return
    try:
        import torch
        from vllm_ascend.attention import attention_v1 as attention_module
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        from vllm_ascend.compilation.acl_graph import get_graph_params
        from vllm_ascend.worker import model_runner_v1 as runner_module
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    except Exception:  # noqa: BLE001 - plugin is imported in non-NPU processes too
        return

    logger = runner_module.logger
    original_init = NPUModelRunner.__init__
    original_capture = AscendAttentionBackendImpl.full_graph_fia
    original_update = AscendAttentionBackendImpl.update_graph_params
    if getattr(original_init, _INSTALL_MARKER, False):
        return

    _require_source = {
        # update_stream belongs to the attention updater, not to every
        # NPUModelRunner.__init__ revision.  Requiring it here made the whole
        # plugin fail closed before the Stage1 eligibility check on the exact
        # judge runtime, even though wrapped_update's own source contract was
        # present and validated below.
        "runner_init": (original_init, ("enable_enpu",)),
        "full_graph_fia": (
            original_capture,
            ("graph_task_group_begin", "attn_params", "event.reset"),
        ),
        "update_graph_params": (
            original_update,
            ("graph_task_update_begin", "event.record", "attn_metadata"),
        ),
    }
    for label, (callable_obj, anchors) in _require_source.items():
        try:
            source = inspect.getsource(callable_obj)
        except (OSError, TypeError):
            logger.warning(
                "MiniCPMO45Stage1FIAFixed192 skipped: %s source unavailable", label
            )
            return
        missing = [anchor for anchor in anchors if anchor not in source]
        if missing:
            logger.warning(
                "MiniCPMO45Stage1FIAFixed192 skipped: %s missing anchors=%s",
                label,
                missing,
            )
            return

    @functools.wraps(original_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        global _RUNTIME
        original_init(self, *args, **kwargs)
        eligible, reason = _is_stage1_runner(self)
        if not eligible:
            if (
                getattr(getattr(self, "model_config", None), "max_model_len", None)
                == 4096
            ):
                logger.info(
                    "MiniCPMO45Stage1FIAFixed192 init_fallback reason=%s", reason
                )
            return
        _RUNTIME = _Runtime.create(torch, self.device)
        _RUNTIME.weak_ref_tensors = attention_module.weak_ref_tensors
        logger.info(
            "MiniCPMO45Stage1FIAFixed192 initialized default_off=false capacity=%d "
            "mask_width=256 expected_tasks=%d stock_kv_writer=true enpu=false",
            FIXED_CAPACITY,
            _EXPECTED_TASKS,
        )

    @functools.wraps(original_capture)
    def wrapped_capture(
        self: Any,
        query: Any,
        key: Any,
        value: Any,
        attn_metadata: Any,
        output: Any,
        layer: Any = None,
    ) -> Any:
        runtime = _RUNTIME
        candidate_shape = bool(
            runtime is not None
            and int(query.shape[0]) == 1
            and int(getattr(self, "num_heads", 0)) == 12
            and int(getattr(self, "num_kv_heads", 0)) == 12
            and int(getattr(self, "head_size", 0)) == 64
            and getattr(self, "sliding_window", None) is None
            and not getattr(self, "enable_c8_quant", False)
            and not getattr(self, "enable_hamming_sparse", False)
            and getattr(self, "sinks", None) is None
        )
        stock_contract = bool(
            bool(getattr(attn_metadata, "causal", False))
            and getattr(attn_metadata, "attn_mask", None) is not None
            and _mode_name(getattr(attn_metadata, "attn_state", "")) == "DECODEONLY"
            and getattr(attn_metadata, "actual_seq_lengths_q", None) == [1]
            and isinstance(getattr(attn_metadata, "seq_lens_list", None), list)
            and len(attn_metadata.seq_lens_list) == 1
            and getattr(getattr(attn_metadata, "block_tables", None), "ndim", 0) == 2
            and int(attn_metadata.block_tables.shape[0]) == 1
        )
        exact = candidate_shape and stock_contract
        if not exact:
            if runtime is not None:
                runtime.counters["capture.fallback"] += 1
            return original_capture(
                self, query, key, value, attn_metadata, output, layer
            )

        saved = (
            attn_metadata.seq_lens_list,
            attn_metadata.block_tables,
            getattr(attn_metadata, "attn_mask", None),
            bool(getattr(attn_metadata, "causal", True)),
        )
        stock_mask = saved[2]
        stock_sparse = 3 if saved[3] else 0
        stock_pre = 2_147_483_647
        stock_next = 2_147_483_647
        attn_metadata.seq_lens_list = [FIXED_CAPACITY]
        attn_metadata.block_tables = runtime.target_table
        attn_metadata.attn_mask = runtime.target_mask
        attn_metadata.causal = False
        try:
            with _prefix_capture_event(runtime):
                result = original_capture(
                    self, query, key, value, attn_metadata, output, layer
                )
            runtime.stock_capture_descriptors.append(
                (stock_mask, stock_sparse, stock_pre, stock_next)
            )
            runtime.counters["capture.rewrite"] += 1
            return result
        finally:
            (
                attn_metadata.seq_lens_list,
                attn_metadata.block_tables,
                attn_metadata.attn_mask,
                attn_metadata.causal,
            ) = saved

    @functools.wraps(original_update)
    def wrapped_update(
        update_stream: Any,
        forward_context: Any,
        num_tokens: int,
        vllm_config: Any,
        speculative_config: Any = None,
        num_dcp_pcp_tokens: Any = None,
        draft_attn_metadatas: Any = None,
    ) -> Any:
        runtime = _RUNTIME
        if runtime is None:
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )
        capture_rewrites = int(runtime.counters["capture.rewrite"])
        if capture_rewrites != _EXPECTED_TASKS:
            if capture_rewrites:
                raise RuntimeError(
                    "MiniCPM-o Stage1 fixed192 captured only "
                    f"{capture_rewrites}/{_EXPECTED_TASKS} FIA tasks"
                )
            return original_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )
        graph_params = get_graph_params()
        observation, unique, source_table = _observe(
            forward_context, num_tokens, graph_params
        )
        action = runtime.state.decide(observation)
        runtime.counters[f"action.{action.value}"] += 1
        if action is UpdateAction.STOCK_DYNAMIC:
            try:
                with _stock_task_params(
                    runtime, graph_params, num_tokens, source_table
                ):
                    with _prefix_task_epoch(
                        runtime, graph_params, num_tokens, "STOCK_DYNAMIC", update_stream
                    ):
                        result = original_update(
                            update_stream,
                            forward_context,
                            num_tokens,
                            vllm_config,
                            speculative_config,
                            num_dcp_pcp_tokens,
                            draft_attn_metadatas,
                        )
            except Exception:
                runtime.state.commit(observation, action, success=False)
                raise
            runtime.state.commit(observation, action, success=True)
            runtime.counters[f"fallback.{observation.reason}"] += 1
            _maybe_log(runtime, logger, observation.live_len or -1, action)
            return result

        assert observation.live_len is not None and source_table is not None
        try:
            _copy_fixed_descriptors(
                runtime, update_stream, source_table, observation.live_len
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            runtime.state.commit(observation, action, success=False)
            runtime.counters["fallback.descriptor_copy"] += 1
            # The graph was captured with the fixed non-causal descriptor.
            # A copy failure must therefore restore the complete stock FIA
            # task contract, not merely call the stock updater with the fixed
            # captured tuple still installed.
            with _stock_task_params(runtime, graph_params, num_tokens, source_table):
                with _prefix_task_epoch(
                    runtime,
                    graph_params,
                    num_tokens,
                    "DESCRIPTOR_COPY_STOCK_FALLBACK",
                    update_stream,
                ):
                    return original_update(
                        update_stream,
                        forward_context,
                        num_tokens,
                        vllm_config,
                        speculative_config,
                        num_dcp_pcp_tokens,
                        draft_attn_metadatas,
                    )

        if action is UpdateAction.FIXED_REBIND:
            try:
                with _fixed_metadata(unique, runtime):
                    with _prefix_task_epoch(
                        runtime, graph_params, num_tokens, "FIXED_REBIND", update_stream
                    ):
                        result = original_update(
                            update_stream,
                            forward_context,
                            num_tokens,
                            vllm_config,
                            speculative_config,
                            num_dcp_pcp_tokens,
                            draft_attn_metadatas,
                        )
            except Exception:
                runtime.state.commit(observation, action, success=False)
                raise
        else:
            try:
                _record_only(runtime, update_stream, graph_params)
                result = None
            except Exception:
                runtime.state.commit(observation, action, success=False)
                raise
        runtime.state.commit(observation, action, success=True)
        _maybe_log(runtime, logger, observation.live_len, action)
        return result

    setattr(wrapped_init, _INSTALL_MARKER, True)
    setattr(wrapped_capture, _CAPTURE_MARKER, True)
    setattr(wrapped_update, _UPDATE_MARKER, True)
    setattr(wrapped_init, _RUNNER_MARKER, True)
    NPUModelRunner.__init__ = wrapped_init
    AscendAttentionBackendImpl.full_graph_fia = wrapped_capture
    AscendAttentionBackendImpl.update_graph_params = staticmethod(wrapped_update)
    attention_module.AscendAttentionBackendImpl = AscendAttentionBackendImpl
    logger.info(
        "MiniCPMO45Stage1FIAFixed192 plugin_installed env=%s default=off "
        "conditional_on_persistence_gate=true",
        _ENV,
    )
