# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware smoke for sealed NPUGraph ownership and request-time replay."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.platforms.npu.graph_tools import (
    EstimatorSemantic,
    SealedNPUExactGraphRunner,
    _clone_tensor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.npu]


def _npu_available() -> bool:
    npu = getattr(torch, "npu", None)
    return npu is not None and bool(npu.is_available())


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
@pytest.mark.parametrize("npu_format", [2, 30])
def test_clone_preserves_npu_format(npu_format):
    import torch_npu

    source = torch.arange(4096, device="npu", dtype=torch.float32).reshape(16, 2, 64, 2)
    source = torch_npu.npu_format_cast(source, npu_format)
    cloned = _clone_tensor(source)
    torch.npu.synchronize()

    assert torch.equal(source, cloned)
    assert torch_npu.get_npu_format(cloned) == torch_npu.get_npu_format(source)
    assert cloned.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_private_pool_capture_seal_and_changed_value_replay():
    semantic = EstimatorSemantic(
        role="prompt_b1",
        request_batch=1,
        prompt_frames=302,
        codec_token_count=0,
        last_chunk=False,
        flush_encoder=False,
        cache_present=False,
        attention_cache_length=0,
        cfm_steps=3,
        codec_chunk_frames=25,
        left_context_frames=3,
    )
    runner = SealedNPUExactGraphRunner(
        mode="on",
        expected_roles={"prompt_b1"},
        component_name="sealed-runner-hardware-smoke",
    )

    def compute(value):
        return (torch.sin(value) * 2.0,)

    first_input = torch.arange(128, device="npu", dtype=torch.float32)
    first = runner.run(
        "toy",
        semantic,
        (first_input,),
        stock_compute=lambda: compute(first_input),
        graph_compute=compute,
    )
    runner.seal()
    second_input = first_input + 3.0
    second = runner.run(
        "toy",
        semantic,
        (second_input,),
        stock_compute=lambda: compute(second_input),
        graph_compute=compute,
    )
    torch.npu.synchronize()

    assert torch.equal(first[0], compute(first_input)[0])
    assert torch.equal(second[0], compute(second_input)[0])
    assert first[0].untyped_storage().data_ptr() != second[0].untyped_storage().data_ptr()
    assert runner.snapshot("done")["resident_graphs"] == 1
    assert runner.snapshot("done")["replay_successes"] == 1
