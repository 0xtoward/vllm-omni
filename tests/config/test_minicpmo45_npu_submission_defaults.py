# SPDX-License-Identifier: Apache-2.0
"""Submission-only MiniCPM-o 4.5 NPU startup defaults."""

import os
import sys

import pytest

from vllm_omni.entrypoints.cli.main import (
    _maybe_reexec_minicpmo45_npu_under_numactl,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_minicpmo45_npu_serve_reexecs_under_configured_numa_node(monkeypatch):
    calls = []
    monkeypatch.delenv("_MINICPMO45_NUMACTL_DONE", raising=False)
    monkeypatch.delenv("MINICPMO45_NO_NUMACTL", raising=False)
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM",
        raising=False,
    )
    monkeypatch.setenv("MINICPMO45_NUMACTL_NODE", "3")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm", "serve", "/models/MiniCPM-o-4_5", "--omni"],
    )
    # Containers may expose physical nodes such as davinci6/7 instead of
    # davinci0.  The manager device is the stable Ascend runtime signal.
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path == "/dev/davinci_manager",
    )

    def record_exec(file, argv, env):
        calls.append((file, argv, env))
        raise OSError("test sentinel")

    monkeypatch.setattr(os, "execvpe", record_exec)
    _maybe_reexec_minicpmo45_npu_under_numactl()

    assert len(calls) == 1
    file, argv, env = calls[0]
    assert file == "numactl"
    assert argv[:3] == ["numactl", "--cpunodebind=3", sys.executable]
    assert argv[3:] == sys.argv
    assert env["_MINICPMO45_NUMACTL_DONE"] == "1"
    assert env["VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES"] == "25"
    assert env["VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM"] == "1"


def test_minicpmo45_numactl_opt_out_preserves_official_command(monkeypatch):
    monkeypatch.setenv("MINICPMO45_NO_NUMACTL", "1")
    monkeypatch.delenv("_MINICPMO45_NUMACTL_DONE", raising=False)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES",
        "0",
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM",
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm", "serve", "/models/MiniCPM-o-4_5", "--omni"],
    )
    monkeypatch.setattr(
        os,
        "execvpe",
        lambda *_args, **_kwargs: pytest.fail("unexpected re-exec"),
    )

    _maybe_reexec_minicpmo45_npu_under_numactl()

    # NUMA opt-out does not disable the model defaults, and explicit model
    # opt-outs remain authoritative.
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES"] == "0"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM"] == "1"
