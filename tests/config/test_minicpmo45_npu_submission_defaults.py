# SPDX-License-Identifier: Apache-2.0
"""Submission-only MiniCPM-o 4.5 NPU startup defaults."""

import os
import sys

import pytest

from vllm_omni.entrypoints.cli.main import (
    _maybe_prepare_minicpmo45_npu_runtime,
    _select_local_numa_cpuset,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_select_local_numa_cpuset_respects_allowed_and_largest_node():
    selected = _select_local_numa_cpuset(
        {3, 4, 5, 20, 21},
        {0: {0, 1, 2, 3, 4, 5}, 1: {20, 21, 22}},
    )
    assert selected == (0, {3, 4, 5})


def test_select_local_numa_cpuset_honors_available_preference():
    selected = _select_local_numa_cpuset(
        {0, 1, 20, 21},
        {0: {0, 1}, 1: {20, 21}},
        preferred_node=1,
    )
    assert selected == (1, {20, 21})


def test_minicpmo45_npu_serve_limits_parent_before_native_binding(monkeypatch):
    calls: list[tuple[int, set[int]]] = []
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.delenv("MINICPMO45_NUMA_NODE", raising=False)
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM",
        raising=False,
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm", "serve", "/models/MiniCPM-o-4_5", "--omni"],
    )
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path == "/dev/davinci_manager",
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append((pid, set(cpus))),
    )
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: [
            "/sys/devices/system/node/node0/cpulist",
            "/sys/devices/system/node/node1/cpulist",
        ],
    )
    monkeypatch.setattr(
        "pathlib.Path.read_text",
        lambda path: "0-3" if "node0" in str(path) else "4-7",
    )

    _maybe_prepare_minicpmo45_npu_runtime()

    assert calls == [(0, {0, 1, 2, 3})]
    assert os.environ["_MINICPMO45_NUMA_AFFINITY"] == "node0:4"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES"] == "25"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM"] == "1"
