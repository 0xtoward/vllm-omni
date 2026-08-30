# SPDX-License-Identifier: Apache-2.0
"""Submission-only MiniCPM-o 4.5 NPU startup defaults."""

import os
import sys

import pytest

from vllm_omni.entrypoints.cli.main import (
    _format_linux_cpu_list,
    _maybe_prepare_minicpmo45_npu_runtime,
    _select_local_numa_cpuset,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_format_linux_cpu_list_compacts_ranges():
    assert _format_linux_cpu_list({0, 1, 2, 5, 7, 8}) == "0-2,5,7-8"


def test_select_local_numa_cpuset_respects_allowed_and_largest_node():
    selected = _select_local_numa_cpuset(
        {3, 4, 5, 20, 21},
        {0: {0, 1, 2, 3, 4, 5}, 1: {20, 21, 22}},
    )
    assert selected == (0, {3, 4, 5})


def test_select_local_numa_cpuset_breaks_equal_size_ties_deterministically():
    selected = _select_local_numa_cpuset(
        {0, 1, 20, 21},
        {0: {0, 1}, 1: {20, 21}},
    )
    assert selected == (0, {0, 1})


def test_minicpmo45_npu_serve_limits_parent_before_native_binding(monkeypatch):
    calls: list[tuple[int, set[int]]] = []
    affinity = set(range(32))
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192",
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
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(affinity))

    def set_affinity(pid, cpus):
        calls.append((pid, set(cpus)))
        affinity.clear()
        affinity.update(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: [
            "/sys/devices/system/node/node0/cpulist",
            "/sys/devices/system/node/node1/cpulist",
        ],
    )
    monkeypatch.setattr(
        "pathlib.Path.read_text",
        lambda path: "0-15" if "node0" in str(path) else "16-31",
    )

    _maybe_prepare_minicpmo45_npu_runtime()

    assert calls == [(0, set(range(16)))]
    assert os.environ["_MINICPMO45_NUMA_AFFINITY"] == "node0:16"
    assert os.environ["_MINICPMO45_NUMA_INHERITED_CPUSET"] == "0-31"
    assert os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] == "0-15"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES"] == "25"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192"] == "1"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM"] == "1"


def test_minicpmo45_numa_fails_if_kernel_does_not_apply_requested_mask(
    monkeypatch,
):
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, _cpus: None)
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

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    with pytest.raises(RuntimeError, match="affinity verification failed"):
        _limit_minicpmo45_npu_to_local_cpuset()


def test_minicpmo45_numa_preserves_an_inherited_single_node_mask(monkeypatch):
    inherited = {4, 5, 6, 7}
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(inherited))
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda _pid, _cpus: pytest.fail("must preserve an already-local mask"),
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
        lambda path: "0-7" if "node0" in str(path) else "8-15",
    )

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    _limit_minicpmo45_npu_to_local_cpuset()
    assert os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] == "4-7"


def test_minicpmo45_numa_skips_one_bad_sysfs_node(monkeypatch):
    affinity = set(range(16))
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(affinity))

    def set_affinity(_pid, cpus):
        affinity.clear()
        affinity.update(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: [
            "/sys/devices/system/node/node0/cpulist",
            "/sys/devices/system/node/node1/cpulist",
        ],
    )

    def read_cpulist(path):
        if "node0" in str(path):
            raise OSError("transient sysfs read failure")
        return "8-15"

    monkeypatch.setattr("pathlib.Path.read_text", read_cpulist)

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    _limit_minicpmo45_npu_to_local_cpuset()
    assert os.environ["_MINICPMO45_NUMA_AFFINITY"] == "node1:8"
