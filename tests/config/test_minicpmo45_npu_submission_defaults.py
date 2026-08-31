# SPDX-License-Identifier: Apache-2.0
"""Submission-only MiniCPM-o 4.5 NPU startup defaults."""

import os
import sys
import types
from pathlib import Path

import pytest

from vllm_omni.entrypoints.cli.main import (
    _format_linux_cpu_list,
    _maybe_prepare_minicpmo45_npu_runtime,
    _prepare_and_install_minicpmo45_stage1_fia_fixed192,
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


def test_official_vllm_entrypoint_registers_minicpmo_runtime_defaults():
    project_root = Path(__file__).resolve().parents[2]
    project_metadata = (project_root / "pyproject.toml").read_text()

    assert '[project.entry-points."vllm.general_plugins"]' in project_metadata
    assert (
        "vllm_omni_minicpmo45_runtime_defaults = "
        '"vllm_omni.entrypoints.cli.main:'
        '_maybe_prepare_minicpmo45_npu_runtime"' in project_metadata
    )
    assert (
        "vllm_omni_minicpmo45_stage1_fia_fixed192 = "
        '"vllm_omni.entrypoints.cli.main:'
        '_prepare_and_install_minicpmo45_stage1_fia_fixed192"' in project_metadata
    )


def test_fixed192_plugin_prepares_defaults_before_install(monkeypatch):
    observed: list[tuple[str | None, str | None]] = []
    plugin_module = types.ModuleType("vllm_omni.platforms.npu.ascend_stage1_fia_fixed192_patch")
    plugin_module.install_stage1_fia_fixed192_candidate = lambda: observed.append(
        (
            os.environ.get("VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192"),
            os.environ.get("VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0"),
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms.npu.ascend_stage1_fia_fixed192_patch",
        plugin_module,
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_EOS_BATCH_K",
        raising=False,
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MINICPMO45_NO_NUMA_AFFINITY", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm", "serve", "/models/MiniCPM-o-4_5", "--omni"],
    )

    _prepare_and_install_minicpmo45_stage1_fia_fixed192()

    assert observed == [("1", "1")]


def test_minicpmo45_npu_serve_limits_parent_before_native_binding(monkeypatch):
    calls: list[tuple[int, set[int]]] = []
    affinity = set(range(64))
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
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0",
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
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(affinity), raising=False)

    def set_affinity(pid, cpus):
        calls.append((pid, set(cpus)))
        affinity.clear()
        affinity.update(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity, raising=False)
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: [
            "/sys/devices/system/node/node0/cpulist",
            "/sys/devices/system/node/node1/cpulist",
        ],
    )
    monkeypatch.setattr(
        "pathlib.Path.read_text",
        lambda path: "0-31" if "node0" in str(path) else "32-63",
    )

    _maybe_prepare_minicpmo45_npu_runtime()

    assert calls == [(0, set(range(32)))]
    assert os.environ["_MINICPMO45_NUMA_AFFINITY"] == "node0:32"
    assert os.environ["_MINICPMO45_NUMA_INHERITED_CPUSET"] == "0-63"
    assert os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] == "0-31"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES"] == "25"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192"] == "1"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0"] == "1"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_EOS_BATCH_K"] == "8"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM"] == "1"


def test_minicpmo45_runtime_preparation_is_idempotent(monkeypatch):
    affinity = set(range(64))
    calls: list[set[int]] = []
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm", "serve", "/models/MiniCPM-o-4_5", "--omni"],
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(affinity), raising=False)

    def set_affinity(_pid, cpus):
        calls.append(set(cpus))
        affinity.clear()
        affinity.update(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity, raising=False)
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: [
            "/sys/devices/system/node/node0/cpulist",
            "/sys/devices/system/node/node1/cpulist",
        ],
    )
    monkeypatch.setattr(
        "pathlib.Path.read_text",
        lambda path: "0-31" if "node0" in str(path) else "32-63",
    )

    _maybe_prepare_minicpmo45_npu_runtime()
    _maybe_prepare_minicpmo45_npu_runtime()

    assert calls == [set(range(32))]
    assert affinity == set(range(32))
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192"] == "1"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0"] == "1"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_EOS_BATCH_K"] == "8"


def test_minicpmo45_numa_fails_open_if_kernel_does_not_apply_requested_mask(
    monkeypatch,
):
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(64)), raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, _cpus: None, raising=False)
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: [
            "/sys/devices/system/node/node0/cpulist",
            "/sys/devices/system/node/node1/cpulist",
        ],
    )
    monkeypatch.setattr(
        "pathlib.Path.read_text",
        lambda path: "0-31" if "node0" in str(path) else "32-63",
    )

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    _limit_minicpmo45_npu_to_local_cpuset()

    assert set(os.sched_getaffinity(0)) == set(range(64))
    assert os.environ["_MINICPMO45_NUMA_STATUS"] == "skipped:verify-affinity-mismatch"
    assert os.environ["_MINICPMO45_NUMA_INHERITED_CPUSET"] == "0-63"
    assert os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] == "0-63"


def test_minicpmo45_numa_fails_open_without_node_intersection(monkeypatch):
    inherited = set(range(200, 240))
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(inherited), raising=False)
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda _pid, _cpus: pytest.fail("must not change inherited affinity"),
        raising=False,
    )
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: ["/sys/devices/system/node/node0/cpulist"],
    )
    monkeypatch.setattr("pathlib.Path.read_text", lambda _path: "0-79")

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    _limit_minicpmo45_npu_to_local_cpuset()

    assert set(os.sched_getaffinity(0)) == inherited
    assert os.environ["_MINICPMO45_NUMA_STATUS"] == "skipped:no-node-intersection"
    assert os.environ["_MINICPMO45_NUMA_INHERITED_CPUSET"] == "200-239"
    assert os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] == "200-239"


def test_minicpmo45_numa_fails_open_if_setaffinity_is_denied(monkeypatch):
    inherited = set(range(64))
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(inherited), raising=False)

    def deny_affinity(_pid, _cpus):
        raise PermissionError("cpuset is managed by the evaluator")

    monkeypatch.setattr(os, "sched_setaffinity", deny_affinity, raising=False)
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: [
            "/sys/devices/system/node/node0/cpulist",
            "/sys/devices/system/node/node1/cpulist",
        ],
    )
    monkeypatch.setattr(
        "pathlib.Path.read_text",
        lambda path: "0-31" if "node0" in str(path) else "32-63",
    )

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    _limit_minicpmo45_npu_to_local_cpuset()

    assert set(os.sched_getaffinity(0)) == inherited
    assert os.environ["_MINICPMO45_NUMA_STATUS"] == "skipped:set-affinity-failed"


def test_minicpmo45_numa_restores_inherited_mask_after_verify_mismatch(
    monkeypatch,
):
    inherited = set(range(40))
    affinity = set(inherited)
    calls: list[set[int]] = []
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(affinity), raising=False)

    def set_affinity(_pid, cpus):
        requested = set(cpus)
        calls.append(requested)
        affinity.clear()
        if requested == set(range(20)):
            affinity.update({0, 1})
        else:
            affinity.update(requested)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity, raising=False)
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: [
            "/sys/devices/system/node/node0/cpulist",
            "/sys/devices/system/node/node1/cpulist",
        ],
    )
    monkeypatch.setattr(
        "pathlib.Path.read_text",
        lambda path: "0-19" if "node0" in str(path) else "20-39",
    )

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    _limit_minicpmo45_npu_to_local_cpuset()

    assert calls == [set(range(20)), inherited]
    assert affinity == inherited
    assert os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] == "0-39"


def test_minicpmo45_numa_preserves_an_inherited_small_mask(monkeypatch):
    inherited = {4, 5, 6, 7}
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(inherited), raising=False)
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda _pid, _cpus: pytest.fail("must preserve an already-local mask"),
        raising=False,
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
    affinity = set(range(64))
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(affinity), raising=False)

    def set_affinity(_pid, cpus):
        affinity.clear()
        affinity.update(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", set_affinity, raising=False)
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
        return "32-63"

    monkeypatch.setattr("pathlib.Path.read_text", read_cpulist)

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    _limit_minicpmo45_npu_to_local_cpuset()
    assert os.environ["_MINICPMO45_NUMA_AFFINITY"] == "node1:32"


def test_minicpmo45_numa_preserves_judge_managed_24_core_mask(monkeypatch):
    inherited = set(range(12)) | set(range(80, 92))
    monkeypatch.delenv("MINICPMO45_NO_NUMA_AFFINITY", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(inherited), raising=False)
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda _pid, _cpus: pytest.fail("already node-local; must not rebind"),
        raising=False,
    )
    monkeypatch.setattr(
        "glob.glob",
        lambda _pattern: pytest.fail("small scheduler cpuset must be preserved before NUMA discovery"),
    )

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    _limit_minicpmo45_npu_to_local_cpuset()

    assert set(os.sched_getaffinity(0)) == inherited
    assert os.environ["_MINICPMO45_NUMA_AFFINITY"] == "inherited:24"
    assert os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] == "0-11,80-91"
    assert os.environ["_MINICPMO45_NUMA_STATUS"] == "preserved:small-cpuset"


def test_minicpmo45_numa_explicit_opt_out_skips_affinity_calls(monkeypatch):
    monkeypatch.setenv("MINICPMO45_NO_NUMA_AFFINITY", "1")
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: pytest.fail("opt-out must skip NUMA discovery"),
        raising=False,
    )

    from vllm_omni.entrypoints.cli.main import (
        _limit_minicpmo45_npu_to_local_cpuset,
    )

    _limit_minicpmo45_npu_to_local_cpuset()


def test_non_minicpmo_serve_does_not_change_defaults_or_affinity(monkeypatch):
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0",
        raising=False,
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm", "serve", "/models/Qwen3-Omni", "--omni"],
    )
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: pytest.fail("other models must not enter MiniCPM NUMA"),
        raising=False,
    )

    _maybe_prepare_minicpmo45_npu_runtime()

    assert "VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192" not in os.environ
    assert "VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0" not in os.environ


def test_minicpmo45_explicit_fixed192_opt_out_is_preserved(monkeypatch):
    inherited = set(range(24))
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192", "0")
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0",
        "0",
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("MINICPMO45_NO_NUMA_AFFINITY", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm", "serve", "/models/MiniCPM-o-4_5", "--omni"],
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(inherited), raising=False)

    observed: list[tuple[str | None, str | None]] = []
    plugin_module = types.ModuleType("vllm_omni.platforms.npu.ascend_stage1_fia_fixed192_patch")
    plugin_module.install_stage1_fia_fixed192_candidate = lambda: observed.append(
        (
            os.environ.get("VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192"),
            os.environ.get("VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0"),
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms.npu.ascend_stage1_fia_fixed192_patch",
        plugin_module,
    )

    _prepare_and_install_minicpmo45_stage1_fia_fixed192()

    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192"] == "0"
    assert os.environ["VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0"] == "0"
    assert observed == [("0", "0")]
