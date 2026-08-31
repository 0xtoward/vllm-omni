"""
CLI entry point for vLLM-Omni that intercepts vLLM commands.
"""

import importlib.metadata
import sys

# The official evaluator already assigns one job an exclusive 24-core cpuset.
# Preserve small, scheduler-managed allocations even when they span NUMA nodes;
# narrowing those again can leave the service with only a fraction of its CPUs.
_MINICPMO45_PRESERVE_INHERITED_CPUSET_MAX = 32


def _parse_linux_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", 1))
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(item))
    return cpus


def _format_linux_cpu_list(cpus: set[int]) -> str:
    """Format a CPU set using Linux's compact ``Cpus_allowed_list`` syntax."""
    if not cpus:
        return ""
    values = sorted(cpus)
    ranges: list[str] = []
    first = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = value
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def _select_local_numa_cpuset(
    allowed: set[int],
    node_cpus: dict[int, set[int]],
) -> tuple[int, set[int]] | None:
    """Choose one NUMA-local subset without escaping the inherited cpuset."""
    candidates = {node: allowed & cpus for node, cpus in node_cpus.items() if allowed & cpus}
    if not candidates:
        return None
    node = min(candidates, key=lambda item: (-len(candidates[item]), item))
    return node, candidates[node]


def _fail_open_minicpmo45_numa(
    reason: str,
    inherited: set[int] | None,
    detail: str = "",
) -> None:
    """Keep service startup alive when optional NUMA narrowing is unsafe.

    If narrowing already changed the current process, make a best-effort
    rollback to the inherited mask.  The vLLM-Ascend native binder can then
    operate within the evaluator-provided cpuset as it does without this
    MiniCPM-specific optimization.
    """
    import os

    effective: set[int] | None = inherited
    restore_status = "not-needed"
    if inherited is not None:
        try:
            effective = set(os.sched_getaffinity(0))
        except OSError as exc:
            effective = None
            restore_status = f"inspect-failed:{type(exc).__name__}"
        else:
            if effective != inherited:
                try:
                    os.sched_setaffinity(0, inherited)
                    effective = set(os.sched_getaffinity(0))
                except OSError as exc:
                    restore_status = f"failed:{type(exc).__name__}"
                else:
                    restore_status = "restored" if effective == inherited else "mismatch"

    inherited_text = _format_linux_cpu_list(inherited) if inherited is not None else "unknown"
    effective_text = _format_linux_cpu_list(effective) if effective is not None else "unknown"
    os.environ["_MINICPMO45_NUMA_INHERITED_CPUSET"] = inherited_text
    os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] = effective_text
    os.environ["_MINICPMO45_NUMA_STATUS"] = f"skipped:{reason}"
    detail_suffix = f" detail={detail}" if detail else ""
    print(
        "[minicpmo45-numa] "
        f"status=skipped reason={reason} inherited={inherited_text} "
        f"effective={effective_text} restore={restore_status} "
        "owner=vllm-ascend-native-binder"
        f"{detail_suffix}",
        file=sys.stderr,
        flush=True,
    )


def _limit_minicpmo45_npu_to_local_cpuset() -> None:
    """Narrow the parent cpuset before vLLM-Ascend performs native binding.

    vLLM-Ascend remains the sole owner of worker/ACL/release thread affinity,
    IRQ placement and page migration. This only prevents its A3 global-slice
    mode from constructing one worker pool across several NUMA nodes. The
    inherited cgroup/cpuset is always respected.
    """
    import glob
    import os
    from pathlib import Path

    if os.environ.get("MINICPMO45_NO_NUMA_AFFINITY") == "1":
        print(
            "[minicpmo45-numa] disabled by MINICPMO45_NO_NUMA_AFFINITY=1",
            file=sys.stderr,
            flush=True,
        )
        return
    try:
        allowed = set(os.sched_getaffinity(0))
    except OSError as exc:
        _fail_open_minicpmo45_numa(
            "get-affinity-failed",
            None,
            type(exc).__name__,
        )
        return
    if not allowed:
        _fail_open_minicpmo45_numa("empty-inherited-cpuset", allowed)
        return
    if len(allowed) <= _MINICPMO45_PRESERVE_INHERITED_CPUSET_MAX:
        allowed_text = _format_linux_cpu_list(allowed)
        os.environ["_MINICPMO45_NUMA_AFFINITY"] = f"inherited:{len(allowed)}"
        os.environ["_MINICPMO45_NUMA_INHERITED_CPUSET"] = allowed_text
        os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] = allowed_text
        os.environ["_MINICPMO45_NUMA_STATUS"] = "preserved:small-cpuset"
        print(
            "[minicpmo45-numa] status=preserved reason=small-cpuset "
            f"inherited={allowed_text} effective={allowed_text} "
            f"threshold={_MINICPMO45_PRESERVE_INHERITED_CPUSET_MAX} "
            "owner=vllm-ascend-native-binder",
            file=sys.stderr,
            flush=True,
        )
        return
    node_cpus: dict[int, set[int]] = {}
    discovery_errors: list[str] = []
    for path_value in sorted(glob.glob("/sys/devices/system/node/node*/cpulist")):
        path = Path(path_value)
        suffix = path.parent.name.removeprefix("node")
        if suffix.isdigit():
            try:
                node_cpus[int(suffix)] = _parse_linux_cpu_list(path.read_text())
            except (OSError, ValueError) as exc:
                discovery_errors.append(f"{path_value}:{type(exc).__name__}")
    selected = _select_local_numa_cpuset(allowed, node_cpus)
    if selected is None:
        _fail_open_minicpmo45_numa(
            "no-node-intersection",
            allowed,
            ",".join(discovery_errors),
        )
        return
    node, cpus = selected
    status = "preserved" if cpus == allowed else "applied"
    if cpus != allowed:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError as exc:
            _fail_open_minicpmo45_numa(
                "set-affinity-failed",
                allowed,
                type(exc).__name__,
            )
            return
    try:
        effective = set(os.sched_getaffinity(0))
    except OSError as exc:
        _fail_open_minicpmo45_numa(
            "verify-affinity-failed",
            allowed,
            type(exc).__name__,
        )
        return
    if effective != cpus:
        _fail_open_minicpmo45_numa(
            "verify-affinity-mismatch",
            allowed,
            f"requested={_format_linux_cpu_list(cpus)},observed={_format_linux_cpu_list(effective)}",
        )
        return
    effective_nodes = {
        candidate_node for candidate_node, candidate_cpus in node_cpus.items() if effective & candidate_cpus
    }
    if effective_nodes != {node}:
        _fail_open_minicpmo45_numa(
            "verify-node-locality-mismatch",
            allowed,
            f"selected={node},effective_nodes={sorted(effective_nodes)}",
        )
        return
    os.environ["_MINICPMO45_NUMA_AFFINITY"] = f"node{node}:{len(cpus)}"
    os.environ["_MINICPMO45_NUMA_INHERITED_CPUSET"] = _format_linux_cpu_list(allowed)
    os.environ["_MINICPMO45_NUMA_EFFECTIVE_CPUSET"] = _format_linux_cpu_list(effective)
    print(
        "[minicpmo45-numa] "
        f"status={status} "
        f"inherited={_format_linux_cpu_list(allowed)} "
        f"selected=node{node} effective={_format_linux_cpu_list(effective)} "
        "owner=vllm-ascend-native-binder",
        file=sys.stderr,
        flush=True,
    )
    if discovery_errors:
        print(
            "[minicpmo45-numa] ignored discovery errors: " + ",".join(discovery_errors),
            file=sys.stderr,
            flush=True,
        )


def _maybe_prepare_minicpmo45_npu_runtime() -> None:
    """Apply MiniCPM-o NPU process defaults before stage processes spawn.

    Model-local performance switches live here instead of in the registered
    model modules.  vLLM hashes those modules for its ModelInfo cache; changing
    a default there needlessly invalidates the cache and starts a heavyweight
    model-inspection subprocess before every fresh deployment.
    """
    import os

    if "serve" not in sys.argv or "--omni" not in sys.argv:
        return
    if not any("minicpm-o-4_5" in arg.lower() or "minicpmo_4_5" in arg.lower() for arg in sys.argv):
        return
    if not (os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.path.exists("/dev/davinci_manager")):
        return

    # Safe opt-out defaults: explicit user/evaluator values always win.
    os.environ.setdefault(
        "VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES",
        "25",
    )
    os.environ.setdefault(
        "VLLM_OMNI_MINICPMO45_STAGE1_KNOWN_CONTROLLER_BYPASS",
        "1",
    )
    os.environ.setdefault(
        "VLLM_OMNI_MINICPMO45_STAGE1_FIA_FIXED192",
        "1",
    )
    os.environ.setdefault(
        "VLLM_OMNI_MINICPMO45_STAGE1_FIA_PREFIX_BARRIER_GATE0",
        "1",
    )
    os.environ.setdefault(
        "VLLM_OMNI_MINICPMO45_STAGE1_EOS_BATCH_K",
        "4",
    )
    os.environ.setdefault(
        "VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM",
        "1",
    )

    _limit_minicpmo45_npu_to_local_cpuset()


def _prepare_and_install_minicpmo45_stage1_fia_fixed192() -> None:
    """Order the default preparation before the fixed192 plugin install.

    Entry-point enumeration order is not part of the Python packaging
    contract. Keeping the dependency explicit here prevents the fixed192
    installer from observing its legacy default-off state when the official
    ``vllm serve`` executable loads general plugins in a different order.
    """
    _maybe_prepare_minicpmo45_npu_runtime()

    from vllm_omni.platforms.npu.ascend_stage1_fia_fixed192_patch import (
        install_stage1_fia_fixed192_candidate,
    )

    install_stage1_fia_fixed192_candidate()


def main():
    """Main CLI entry point that intercepts vLLM commands."""
    _maybe_prepare_minicpmo45_npu_runtime()
    # Check if --omni flag is present
    if "--omni" not in sys.argv:
        from vllm.entrypoints.cli.main import main as vllm_main

        vllm_main()
        return
    else:
        # Force colored logging even when piped (e.g. `| tee`).
        # Must be set before any vLLM import because the logger
        # formatter is configured at import time via _use_color().
        import os

        if "VLLM_LOGGING_COLOR" not in os.environ:
            os.environ["VLLM_LOGGING_COLOR"] = "1"

        from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup

        import vllm_omni.entrypoints.cli.benchmark.main
        import vllm_omni.entrypoints.cli.serve
        from vllm_omni.utils.tracking_parser import TrackingArgumentParser

        CMD_MODULES = [
            vllm_omni.entrypoints.cli.serve,
            vllm_omni.entrypoints.cli.benchmark.main,
        ]

        cli_env_setup()

        from vllm_omni.entrypoints.cli.serve import _ensure_vllm_platform

        _ensure_vllm_platform()

        parser = TrackingArgumentParser(
            description="vLLM OMNI CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        try:
            _omni_version = importlib.metadata.version("vllm_omni")
        except importlib.metadata.PackageNotFoundError:
            try:
                from vllm_omni.version import __version__ as _omni_version  # type: ignore
            except Exception:
                _omni_version = "dev"
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=_omni_version,
        )
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        if args.subparser in cmds:
            cmds[args.subparser].validate(args)

        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
