"""
CLI entry point for vLLM-Omni that intercepts vLLM commands.
"""

import importlib.metadata
import sys


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


def _select_local_numa_cpuset(
    allowed: set[int],
    node_cpus: dict[int, set[int]],
    preferred_node: int | None = None,
) -> tuple[int, set[int]] | None:
    """Choose one NUMA-local subset without escaping the inherited cpuset."""
    candidates = {
        node: allowed & cpus for node, cpus in node_cpus.items() if allowed & cpus
    }
    if not candidates:
        return None
    if preferred_node is not None and preferred_node in candidates:
        return preferred_node, candidates[preferred_node]
    node = min(candidates, key=lambda item: (-len(candidates[item]), item))
    return node, candidates[node]


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
        return
    allowed = set(os.sched_getaffinity(0))
    node_cpus: dict[int, set[int]] = {}
    for path_value in glob.glob("/sys/devices/system/node/node*/cpulist"):
        path = Path(path_value)
        suffix = path.parent.name.removeprefix("node")
        if suffix.isdigit():
            node_cpus[int(suffix)] = _parse_linux_cpu_list(path.read_text())
    preferred = os.environ.get("MINICPMO45_NUMA_NODE")
    selected = _select_local_numa_cpuset(
        allowed,
        node_cpus,
        int(preferred) if preferred is not None else None,
    )
    if selected is None:
        return
    node, cpus = selected
    if cpus != allowed:
        os.sched_setaffinity(0, cpus)
    os.environ["_MINICPMO45_NUMA_AFFINITY"] = f"node{node}:{len(cpus)}"


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
    if not any(
        "minicpm-o-4_5" in arg.lower() or "minicpmo_4_5" in arg.lower()
        for arg in sys.argv
    ):
        return
    if not (
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        or os.path.exists("/dev/davinci_manager")
    ):
        return

    # Safe opt-out defaults: explicit user/evaluator values always win.
    os.environ.setdefault(
        "VLLM_OMNI_MINICPMO45_STAGE1_SPARSE_CHUNK_FRAMES",
        "25",
    )
    os.environ.setdefault(
        "VLLM_OMNI_MINICPMO45_STAGE2_FREEZE_HIFT_WEIGHT_NORM",
        "1",
    )

    _limit_minicpmo45_npu_to_local_cpuset()


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
