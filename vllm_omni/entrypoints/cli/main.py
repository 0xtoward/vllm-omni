"""
CLI entry point for vLLM-Omni that intercepts vLLM commands.
"""

import importlib.metadata
import sys


def _maybe_reexec_minicpmo45_npu_under_numactl() -> None:
    """Apply MiniCPM-o NPU process defaults before stage processes spawn.

    The official launcher does not set CPU affinity.  Binding only after the
    engine children exist leaves their first-touch pages and worker threads
    scattered across NUMA nodes, so this narrowly scoped re-exec happens at
    the CLI boundary.  Missing/forbidden ``numactl`` is a safe no-op, and an
    explicit environment setting can opt out or select another measured node.

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

    if os.environ.get("_MINICPMO45_NUMACTL_DONE") == "1":
        return
    if os.environ.get("MINICPMO45_NO_NUMACTL") == "1":
        return

    env = dict(os.environ)
    env["_MINICPMO45_NUMACTL_DONE"] = "1"
    node = env.get("MINICPMO45_NUMACTL_NODE", "0")
    cmd = ["numactl", f"--cpunodebind={node}", sys.executable, *sys.argv]
    try:
        os.execvpe("numactl", cmd, env)
    except OSError:
        # The evaluator image normally provides numactl.  Keep serving valid
        # on other images instead of turning an optional affinity optimization
        # into a startup dependency.
        return


def main():
    """Main CLI entry point that intercepts vLLM commands."""
    _maybe_reexec_minicpmo45_npu_under_numactl()
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
