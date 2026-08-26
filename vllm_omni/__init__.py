"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

# We import version early, because it will warn if vLLM / vLLM Omni
# are not using the same major + minor version (if vLLM is installed).
# We should do this before applying patch, because vLLM imports might
# throw in patch if the versions differ.
from .version import __version__, __version_tuple__  # isort:skip # noqa: F401

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore

# Register custom configs (AutoConfig, AutoTokenizer) as early as possible.
from vllm_omni.transformers_utils import configs as _configs  # noqa: F401, E402
from vllm_omni.transformers_utils import parsers as _parsers  # noqa: F401, E402

from .config import OmniModelConfig


def __getattr__(name: str):
    # Lazy import for AsyncOmni and Omni to avoid pulling in heavy
    # dependencies (vllm model_loader → fused_moe → pynvml) at package
    # import time.  This prevents crashes in lightweight subprocesses
    # (e.g. model-architecture inspection) that lack a CUDA context.
    # See: https://github.com/vllm-project/vllm-omni/issues/1793
    if name == "AsyncOmni":
        from .entrypoints.async_omni import AsyncOmni

        return AsyncOmni
    if name == "Omni":
        from .entrypoints.omni import Omni

        return Omni
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]


def _minicpmo45_allow_view_optimize_option() -> None:
    """Accept torch_npu's own W8 graph compatibility option.

    The image's ``npu_fx_compiler`` sets ``enable_view_optimize=False`` when a
    weight-quant matmul enters an ACL graph, but the matching config class does
    not declare that option. Resolve the actual base class from the config MRO
    and permit only this vendor-owned key.
    """
    bases = []
    try:
        from torch_npu.dynamo.npugraph_ex.configs import experimental_config

        for name in ("_ExperimentalConfig", "_AclGraphExperimentalConfig"):
            config_class = getattr(experimental_config, name, None)
            if config_class is None:
                continue
            for base in config_class.__mro__[1:]:
                if base is not object and "__setattr__" in base.__dict__ and base not in bases:
                    bases.append(base)
    except Exception:
        return

    for base in bases:
        if getattr(base, "_minicpmo45_view_optimize_patched", False):
            continue
        original_setattr = base.__setattr__

        def tolerant_setattr(self, key, value, _original=original_setattr):
            if key == "enable_view_optimize":
                fixed = self.__dict__.get("_fixed_attrs")
                if fixed is not None and key not in fixed:
                    object.__setattr__(self, key, value)
                    return
            return _original(self, key, value)

        base.__setattr__ = tolerant_setattr
        base._minicpmo45_view_optimize_patched = True


_minicpmo45_allow_view_optimize_option()
