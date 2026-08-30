# SPDX-License-Identifier: Apache-2.0
"""Skip request-step NPU allocator queries when DEBUG logging is disabled.

The installed vLLM-Ascend worker calls ``profile_memory`` before every model
step.  On supported revisions the queried fields are consumed only by the
DEBUG log in that same method.  Preserve the original method in DEBUG mode and
fail open on unknown revisions.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

_ENV = "VLLM_OMNI_ASCEND_SKIP_STEP_MEMORY_PROFILE"
_PATCH_MARKER = "_vllm_omni_skip_step_memory_profile_v1"


def _enabled() -> bool:
    return os.environ.get(_ENV, "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def install_npu_worker_step_memory_profile_guard() -> None:
    """Install an idempotent, revision-guarded worker patch."""
    if not _enabled():
        return

    try:
        import vllm_ascend.worker.worker as worker_module
        from vllm_ascend.worker.worker import NPUWorker
    except Exception:
        # This plugin is loaded in non-NPU and orchestration processes too.
        return

    original = NPUWorker.profile_memory
    if getattr(original, _PATCH_MARKER, False):
        return

    names = set(getattr(getattr(original, "__code__", None), "co_names", ()))
    supported_query = (
        {"memory_reserved", "memory_allocated"}.issubset(names)
        or "memory_stats" in names
    )
    if not supported_query or "isEnabledFor" not in names:
        worker_module.logger.warning(
            "MiniCPMO45NPUWorkerMemoryProfilePatch skipped: unsupported "
            "profile_memory implementation names=%s",
            sorted(names),
        )
        return

    @functools.wraps(original)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        if worker_module.logger.isEnabledFor(logging.DEBUG):
            return original(self, *args, **kwargs)
        return None

    setattr(guarded, _PATCH_MARKER, True)
    setattr(guarded, "_vllm_omni_original", original)
    NPUWorker.profile_memory = guarded
    worker_module.logger.info(
        "MiniCPMO45NPUWorkerMemoryProfilePatch enabled env=%s debug_preserved=true",
        _ENV,
    )
