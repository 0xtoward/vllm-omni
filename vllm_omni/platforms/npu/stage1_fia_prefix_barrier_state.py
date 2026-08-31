"""Pure-Python contract for the Stage1 fixed192 prefix-barrier Gate0.

The state object owns no torch/NPU resources.  It proves that one real graph
barrier may be released only after all captured FIA task-update records have
been observed in their original order.  Any partial or malformed epoch is a
permanent poison condition for the worker.
"""

from __future__ import annotations

from dataclasses import dataclass


EXPECTED_TASKS = 20


class PrefixBarrierContractError(RuntimeError):
    """The graph/update ordering contract was violated."""


@dataclass
class PrefixBarrierState:
    capture_sites: int = 0
    capture_wait_calls: int = 0
    capture_reset_calls: int = 0
    task_epochs_started: int = 0
    task_epochs_succeeded: int = 0
    task_epochs_failed: int = 0
    record_only_requests: int = 0
    release_commits: int = 0
    poisoned: bool = False
    poison_reason: str | None = None
    _capture_wait_mask: int = 0
    _capture_reset_mask: int = 0
    _epoch_kind: str | None = None
    _next_task_site: int = 0
    _release_prepared: bool = False
    _record_only_pending: bool = False

    def _fail(self, reason: str) -> PrefixBarrierContractError:
        self.poisoned = True
        self.poison_reason = reason
        return PrefixBarrierContractError(reason)

    def assert_healthy(self) -> None:
        if self.poisoned:
            raise PrefixBarrierContractError(
                f"prefix-barrier state is poisoned: {self.poison_reason}"
            )

    def allocate_capture_site(self) -> int:
        self.assert_healthy()
        if self.capture_sites >= EXPECTED_TASKS:
            raise self._fail("capture emitted more than 20 FIA sites")
        site = self.capture_sites
        self.capture_sites += 1
        return site

    def capture_wait(self, site: int) -> bool:
        self.assert_healthy()
        self._capture_call(site, is_wait=True)
        return site == 0

    def capture_reset(self, site: int) -> bool:
        self.assert_healthy()
        self._capture_call(site, is_wait=False)
        return site == 0

    def _capture_call(self, site: int, *, is_wait: bool) -> None:
        if not 0 <= site < EXPECTED_TASKS:
            raise self._fail(f"invalid capture site {site}")
        bit = 1 << site
        mask_name = "_capture_wait_mask" if is_wait else "_capture_reset_mask"
        mask = getattr(self, mask_name)
        if mask & bit:
            raise self._fail(
                f"duplicate capture {'wait' if is_wait else 'reset'} at site {site}"
            )
        setattr(self, mask_name, mask | bit)
        if is_wait:
            self.capture_wait_calls += 1
        else:
            self.capture_reset_calls += 1

    def abort_capture(self, reason: str) -> None:
        raise self._fail(f"capture aborted: {reason}")

    def assert_capture_complete(self) -> None:
        self.assert_healthy()
        full_mask = (1 << EXPECTED_TASKS) - 1
        if (
            self.capture_sites != EXPECTED_TASKS
            or self.capture_wait_calls != EXPECTED_TASKS
            or self.capture_reset_calls != EXPECTED_TASKS
            or self._capture_wait_mask != full_mask
            or self._capture_reset_mask != full_mask
        ):
            raise self._fail(
                "capture census mismatch: "
                f"sites={self.capture_sites} waits={self.capture_wait_calls} "
                f"resets={self.capture_reset_calls}"
            )

    def begin_task_epoch(self, kind: str) -> None:
        self.assert_capture_complete()
        if self._epoch_kind is not None:
            raise self._fail(f"nested task epoch: {self._epoch_kind} -> {kind}")
        self._epoch_kind = kind
        self._next_task_site = 0
        self._release_prepared = False
        self.task_epochs_started += 1

    def observe_task_record(self, site: int) -> None:
        self.assert_healthy()
        if self._epoch_kind is None:
            raise self._fail(f"task record outside epoch at site {site}")
        if self._release_prepared:
            raise self._fail("task record after release preparation")
        if site != self._next_task_site:
            raise self._fail(
                f"task record out of order: expected={self._next_task_site} got={site}"
            )
        self._next_task_site += 1

    def prepare_task_release(self) -> None:
        self.assert_healthy()
        if self._epoch_kind is None:
            raise self._fail("release preparation outside task epoch")
        if self._next_task_site != EXPECTED_TASKS:
            raise self._fail(
                "partial task epoch cannot release barrier: "
                f"updated={self._next_task_site}/{EXPECTED_TASKS}"
            )
        self._release_prepared = True

    def commit_task_release(self) -> None:
        self.assert_healthy()
        if self._epoch_kind is None or not self._release_prepared:
            raise self._fail("task release committed without preparation")
        self.task_epochs_succeeded += 1
        self.release_commits += 1
        self._clear_epoch()

    def abort_task_epoch(self, reason: str) -> None:
        if self._epoch_kind is not None:
            self.task_epochs_failed += 1
        self._epoch_kind = None
        self._next_task_site = 0
        self._release_prepared = False
        raise self._fail(f"task epoch aborted: {reason}")

    def begin_record_only_release(self) -> None:
        self.assert_capture_complete()
        if self._epoch_kind is not None:
            raise self._fail("record-only release during task epoch")
        if self._record_only_pending:
            raise self._fail("nested record-only release")
        self._record_only_pending = True
        self.record_only_requests += 1

    def commit_record_only_release(self) -> None:
        self.assert_healthy()
        if not self._record_only_pending:
            raise self._fail("record-only release committed without begin")
        self._record_only_pending = False
        self.release_commits += 1

    def abort_record_only_release(self, reason: str) -> None:
        self._record_only_pending = False
        raise self._fail(f"record-only release failed: {reason}")

    def _clear_epoch(self) -> None:
        self._epoch_kind = None
        self._next_task_site = 0
        self._release_prepared = False
