# SPDX-License-Identifier: Apache-2.0
"""Pure-Python state and descriptor planning for the fixed192 FIA candidate."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

BLOCK_SIZE = 128
FIXED_CAPACITY = 256
MASK_WIDTH = 256


class UpdateAction(str, Enum):
    FIXED_REBIND = "FIXED_REBIND"
    FIXED_RECORD_ONLY = "FIXED_RECORD_ONLY"
    STOCK_DYNAMIC = "STOCK_DYNAMIC"


@dataclass(frozen=True)
class Observation:
    eligible: bool
    live_len: int | None
    reason: str = "eligible"


@dataclass
class Fixed192StateMachine:
    fixed_bound: bool = False
    last_live_len: int | None = None
    counters: Counter[str] = field(default_factory=Counter)

    def decide(self, observation: Observation) -> UpdateAction:
        if not observation.eligible or observation.live_len is None:
            return UpdateAction.STOCK_DYNAMIC
        live_len = observation.live_len
        if live_len < 1 or live_len > FIXED_CAPACITY:
            return UpdateAction.STOCK_DYNAMIC
        if not self.fixed_bound:
            return UpdateAction.FIXED_REBIND
        if self.last_live_len is not None and live_len < self.last_live_len:
            return UpdateAction.FIXED_REBIND
        return UpdateAction.FIXED_RECORD_ONLY

    def commit(
        self,
        observation: Observation,
        action: UpdateAction,
        *,
        success: bool,
    ) -> None:
        self.counters[f"decision.{action.value}"] += 1
        if not success:
            self.fixed_bound = False
            self.last_live_len = None
            self.counters["commit.failure"] += 1
            return
        if action is UpdateAction.STOCK_DYNAMIC:
            self.fixed_bound = False
            self.last_live_len = observation.live_len
            self.counters[f"fallback.{observation.reason}"] += 1
            return
        if action is UpdateAction.FIXED_REBIND:
            if (
                self.last_live_len is not None
                and observation.live_len is not None
                and observation.live_len < self.last_live_len
            ):
                self.counters["rebind.sequence_decrease"] += 1
            self.fixed_bound = True
            self.counters["commit.fixed_rebind"] += 1
        else:
            self.counters["commit.fixed_record_only"] += 1
        self.last_live_len = observation.live_len


def build_mask_row(live_len: int) -> bytes:
    if not 1 <= live_len <= FIXED_CAPACITY:
        raise ValueError(f"live_len must be in [1,{FIXED_CAPACITY}], got {live_len}")
    return bytes(live_len) + bytes([1]) * (MASK_WIDTH - live_len)


def normalize_block_table(live_len: int, source_row: list[int]) -> tuple[int, int]:
    """Return a valid two-column table for the fixed192 descriptor.

    The inactive second column is deliberately duplicated from column zero.
    Gate0 also exercises zero and poisoned valid physical IDs, but duplication
    avoids depending on the allocator's padding sentinel in the service patch.
    """

    if not 1 <= live_len <= FIXED_CAPACITY:
        raise ValueError(f"unsupported live_len={live_len}")
    if not source_row or source_row[0] < 0:
        raise ValueError("source block table has no valid first physical block")
    first = int(source_row[0])
    if live_len <= BLOCK_SIZE:
        return first, first
    if len(source_row) < 2 or source_row[1] < 0:
        raise ValueError("live_len > 128 requires a valid second physical block")
    return first, int(source_row[1])
