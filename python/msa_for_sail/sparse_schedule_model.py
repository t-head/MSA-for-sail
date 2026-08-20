# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""CuTe-free host-side sizing model for sparse attention scheduling.
"""

from __future__ import annotations

import torch


class SparseAttentionScheduleModel:
    """Host-side helpers for sparse attention schedule sizing."""

    @staticmethod
    def _round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    @staticmethod
    def _ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y

    def balanced_target_q_per_cta(
        self,
        *,
        total_q: int,
        topk: int,
        blk_kv: int,
        head_kv: int,
        qhead_per_kv: int,
        device: torch.device,
        usable_SM_count: int = -1,
        num_sm_override: int | None = None,
    ) -> int:
        q_tokens_per_group = 128 // qhead_per_kv
        target = max(q_tokens_per_group, 2 * 64)
        sink_balance_cap = max(q_tokens_per_group, int(topk) * int(blk_kv) * 2)
        target = min(target, sink_balance_cap)
        return self._round_up(target, q_tokens_per_group)

    def flat_schedule_capacity(
        self,
        *,
        total_rows: int,
        total_q: int,
        topk: int,
        head_kv: int,
        target_q_per_cta: int,
    ) -> int:
        row_upper = max(total_rows, 0) * max(head_kv, 1)
        refs_upper = max(total_q, 0) * max(topk, 1) * max(head_kv, 1)
        split_upper = self._ceil_div(max(refs_upper, 1), max(target_q_per_cta, 1))
        return max(1, row_upper + split_upper)


SPARSE_SCHEDULE_MODEL = SparseAttentionScheduleModel()
