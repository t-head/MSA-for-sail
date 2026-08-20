# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""CuTe-free runtime wrapper for sparse k2q schedule preparation.

This module replaces the CuTe-dependent schedule-preparation host-side
paths for the PPU backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .sparse_jit import get_sparse_prepare_schedule_module
from .sparse_schedule_model import SPARSE_SCHEDULE_MODEL

_SUPPORTED_TOPK = (4, 8, 16, 32)
_SUPPORTED_BLK_KV = 128


@dataclass
class SparseAttentionSchedule:
    """CuTe-free equivalent of ``prepare_scheduler.SparseAttentionSchedule``."""

    enabled: bool
    scheduler_metadata: Optional[torch.Tensor]
    work_count: Optional[torch.Tensor]
    qsplit_indices: Optional[torch.Tensor] = None
    split_counts: Optional[torch.Tensor] = None
    target_q_per_cta: int = 0

    @property
    def work_capacity(self) -> int:
        return 0 if self.scheduler_metadata is None else int(self.scheduler_metadata.shape[0])


def _validate_cu_seqlens(cu_seqlens: torch.Tensor, *, name: str) -> None:
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32, got {cu_seqlens.dtype}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {tuple(cu_seqlens.shape)}")
    if cu_seqlens.numel() < 1:
        raise ValueError(f"{name} must have at least one element")
    if not cu_seqlens.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def prepare_sparse_schedule(
    *,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_q: int,
    max_seqlen_q: int,
    topk: int,
    head_kv: int,
    qhead_per_kv: int,
    blk_kv: int = 128,
    device: torch.device | None = None,
    enabled: bool = True,
    usable_SM_count: int = -1,
    capture_safe: bool = False,
    out_scheduler_metadata: Optional[torch.Tensor] = None,
    out_work_count: Optional[torch.Tensor] = None,
    out_qsplit_indices: Optional[torch.Tensor] = None,
    out_split_counts: Optional[torch.Tensor] = None,
    target_q_per_cta_override: Optional[int] = None,
    work_capacity_override: Optional[int] = None,
) -> SparseAttentionSchedule:
    """Build the flat worklist and split metadata for sparse forward.

    This is the CuTe-free replacement for
    ``prepare_sparse_fwd_schedule_and_split``.

    When ``capture_safe=True``, skips all host validation and ``.item()``
    calls. Output tensors must be pre-allocated and passed via ``out_*``;
    if not provided, they are allocated eagerly (which will fail inside
    a CUDA graph capture).
    """
    if not enabled:
        return SparseAttentionSchedule(enabled=False, scheduler_metadata=None, work_count=None)

    if not capture_safe:
        if blk_kv != _SUPPORTED_BLK_KV:
            raise ValueError(
                f"sparse_k2q_runtime only supports blk_kv == {_SUPPORTED_BLK_KV}, got {blk_kv}"
            )
        if topk not in _SUPPORTED_TOPK:
            raise ValueError(
                f"sparse_k2q_runtime only supports topK in {_SUPPORTED_TOPK}, got {topk}"
            )
        if topk > 255:
            raise ValueError(f"packed qsplit metadata supports topK <= 255, got {topk}")
        if max_seqlen_q >= (1 << 24):
            raise ValueError(
                "packed qsplit metadata supports batch-local q_idx < 2^24, "
                f"got max_seqlen_q={max_seqlen_q}"
            )

        if k2q_row_ptr.dtype != torch.int32:
            raise TypeError(f"k2q_row_ptr must be torch.int32, got {k2q_row_ptr.dtype}")
        if k2q_row_ptr.ndim != 2:
            raise ValueError(
                f"k2q_row_ptr must be rank-2 [head_kv, total_rows + 1], got shape {tuple(k2q_row_ptr.shape)}"
            )
        if not k2q_row_ptr.is_contiguous():
            raise ValueError("k2q_row_ptr must be contiguous")

        if k2q_q_indices.dtype != torch.int32:
            raise TypeError(f"k2q_q_indices must be torch.int32, got {k2q_q_indices.dtype}")
        if k2q_q_indices.ndim != 2:
            raise ValueError(
                f"k2q_q_indices must be rank-2 [head_kv, nnz], got shape {tuple(k2q_q_indices.shape)}"
            )
        if not k2q_q_indices.is_contiguous():
            raise ValueError("k2q_q_indices must be contiguous")

        _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q")
        _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape [B + 1]")

        if device is None:
            device = k2q_row_ptr.device

        if not (k2q_row_ptr.is_cuda and k2q_q_indices.is_cuda and cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda):
            raise ValueError("all inputs must be CUDA tensors")
        if (
            k2q_row_ptr.device != device
            or k2q_q_indices.device != device
            or cu_seqlens_q.device != device
            or cu_seqlens_k.device != device
        ):
            raise ValueError("all inputs must be on the same device")

        if total_q != int(cu_seqlens_q[-1].item()):
            raise ValueError(
                f"total_q ({total_q}) must equal cu_seqlens_q[-1] ({int(cu_seqlens_q[-1].item())})"
            )
        if qhead_per_kv <= 0:
            raise ValueError(f"qhead_per_kv must be positive, got {qhead_per_kv}")

    head_kv = int(head_kv)
    qhead_per_kv = int(qhead_per_kv)
    topk = int(topk)
    total_q = int(total_q)
    max_seqlen_q = int(max_seqlen_q)

    if device is None:
        device = k2q_row_ptr.device

    total_rows = int(k2q_row_ptr.size(1) - 1)
    if total_rows <= 0 or total_q == 0 or head_kv == 0 or topk == 0:
        scheduler_metadata = torch.zeros((1, 6), dtype=torch.int32, device=device)
        work_count = torch.zeros((1,), dtype=torch.int32, device=device)
        qsplit_indices = torch.empty_like(k2q_q_indices)
        split_counts = torch.zeros((total_q, head_kv), dtype=torch.int32, device=device)
        return SparseAttentionSchedule(
            enabled=True,
            scheduler_metadata=scheduler_metadata,
            work_count=work_count,
            qsplit_indices=qsplit_indices,
            split_counts=split_counts,
            target_q_per_cta=1,
        )

    if target_q_per_cta_override is not None:
        target_q_per_cta = target_q_per_cta_override
    else:
        target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
            total_q=total_q,
            topk=topk,
            blk_kv=blk_kv,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            device=device,
            usable_SM_count=usable_SM_count,
        )
    if work_capacity_override is not None:
        work_capacity = work_capacity_override
    else:
        work_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
            total_rows=total_rows,
            total_q=total_q,
            topk=topk,
            head_kv=head_kv,
            target_q_per_cta=target_q_per_cta,
        )

    if out_scheduler_metadata is not None:
        scheduler_metadata = out_scheduler_metadata
    else:
        scheduler_metadata = torch.zeros(
            (work_capacity, 6), dtype=torch.int32, device=device
        )
    if out_work_count is not None:
        work_count = out_work_count
    else:
        work_count = torch.zeros((1,), dtype=torch.int32, device=device)
    if out_qsplit_indices is not None:
        qsplit_indices = out_qsplit_indices
    else:
        qsplit_indices = torch.empty_like(k2q_q_indices)
    if out_split_counts is not None:
        split_counts = out_split_counts
    else:
        split_counts = torch.empty((total_q, head_kv), dtype=torch.int32, device=device)

    module = get_sparse_prepare_schedule_module()
    module.sparse_prepare_schedule(
        k2q_row_ptr,
        k2q_q_indices,
        cu_seqlens_k,
        cu_seqlens_q,
        scheduler_metadata,
        work_count,
        qsplit_indices,
        split_counts,
        target_q_per_cta,
        work_capacity,
        max_seqlen_q,
        topk,
        total_q,
        torch.cuda.current_stream().cuda_stream,
    )

    return SparseAttentionSchedule(
        enabled=True,
        scheduler_metadata=scheduler_metadata,
        work_count=work_count,
        qsplit_indices=qsplit_indices,
        split_counts=split_counts,
        target_q_per_cta=target_q_per_cta,
    )
