# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Drop-in adapter: msa_for_sail API → sparse_atten_func backend.

Usage:
    from msa_for_sail.sparse_fmha_adapter import sparse_fmha_plan, sparse_fmha

    plan_info = sparse_fmha_plan(qo_lens, kv_lens, num_qo_heads, ...)
    out, _ = sparse_fmha(q, k, v, plan_info=plan_info,
                         kv_indices=kv_indices,
                         kv_block_indexes=kv_block_indexes)

Backend selection:
    MINFER_SPARSE_BACKEND=cuda  (default when unset) — PPU CUDA kernels
    MINFER_SPARSE_BACKEND=cute  — fail-fast ("cute backend unsupported on PPU")
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Tuple

import torch

from .jit import get_ppu_arch

logger = logging.getLogger(__name__)


def _get_backend() -> str:
    """Resolve MINFER_SPARSE_BACKEND.

    unset → "cuda" (adapter-layer default, §5.Phase D three-state table)
    "cuda" → "cuda"
    "cute" → raise immediately with clear message
    """
    val = os.environ.get("MINFER_SPARSE_BACKEND", "cuda").lower().strip()
    if val == "cute":
        raise RuntimeError(
            "MINFER_SPARSE_BACKEND=cute is unsupported on PPU. "
            "cute backend requires SM100 CuTe DSL which is not available on this device. "
            "Use MINFER_SPARSE_BACKEND=cuda (or unset, which defaults to cuda)."
        )
    if val != "cuda":
        raise ValueError(
            f"Unknown MINFER_SPARSE_BACKEND value: {val!r}. "
            "Supported values: 'cuda' (default), 'cute' (PPU: fail-fast)."
        )
    return "cuda"


def _build_page_table(
    kv_indices: torch.Tensor,
    kv_segment_lens,
    page_size: int,
    batch: int,
) -> torch.Tensor:
    """Build page_table [batch, max_pages_per_seq] in eager phase.

    kv_segment_lens must be a CPU list/tensor so .tolist() is free of CUDA
    device→host sync (graph-safe when called before capture region).
    """
    if isinstance(kv_segment_lens, torch.Tensor):
        kv_lens = kv_segment_lens.cpu().tolist()
    else:
        kv_lens = list(kv_segment_lens)

    pages_per_batch = [(int(kl) + page_size - 1) // page_size for kl in kv_lens]
    max_pages = max(pages_per_batch) if pages_per_batch else 1
    total = batch * max_pages
    buf = torch.zeros(total + 4, dtype=torch.int32, device=kv_indices.device)
    shift = ((-buf.data_ptr()) % 16) // 4
    page_table = buf[shift: shift + total].view(batch, max_pages)
    assert page_table.data_ptr() % 16 == 0
    offset = 0
    for b in range(batch):
        n = pages_per_batch[b]
        page_table[b, :n] = kv_indices[offset: offset + n].to(torch.int32)
        offset += n
    return page_table


def _build_kv_page_indptr(kv_segment_lens, page_size: int) -> torch.Tensor:
    """Build kv_page_indptr [batch+1] for the fwd kernel (eager phase)."""
    if isinstance(kv_segment_lens, torch.Tensor):
        kv_lens = kv_segment_lens.cpu().tolist()
    else:
        kv_lens = list(kv_segment_lens)
    pages = [0]
    for kl in kv_lens:
        pages.append(pages[-1] + (int(kl) + page_size - 1) // page_size)
    return torch.tensor(pages, dtype=torch.int32)


def _convert_kv_block_indexes_to_q2k(
    kv_block_indexes: torch.Tensor,
    num_kv_heads: int,
    num_qo_heads: int,
    qhead_per_kv: int,
) -> torch.Tensor:
    """kv_block_indexes [total_q, H, topk] → q2k [H_kv, total_q, topk]."""
    h_dim = kv_block_indexes.shape[1]
    if h_dim == num_kv_heads:
        q2k = kv_block_indexes.permute(1, 0, 2).contiguous()
    elif h_dim == num_qo_heads and num_qo_heads != num_kv_heads:
        q2k = kv_block_indexes[:, ::qhead_per_kv, :].permute(1, 0, 2).contiguous()
    else:
        raise ValueError(
            f"kv_block_indexes head dim {h_dim} doesn't match "
            f"num_kv_heads={num_kv_heads} or num_qo_heads={num_qo_heads}"
        )
    return q2k.to(torch.int32)


_GRAPH_WS_ALIGN_BYTES = 16


def _sparse_graph_ws_layout(batch, head_kv, total_q_max, total_rows_max, topk,
                            capacity_max):
    """Ordered slot layout for the graph-safe workspace.

    plan-time slicing and sparse_fmha_workspace_bytes() must share this layout
    byte-for-byte. Slots carry per-replay metadata (cu_seqlens,
    kv_page_indptr, seqused_k), staging for the device-side metadata ops, and
    every kernel output the eager path pre-allocates.
    """
    nnz_per_head = total_q_max * topk
    return [
        ("_ws_cu_seqlens_q", (batch + 1,), torch.int32),
        ("_ws_cu_seqlens_k", (batch + 1,), torch.int32),
        ("_ws_kv_page_indptr", (batch + 1,), torch.int32),
        ("_ws_seqused_k", (batch,), torch.int32),
        ("_ws_qo_lens_stage", (batch,), torch.int32),
        ("_ws_kv_lens_stage", (batch,), torch.int32),
        ("_ws_pages_stage", (batch,), torch.int32),
        ("_ws_qo_offset_gpu", (batch,), torch.int32),
        ("_ws_row_index", (total_q_max,), torch.int32),
        ("_ws_pad_mask", (total_q_max,), torch.bool),
        ("_ws_row_ptr", (head_kv, total_rows_max + 1), torch.int32),
        ("_ws_q_indices", (head_kv, nnz_per_head), torch.int32),
        ("_ws_q2k", (head_kv, total_q_max, topk), torch.int32),
        ("_ws_scheduler_metadata", (capacity_max, 6), torch.int32),
        ("_ws_work_count", (1,), torch.int32),
        ("_ws_qsplit_indices", (head_kv, nnz_per_head), torch.int32),
        ("_ws_split_counts", (total_q_max, head_kv), torch.int32),
        ("_ws_page_table_sentinel", (1,), torch.int32),
    ]


def _slice_graph_workspace(ws_bytes, layout):
    """Slice a contiguous uint8 workspace into typed slot views."""
    out = {}
    off = 0
    for name, shape, dtype in layout:
        itemsize = torch.empty((), dtype=dtype).element_size()
        nbytes = math.prod(shape) * itemsize
        if off + nbytes > ws_bytes.shape[0]:
            raise ValueError(
                f"given_workspace too small: {ws_bytes.shape[0]} bytes < "
                f"required {off + nbytes} bytes. Re-allocate with "
                "sparse_fmha_workspace_bytes(...).")
        out[name] = ws_bytes[off:off + nbytes].view(dtype).view(*shape)
        off += ((nbytes + _GRAPH_WS_ALIGN_BYTES - 1)
                // _GRAPH_WS_ALIGN_BYTES) * _GRAPH_WS_ALIGN_BYTES
    return out


def _sparse_graph_ws_total_bytes(layout):
    total = 0
    for _name, shape, dtype in layout:
        itemsize = torch.empty((), dtype=dtype).element_size()
        nbytes = math.prod(shape) * itemsize
        total += ((nbytes + _GRAPH_WS_ALIGN_BYTES - 1)
                  // _GRAPH_WS_ALIGN_BYTES) * _GRAPH_WS_ALIGN_BYTES
    return total


def _graph_ws_sizing(batch, num_qo_heads, num_kv_heads, max_kv_len,
                     max_total_q, kv_block_num, page_size, usable_SM_count):
    """Host-only static sizing shared by plan and sparse_fmha_workspace_bytes."""
    head_kv = num_kv_heads
    qhead_per_kv = num_qo_heads // head_kv
    blk_kv = page_size
    total_rows_max = batch * ((max_kv_len + blk_kv - 1) // blk_kv)
    from .sparse_schedule_model import SPARSE_SCHEDULE_MODEL
    target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
        total_q=max_total_q,
        topk=kv_block_num,
        blk_kv=blk_kv,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        device=torch.device("cuda"),
        usable_SM_count=usable_SM_count,
    )
    capacity_max = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
        total_rows=total_rows_max,
        total_q=max_total_q,
        topk=kv_block_num,
        head_kv=head_kv,
        target_q_per_cta=target_q_per_cta,
    )
    layout = _sparse_graph_ws_layout(
        batch, head_kv, max_total_q, total_rows_max, kv_block_num, capacity_max)
    return layout, {
        "total_rows_max": total_rows_max,
        "target_q_per_cta": target_q_per_cta,
        "capacity_max": capacity_max,
    }


def sparse_fmha_workspace_bytes(
    batch: int,
    num_qo_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
    max_total_q: int,
    kv_block_num: int,
    page_size: int = 128,
    usable_SM_count: int = -1,
) -> int:
    """Byte size for the uint8 given_workspace required by graph_safe plans.

    Size from STATIC upper bounds — the same bounds passed to
    sparse_fmha_plan(graph_safe=True, ...). The caller allocates once::

        ws = torch.empty(sparse_fmha_workspace_bytes(...), dtype=torch.uint8,
                         device="cuda")

    and may reuse it across plans/re-captures that share the same bounds.
    """
    if num_kv_heads <= 0:
        num_kv_heads = num_qo_heads
    layout, _ = _graph_ws_sizing(
        batch, num_qo_heads, num_kv_heads, max_kv_len, max_total_q,
        kv_block_num, page_size, usable_SM_count)
    return _sparse_graph_ws_total_bytes(layout)


def _graph_meta_recompute(plan_info):
    """Refresh graph-safe plan metadata from the CURRENT lens contents.

    Allocation-free, sync-free device ops only: safe inside a CUDA graph
    capture and on every replay, so a captured run picks up new
    qo/kv_segment_lens contents (within the static upper bounds fixed at
    plan time).
    """
    qo_stage = plan_info["_ws_qo_lens_stage"]
    kv_stage = plan_info["_ws_kv_lens_stage"]
    qo_stage.copy_(plan_info["qo_segment_lens"])
    kv_stage.copy_(plan_info["kv_segment_lens"])

    cu_q = plan_info["_ws_cu_seqlens_q"]
    cu_k = plan_info["_ws_cu_seqlens_k"]
    cu_q[0].zero_()
    torch.cumsum(qo_stage, dim=0, out=cu_q[1:])
    cu_k[0].zero_()
    torch.cumsum(kv_stage, dim=0, out=cu_k[1:])

    pages = plan_info["_ws_pages_stage"]
    pages.copy_(kv_stage)
    pages.add_(plan_info["blk_kv"] - 1)
    pages.div_(plan_info["blk_kv"], rounding_mode="floor")
    indptr = plan_info["_ws_kv_page_indptr"]
    indptr[0].zero_()
    torch.cumsum(pages, dim=0, out=indptr[1:])

    seqused = plan_info["_ws_seqused_k"]
    qo_offset = plan_info.get("_qo_offset_src")
    if qo_offset is None:
        seqused.copy_(kv_stage)
    else:
        off = plan_info["_ws_qo_offset_gpu"]
        off.copy_(qo_offset)
        seqused.copy_(qo_stage)
        seqused.add_(off)

    torch.ge(plan_info["_ws_row_index"], cu_q[-1], out=plan_info["_ws_pad_mask"])


def _sparse_fmha_plan_graph_safe(
    qo_segment_lens, kv_segment_lens, num_qo_heads, num_kv_heads, qo_offset,
    page_size, kv_block_num, usable_SM_count, use_fp8_kvcache, backend,
    graph_safe, max_qo_len_override, max_kv_len_override, max_total_q,
    given_workspace, stream,
):
    """Graph-safe plan branch: static-bound sizing, workspace-injected buffers.

    Every host value derives from the caller's static upper bounds (no device
    read), every buffer is a view into given_workspace, and per-replay
    metadata is recomputed on-device (see _graph_meta_recompute) so plan+run
    are CUDA-graph capturable with lens contents free to change on replay.
    """
    if max_qo_len_override <= 0:
        raise ValueError("graph_safe=True requires max_qo_len_override > 0")
    if max_kv_len_override <= 0:
        raise ValueError("graph_safe=True requires max_kv_len_override > 0")
    if max_total_q <= 0:
        raise ValueError("graph_safe=True requires max_total_q > 0")
    if given_workspace is None:
        raise ValueError("graph_safe=True requires given_workspace (pre-allocated)")
    if not qo_segment_lens.is_cuda or not kv_segment_lens.is_cuda:
        raise ValueError("graph_safe=True requires qo_segment_lens/kv_segment_lens on CUDA")
    if isinstance(qo_offset, int):
        raise ValueError(
            "graph_safe=True requires qo_offset as a CUDA tensor; an int "
            "triggers an in-capture allocation")
    if qo_offset is not None and not qo_offset.is_cuda:
        raise ValueError("graph_safe=True requires qo_offset tensor on CUDA")
    if max_total_q >= (1 << 24):
        raise ValueError(
            f"graph_safe=True requires max_total_q < 2^24 (qsplit packing), got {max_total_q}")
    if qo_offset is not None and qo_offset.shape[0] != qo_segment_lens.shape[0]:
        raise ValueError(
            f"qo_offset has {qo_offset.shape[0]} entries, expected batch={qo_segment_lens.shape[0]}")

    batch = qo_segment_lens.shape[0]
    if kv_segment_lens.shape[0] != batch:
        raise ValueError(
            f"kv_segment_lens has {kv_segment_lens.shape[0]} entries, expected batch={batch}")
    gpu_device = qo_segment_lens.device
    head_kv = num_kv_heads
    qhead_per_kv = num_qo_heads // head_kv
    blk_kv = page_size
    total_q = max_total_q
    max_seqlen_q = max_qo_len_override
    max_seqlen_k = max_kv_len_override
    total_rows = batch * ((max_kv_len_override + blk_kv - 1) // blk_kv)
    total_k = batch * max_kv_len_override

    layout, sizing = _graph_ws_sizing(
        batch, num_qo_heads, head_kv, max_kv_len_override, total_q,
        kv_block_num, blk_kv, usable_SM_count)
    ws = _slice_graph_workspace(given_workspace, layout)

    plan_info = {
        "backend": backend,
        "graph_safe": True,
        "stream": stream,
        "qo_segment_lens": qo_segment_lens,
        "kv_segment_lens": kv_segment_lens,
        "kv_segment_lens_cpu": None,
        "cu_seqlens_q": ws["_ws_cu_seqlens_q"],
        "qo_segment_offsets": ws["_ws_cu_seqlens_q"],
        "cu_seqlens_k": ws["_ws_cu_seqlens_k"],
        "seqused_k": ws["_ws_seqused_k"],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "total_k": total_k,
        "total_rows": total_rows,
        "total_q": total_q,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "qhead_per_kv": qhead_per_kv,
        "page_size": page_size,
        "blk_kv": blk_kv,
        "kv_block_num": kv_block_num,
        "causal": True,
        "batch": batch,
        "MM-SA-Nv": True,
        "use_fp8_kvcache": use_fp8_kvcache,
        "usable_SM_count": usable_SM_count,
        "target_q_per_cta": sizing["target_q_per_cta"],
        "scheduler_metadata_capacity": sizing["capacity_max"],
        "kv_page_indptr_cpu": None,
        "page_table": ws["_ws_page_table_sentinel"],
        "_kv_page_indptr_gpu": ws["_ws_kv_page_indptr"],
        "_row_ptr": ws["_ws_row_ptr"],
        "_q_indices": ws["_ws_q_indices"],
        "_q2k": ws["_ws_q2k"],
        "_scheduler_metadata": ws["_ws_scheduler_metadata"],
        "_work_count": ws["_ws_work_count"],
        "_qsplit_indices": ws["_ws_qsplit_indices"],
        "_split_counts": ws["_ws_split_counts"],
        "_qo_offset_src": qo_offset,
        "_ws_qo_lens_stage": ws["_ws_qo_lens_stage"],
        "_ws_kv_lens_stage": ws["_ws_kv_lens_stage"],
        "_ws_pages_stage": ws["_ws_pages_stage"],
        "_ws_qo_offset_gpu": ws["_ws_qo_offset_gpu"],
        "_ws_cu_seqlens_q": ws["_ws_cu_seqlens_q"],
        "_ws_cu_seqlens_k": ws["_ws_cu_seqlens_k"],
        "_ws_kv_page_indptr": ws["_ws_kv_page_indptr"],
        "_ws_seqused_k": ws["_ws_seqused_k"],
        "_ws_row_index": ws["_ws_row_index"],
        "_ws_pad_mask": ws["_ws_pad_mask"],
    }

    def _init():
        ws["_ws_q2k"].fill_(-1)
        torch.arange(total_q, out=ws["_ws_row_index"])
        ws["_ws_work_count"].zero_()
        ws["_ws_scheduler_metadata"].zero_()
        ws["_ws_split_counts"].zero_()
        _graph_meta_recompute(plan_info)

    if stream is not None:
        _s = stream if isinstance(stream, torch.cuda.Stream) \
            else torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(_s):
            _init()
    else:
        _init()
    return plan_info


def sparse_fmha_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int = -1,
    qo_offset: Optional[torch.Tensor] = None,
    num_kv_splits: int = -1,
    page_size: int = -1,
    output_maxscore: bool = False,
    kv_block_num: int = -1,
    causal: bool = True,
    usable_SM_count=-1,
    use_fp8_kvcache: bool = False,
    graph_safe: bool = False,
    max_qo_len_override: int = -1,
    max_kv_len_override: int = -1,
    max_total_q: int = -1,
    given_workspace: Optional[torch.Tensor] = None,
    stream=None,
) -> dict:
    """Build a reusable sparse-prefill plan.

    cuda backend: pre-allocates graph-safe buffers and caches page_table.
    cute backend: raises immediately (unsupported on PPU).
    """
    backend = _get_backend()

    use_fp8_kvcache = bool(use_fp8_kvcache)
    if use_fp8_kvcache and get_ppu_arch() != "ppu_15":
        raise ValueError(
            "use_fp8_kvcache=True requires PPU0015; fp8 sparse FMHA is not "
            f"supported on {get_ppu_arch()} (use bf16 K/V)"
        )

    assert kv_block_num in {4, 8, 16, 32}, f"kv_block_num={kv_block_num} not in {{4,8,16,32}}"
    assert page_size == 128, (
        f"sparse_fmha_plan requires page_size=128 (CSR blk_kv), got {page_size}")
    assert not output_maxscore, "sparse_fmha_plan: output_maxscore must be False"
    assert causal, "sparse_fmha_plan: causal must be True"
    if num_kv_heads <= 0:
        num_kv_heads = num_qo_heads
    assert num_qo_heads % num_kv_heads == 0, (
        f"num_qo_heads={num_qo_heads} must be divisible by num_kv_heads={num_kv_heads}")

    batch = qo_segment_lens.shape[0]
    gpu_device = qo_segment_lens.device if qo_segment_lens.is_cuda else torch.device("cuda")
    blk_kv = page_size
    qhead_per_kv = num_qo_heads // num_kv_heads if num_kv_heads > 0 else 1

    if graph_safe:
        return _sparse_fmha_plan_graph_safe(
            qo_segment_lens, kv_segment_lens, num_qo_heads, num_kv_heads,
            qo_offset, page_size, kv_block_num, usable_SM_count,
            use_fp8_kvcache, backend, graph_safe, max_qo_len_override,
            max_kv_len_override, max_total_q, given_workspace, stream)

    qo_lens_cpu = qo_segment_lens.cpu()
    kv_lens_cpu = kv_segment_lens.cpu()

    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=gpu_device)
    cu_seqlens_q[1:] = torch.cumsum(qo_lens_cpu.to(torch.int32), dim=0).to(gpu_device)

    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=gpu_device)
    cu_seqlens_k[1:] = torch.cumsum(kv_lens_cpu.to(torch.int32), dim=0).to(gpu_device)

    # .item()/.tolist() done here (eager phase, before any capture region)
    max_seqlen_q = int(qo_lens_cpu.max().item())
    max_seqlen_k = int(kv_lens_cpu.max().item())
    total_k = int(kv_lens_cpu.sum().item())

    kv_lens_list = kv_lens_cpu.tolist()
    total_rows = sum((int(kl) + blk_kv - 1) // blk_kv for kl in kv_lens_list)

    if qo_offset is not None:
        seqused_k = (qo_lens_cpu + qo_offset.cpu()).to(torch.int32).to(gpu_device)
    else:
        seqused_k = kv_lens_cpu.to(torch.int32).to(gpu_device)

    total_q = int(cu_seqlens_q[-1].item())

    # Sizing (cute-free schedule model)
    from .sparse_schedule_model import SPARSE_SCHEDULE_MODEL
    target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
        total_q=total_q,
        topk=kv_block_num,
        blk_kv=page_size,
        head_kv=num_kv_heads,
        qhead_per_kv=qhead_per_kv,
        device=gpu_device,
        usable_SM_count=usable_SM_count,
    )
    scheduler_metadata_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
        total_rows=total_rows,
        total_q=total_q,
        topk=kv_block_num,
        head_kv=num_kv_heads,
        target_q_per_cta=target_q_per_cta,
    )

    # Pre-build kv_page_indptr (needed by fwd kernel, graph-safe: pure CPU)
    pages_per_batch = [(int(kl) + blk_kv - 1) // blk_kv for kl in kv_lens_list]
    kv_page_indptr_list = [0]
    for pp in pages_per_batch:
        kv_page_indptr_list.append(kv_page_indptr_list[-1] + pp)
    kv_page_indptr_cpu = torch.tensor(kv_page_indptr_list, dtype=torch.int32)

    head_kv = num_kv_heads
    row_ptr_buf = torch.empty((head_kv, total_rows + 1), dtype=torch.int32, device=gpu_device)
    nnz = head_kv * total_q * kv_block_num
    q_indices_buf = torch.empty((head_kv, nnz // head_kv), dtype=torch.int32, device=gpu_device)
    q2k_buf = torch.empty((head_kv, total_q, kv_block_num), dtype=torch.int32, device=gpu_device)

    scheduler_metadata_buf = torch.zeros(
        (scheduler_metadata_capacity, 6), dtype=torch.int32, device=gpu_device
    )
    work_count_buf = torch.zeros((1,), dtype=torch.int32, device=gpu_device)
    qsplit_indices_buf = torch.empty_like(q_indices_buf)
    split_counts_buf = torch.empty((total_q, head_kv), dtype=torch.int32, device=gpu_device)

    return {
        "backend": backend,
        "qo_segment_lens": qo_segment_lens,
        "kv_segment_lens_cpu": kv_lens_cpu,  # CPU copy for graph-safe page_table build
        "cu_seqlens_q": cu_seqlens_q,
        "qo_segment_offsets": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "kv_segment_lens": kv_segment_lens.to(torch.int32).to(gpu_device),
        "seqused_k": seqused_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "total_k": total_k,
        "total_rows": total_rows,
        "total_q": total_q,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "qhead_per_kv": qhead_per_kv,
        "page_size": page_size,
        "blk_kv": blk_kv,
        "kv_block_num": kv_block_num,
        "causal": causal,
        "batch": batch,
        "MM-SA-Nv": True,
        "use_fp8_kvcache": use_fp8_kvcache,
        "usable_SM_count": usable_SM_count,
        "target_q_per_cta": target_q_per_cta,
        "scheduler_metadata_capacity": scheduler_metadata_capacity,
        "kv_page_indptr_cpu": kv_page_indptr_cpu,
        "page_table": None,
        "_row_ptr": row_ptr_buf,
        "_q_indices": q_indices_buf,
        "_q2k": q2k_buf,
        "_scheduler_metadata": scheduler_metadata_buf,
        "_work_count": work_count_buf,
        "_qsplit_indices": qsplit_indices_buf,
        "_split_counts": split_counts_buf,
    }


# ---------------------------------------------------------------------------

def sparse_fmha(q, k, v, plan_info, *args, **kwargs):
    """Run sparse prefill through the selected backend (public entry).

    Thin stream-aware wrapper over _sparse_fmha_run: a plan built with
    stream=... (Phase G) launches its device work on that stream — accepts a
    torch.cuda.Stream or a raw cudaStream_t handle (int) — everything else is
    forwarded unchanged.
    """
    _stream = plan_info.get("stream") if isinstance(plan_info, dict) else None
    if _stream is not None:
        if not isinstance(_stream, torch.cuda.Stream):
            _stream = torch.cuda.ExternalStream(_stream)
        with torch.cuda.stream(_stream):
            return _sparse_fmha_run(q, k, v, plan_info, *args, **kwargs)
    return _sparse_fmha_run(q, k, v, plan_info, *args, **kwargs)


def _sparse_fmha_run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info: dict,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    o_scale: Optional[float] = None,
    kv_indices: Optional[torch.Tensor] = None,
    output_maxscore: bool = True,
    output_o: bool = True,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override=None,
    check_input_valid: bool = False,
    lse_temperature_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    return_temperature_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run sparse prefill through the selected backend.

    cuda backend:
      usable_SM_count <= 0 (default -1): build_k2q_csr_with_schedule (fused)
        → sparse_fmha_fwd → combine
      usable_SM_count > 0 (SM-limited): build_k2q_csr (CSR only)
        → two-stage prepare kernels (schedule rebuilt with SM-limited sizing)
        → sparse_fmha_fwd → combine
    cute backend: raises immediately

    Optional returns (mirror cute sparse_atten_func semantics):
      return_softmax_lse=True → (O, LSE)
      return_temperature_lse=True → (O, LSE, LSE_temperature);
        LSE_temperature aliases LSE when scale ≈ 1.0 (fast path).
        lse_temperature_scale is a divisor, matching cute:
        LSE_temperature = logsumexp(scores / lse_temperature_scale).
    """
    backend = plan_info.get("backend", _get_backend())
    if backend == "cute":
        raise RuntimeError(
            "cute backend unsupported on PPU. "
            "Use MINFER_SPARSE_BACKEND=cuda (or unset)."
        )

    if kv_block_indexes is None:
        raise ValueError("sparse_fmha requires kv_block_indexes")

    # Validate dtype combination through the runtime resolver.
    _ignored_scales = []
    for scale_name, scale_val in (
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
        ("o_scale", o_scale),
    ):
        if scale_val is not None and not math.isclose(scale_val, 1.0):
            _ignored_scales.append(f"{scale_name}={scale_val}")
    if _ignored_scales:
        logger.warning(
            "sparse_fmha: per-tensor scale(s) [%s] accepted but ignored "
            "(GPU-backend parity; only sm_scale is applied, quantization is "
            "the caller's responsibility).", ", ".join(_ignored_scales))

    if plan_info.get("use_fp8_kvcache", False) and k.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "plan_info requested use_fp8_kvcache=True but K/V dtype is "
            f"{k.dtype}; fp8 (float8_e4m3fn) K/V inputs are required for the "
            "fp8 KV-cache path."
        )

    if lse_temperature_scale is not None:
        lse_temperature_scale = float(lse_temperature_scale)
        if not math.isfinite(lse_temperature_scale) or lse_temperature_scale <= 0.0:
            raise ValueError(
                f"lse_temperature_scale must be finite and > 0, got {lse_temperature_scale}"
            )
    return_temperature_lse = bool(return_temperature_lse)
    if return_temperature_lse and not return_softmax_lse:
        raise ValueError("return_temperature_lse=True requires return_softmax_lse=True")
    if lse_temperature_scale is None or math.isclose(
        lse_temperature_scale, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        kernel_lse_temperature_scale = lse_temperature_scale
    else:
        kernel_lse_temperature_scale = 1.0 / lse_temperature_scale

    cu_seqlens_q = plan_info["cu_seqlens_q"]
    cu_seqlens_k = plan_info["cu_seqlens_k"]
    num_qo_heads = plan_info["num_qo_heads"]
    page_size = plan_info["page_size"]
    # Paged K/V may be NHD [pages, ps, Hkv, D] (default) or HND
    if page_size > 0 and k.ndim == 4:
        from .sparse_fwd_runtime import _detect_paged_kv_layout
        _, num_kv_heads = _detect_paged_kv_layout(k.shape, page_size)
    else:
        num_kv_heads = k.shape[1] if k.ndim >= 2 else plan_info["num_kv_heads"]
    if num_kv_heads != plan_info["num_kv_heads"]:
        raise ValueError(
            f"runtime K/V has {num_kv_heads} KV heads but the plan was built "
            f"for {plan_info['num_kv_heads']} per-rank heads (K shape="
            f"{tuple(k.shape)}, page_size={page_size})")
    qhead_per_kv = num_qo_heads // num_kv_heads
    blk_kv = plan_info["blk_kv"]
    topk = plan_info["kv_block_num"]
    causal = plan_info["causal"]
    max_seqlen_q = plan_info["max_seqlen_q"]
    max_seqlen_k = plan_info["max_seqlen_k"]
    total_k = plan_info["total_k"]
    total_rows = plan_info["total_rows"]
    total_q = plan_info["total_q"]
    batch = plan_info["batch"]
    usable_SM_count = int(plan_info.get("usable_SM_count", -1))
    head_dim = q.shape[-1]
    if head_dim != 128:
        raise ValueError(
            f"sparse_fmha only supports head_dim=128, got {head_dim}"
        )

    device = q.device

    if plan_info.get("graph_safe"):
        _graph_meta_recompute(plan_info)

    # Handle q_offset_override
    qo_segment_lens = plan_info["qo_segment_lens"]
    if plan_info.get("graph_safe") and q_offset_override is not None:
        raise ValueError(
            "graph_safe plans do not support run-time q_offset_override "
            "(it allocates a fresh seqused_k); pass qo_offset at plan time")
    if isinstance(q_offset_override, int):
        q_offset_override = torch.full(
            (batch,), q_offset_override, dtype=torch.int32, device=device
        )
    if q_offset_override is not None:
        seqused_k = (
            qo_segment_lens.to(torch.int32).to(device) + q_offset_override.to(device)
        ).to(torch.int32)
    else:
        seqused_k = plan_info["seqused_k"]

    softmax_scale = sm_scale if sm_scale is not None else head_dim ** -0.5

    q2k = plan_info.get("_q2k")
    if q2k is not None:
        if plan_info.get("graph_safe") and kv_block_indexes.shape[0] != total_q:
            raise ValueError(
                f"graph_safe requires kv_block_indexes padded to the static "
                f"max_total_q={total_q} rows (pad with -1); got "
                f"{kv_block_indexes.shape[0]}")
        h_dim = kv_block_indexes.shape[1]
        if h_dim == num_kv_heads:
            q2k.copy_(kv_block_indexes.permute(1, 0, 2))
        elif h_dim == num_qo_heads and num_qo_heads != num_kv_heads:
            q2k.copy_(kv_block_indexes[:, ::qhead_per_kv, :].permute(1, 0, 2))
        else:
            raise ValueError(
                f"kv_block_indexes head dim {h_dim} doesn't match "
                f"num_kv_heads={num_kv_heads} or num_qo_heads={num_qo_heads}"
            )
    else:
        q2k = _convert_kv_block_indexes_to_q2k(
            kv_block_indexes, num_kv_heads, num_qo_heads, qhead_per_kv
        )
    if plan_info.get("graph_safe"):
        q2k.masked_fill_(plan_info["_ws_pad_mask"].view(1, -1, 1), -1)

    is_paged = page_size > 0 and k.ndim == 4

    # ---- Lazy page_table construction (eager, graph-safe) ------------------
    if is_paged and kv_indices is not None:
        if plan_info.get("page_table") is None:
            kv_lens_cpu = plan_info.get("kv_segment_lens_cpu", None)
            if kv_lens_cpu is None:
                kv_lens_cpu = plan_info["kv_segment_lens"].cpu()
            plan_info["page_table"] = _build_page_table(
                kv_indices, kv_lens_cpu, page_size, batch
            )
        page_table = plan_info["page_table"]
    else:
        page_table = None

    if plan_info.get("_kv_page_indptr_gpu") is None:
        kv_page_indptr_cpu = plan_info.get("kv_page_indptr_cpu")
        if kv_page_indptr_cpu is None:
            kv_lens_cpu = plan_info.get("kv_segment_lens_cpu", plan_info["kv_segment_lens"].cpu())
            kv_page_indptr_cpu = _build_kv_page_indptr(kv_lens_cpu, page_size)
        plan_info["_kv_page_indptr_gpu"] = kv_page_indptr_cpu.to(device, dtype=torch.int32)
    kv_page_indptr = plan_info["_kv_page_indptr_gpu"]

    from .sparse_jit import get_build_k2q_csr_module
    k2q_module = get_build_k2q_csr_module()

    head_kv = num_kv_heads
    row_ptr = plan_info["_row_ptr"]
    q_indices = plan_info["_q_indices"]
    max_kv_blocks = (max_seqlen_k + blk_kv - 1) // blk_kv
    from .sparse_k2q_runtime import SparseAttentionSchedule, prepare_sparse_schedule
    target_q_per_cta = int(plan_info["target_q_per_cta"])
    work_capacity = int(plan_info["scheduler_metadata_capacity"])
    if usable_SM_count > 0:
        k2q_module.run_build_k2q_csr(
            q2k, cu_seqlens_q, cu_seqlens_k,
            row_ptr, q_indices,
            topk, blk_kv, total_rows, max_kv_blocks,
            torch.cuda.current_stream().cuda_stream,
        )
        sched = prepare_sparse_schedule(
            k2q_row_ptr=row_ptr,
            k2q_q_indices=q_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            total_q=total_q,
            max_seqlen_q=max_seqlen_q,
            topk=topk,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            blk_kv=blk_kv,
            device=device,
            enabled=True,
            usable_SM_count=usable_SM_count,
            capture_safe=True,
            out_scheduler_metadata=plan_info["_scheduler_metadata"],
            out_work_count=plan_info["_work_count"],
            out_qsplit_indices=plan_info["_qsplit_indices"],
            out_split_counts=plan_info["_split_counts"],
            target_q_per_cta_override=target_q_per_cta,
            work_capacity_override=work_capacity,
        )
    else:
        k2q_module.run_build_k2q_csr_with_schedule(
            q2k, cu_seqlens_q, cu_seqlens_k,
            row_ptr, q_indices,
            plan_info["_scheduler_metadata"],
            plan_info["_work_count"],
            plan_info["_qsplit_indices"],
            plan_info["_split_counts"],
            topk, blk_kv, total_rows, max_kv_blocks,
            target_q_per_cta, work_capacity, max_seqlen_q,
            torch.cuda.current_stream().cuda_stream,
        )
        sched = SparseAttentionSchedule(
            enabled=True,
            scheduler_metadata=plan_info["_scheduler_metadata"],
            work_count=plan_info["_work_count"],
            qsplit_indices=plan_info["_qsplit_indices"],
            split_counts=plan_info["_split_counts"],
            target_q_per_cta=target_q_per_cta,
        )

    # ---- Phase C: sparse fwd + combine ------------------------------------
    from .sparse_fwd_runtime import (
        sparse_fmha_plan as _fwd_plan,
        sparse_fmha_fwd as _fwd,
        sparse_fmha_combine as _combine,
    )

    if not is_paged or page_table is None:
        if plan_info.get("_kv_indices_gpu") is None:
            plan_info["_kv_indices_gpu"] = torch.zeros(1, dtype=torch.int32, device=device)
            plan_info["_is_paged_flag"] = False
        _kv_indices = plan_info["_kv_indices_gpu"]
        _kv_page_indptr = kv_page_indptr
        _is_paged = False
    else:
        if plan_info.get("graph_safe") and kv_indices.dtype != torch.int32:
            raise ValueError(
                "graph_safe paged runs require int32 kv_indices "
                f"(got {kv_indices.dtype}); a cached dtype-cast copy would "
                "read stale contents on replay")
        if plan_info.get("_kv_indices_gpu") is None:
            plan_info["_kv_indices_gpu"] = kv_indices.to(torch.int32)
            plan_info["_is_paged_flag"] = True
        _kv_indices = plan_info["_kv_indices_gpu"]
        _kv_page_indptr = kv_page_indptr
        _is_paged = True

    cacheable = q_offset_override is None
    fwd_plan = plan_info.get("_fwd_plan") if cacheable else None
    if fwd_plan is not None and plan_info.get("_fwd_plan_temp_scale") != kernel_lse_temperature_scale:
        fwd_plan = None
    if fwd_plan is None:
        fwd_plan = _fwd_plan(
            Q=q, K=k, V=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            kv_indices=_kv_indices,
            kv_page_indptr=_kv_page_indptr,
            schedule=sched,
            sm_scale=softmax_scale,
            head_dim=head_dim,
            num_heads_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            total_q=total_q,
            max_seqlen_q=max_seqlen_q,
            topk=topk,
            is_paged=_is_paged,
            page_size=page_size,
            work_capacity=sched.work_capacity,
            lse_temperature_scale=kernel_lse_temperature_scale,
            seqused_k=seqused_k,
        )
        if cacheable:
            plan_info["_fwd_plan"] = fwd_plan
            plan_info["_fwd_plan_temp_scale"] = kernel_lse_temperature_scale

    _fwd(
        Q=q, K=k, V=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        kv_indices=_kv_indices,
        kv_page_indptr=_kv_page_indptr,
        k2q_row_ptr=row_ptr,
        k2q_q_indices=q_indices,
        qsplit_indices=sched.qsplit_indices,
        scheduler_metadata=sched.scheduler_metadata,
        work_count=sched.work_count,
        split_counts=sched.split_counts,
        plan=fwd_plan,
    )

    result_o, result_lse, result_lse_temp = _combine(
        split_counts=sched.split_counts,
        plan=fwd_plan,
    )

    if out is not None:
        out.copy_(result_o)
        result_o = out
    if return_temperature_lse:
        return result_o, result_lse, result_lse_temp
    if return_softmax_lse:
        return result_o, result_lse
    return result_o, None
