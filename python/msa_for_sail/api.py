# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""FMHA varlen attention API: plan + run for PPU.

fmha_sail_plan and fmha_sail.

Doc is REPO/docs/msa_for_sail_api.md
"""

import math
from typing import Optional, Tuple, Union

__all__ = [
    "fmha_sail_plan", "fmha_sail", "sparse_topk_select",
    "_fmha_sail_plan", "_fmha_sail",
    "allocate_graph_workspace",
]

import numpy as np
import torch

from .jit import _dlpack_dtype_code, _PACK_FACTORS, get_fmha_variant, get_reduction_module, get_plan_fn, get_sparse_topk_module, get_prepare_metadata_fn
import os

_np_staging = np.empty(4096 * 1024, dtype=np.int32)
_np_staging_offset = 0

def _reset_np_staging():
    global _np_staging_offset
    _np_staging_offset = 0

def _plan_buf_from_list(data, device):
    global _np_staging, _np_staging_offset
    n = len(data)
    end = _np_staging_offset + n
    if end > _np_staging.shape[0]:
        _np_staging = np.empty(max(end, _np_staging.shape[0] * 2), dtype=np.int32)
        _np_staging_offset = 0
        end = n
    _np_staging[_np_staging_offset:end] = data
    buf = torch.empty(n, dtype=torch.int32, device=device)
    buf.copy_(torch.from_numpy(_np_staging[_np_staging_offset:end]), non_blocking=True)
    _np_staging_offset = end
    return buf

from enum import IntEnum, auto

class _BuffTag(IntEnum):
    fmha_sail_cutlass_workspace = auto()

    packed_work_range = auto()
    packed_work_info = auto()
    kv_tile_begin_indices = auto()
    kv_tile_end_indices = auto()
    kv_split_indices = auto()
    plan_cost = auto()
    num_kv_splits_per_row = auto()
    workspace_lse = auto()

    workspace_o = auto()

    sparse_topk_workspace = auto()

    Total = auto()


_workspace_cache = [[None] * _BuffTag.Total for _ in range(16)]
# _workspace_cache_per_plan = []

def _new_ws_cache():
    pass
    # global _workspace_cache
    # if len(_workspace_cache) == 0:
    #     print("new")
    #     _workspace_cache_per_plan.append([[None] * _BuffTag.Total for _ in range(16)])

def _alloc_workspace_buf(tag, size, device, dtype):
    global _workspace_cache
    device_id = torch.device(device).index
    buf = _workspace_cache[device_id][tag]
    if buf is not None and buf.dtype == dtype and buf.shape[0] >= size:
        return buf
    buf = torch.empty(size, dtype=dtype, device=device)
    _workspace_cache[device_id][tag] = buf
    return buf

def _get_workspace_buf(tag, device):
    global _workspace_cache
    device_id = torch.device(device).index
    return _workspace_cache[device_id][tag]

def _alloc_perplan_buf(tag, size, device, dtype):
    # global _workspace_cache
    # device_id = torch.device(device).index
    # buf = _workspace_cache[-1][device_id][tag]
    # if buf is not None and buf.shape[0] >= size:
    #     return buf
    buf = torch.empty(size, dtype=dtype, device=device)
    # _workspace_cache[-1][device_id][tag] = buf
    return buf

_NUM_CTA = None
def _get_num_cta(device):
    global _NUM_CTA
    if _NUM_CTA is None:
        props = torch.cuda.get_device_properties(device)
        _NUM_CTA = 20 if "810E" in props.name else props.multi_processor_count
    return _NUM_CTA

# ============================================================================
# Plan workspace: unified fixed-address buffer for graph-safe plan + run
# ============================================================================

from dataclasses import dataclass

@dataclass
class PlanWorkspace:
    """All plan/run buffers sliced from one contiguous uint8 allocation."""
    # metadata (int32)
    qo_segment_lens: torch.Tensor
    kv_segment_lens: torch.Tensor
    qo_segment_offsets: torch.Tensor
    kv_segment_offsets: torch.Tensor
    qo_offset: torch.Tensor
    kv_page_indptr: torch.Tensor
    plan_kv_lens: torch.Tensor
    # worklist (int64)
    packed_work_range: torch.Tensor
    packed_work_info: torch.Tensor
    # worklist (int32)
    kv_tile_begin_indices: torch.Tensor
    kv_tile_end_indices: torch.Tensor
    kv_split_indices: torch.Tensor
    num_kv_splits_per_row: torch.Tensor
    # reduction (float32)
    plan_cost: torch.Tensor
    workspace_lse: torch.Tensor
    # split-kv output (bfloat16)
    workspace_o: torch.Tensor
    # run kernel workspace (uint8)
    cute_workspace_buffer: torch.Tensor

_CUTE_WS_BYTES = 32 * 1024 * 1024  # 32MB fixed

def _align(offset, alignment):
    return (offset + alignment - 1) & ~(alignment - 1)

def compute_plan_workspace_bytes(n, num_ctas, max_work_items, total_qo_len,
                                  lse_total_size, workspace_o_elems):
    """Compute total bytes needed for the unified plan workspace."""
    off = 0
    # int32 metadata: 4*n + 3*(n+1) = 7n+3 elements
    off += (7 * n + 3) * 4
    # int32 worklist: max_work_items*3 + total_qo_len
    off += (max_work_items * 3 + total_qo_len) * 4
    # float32: plan_cost(2) + workspace_lse
    off += (2 + lse_total_size) * 4
    # align 8 for int64
    off = _align(off, 8)
    # int64: num_ctas + max_work_items
    off += (num_ctas + max_work_items) * 8
    # align 2 for bfloat16
    off = _align(off, 2)
    # bfloat16: workspace_o
    off += workspace_o_elems * 2
    # align 16 for cute_workspace_buffer (C++ std::align requirement)
    off = _align(off, 16)
    # uint8: cute_workspace_buffer
    off += _CUTE_WS_BYTES
    return off

def _split_lse_o_workspace_sizes(total_qo_for_ws, num_qo_heads, effective_splits):
    """Compute lse_total_size and workspace_o_elems for a given split count."""
    lse_total_size = effective_splits * total_qo_for_ws * num_qo_heads
    workspace_o_elems = total_qo_for_ws * effective_splits * num_qo_heads * 128
    return lse_total_size, workspace_o_elems

def _compute_per_split_bytes(ws_n, num_ctas, total_qo_for_ws, num_qo_heads):
    """Marginal workspace bytes per additional split, by differencing
    compute_plan_workspace_bytes at split=1 vs split=2."""
    per_split_items = max(131072, total_qo_for_ws * num_qo_heads)
    lse_1, wo_1 = _split_lse_o_workspace_sizes(total_qo_for_ws, num_qo_heads, 1)
    lse_2, wo_2 = _split_lse_o_workspace_sizes(total_qo_for_ws, num_qo_heads, 2)
    bytes_1 = compute_plan_workspace_bytes(
        ws_n, num_ctas, per_split_items, total_qo_for_ws, lse_1, wo_1)
    bytes_2 = compute_plan_workspace_bytes(
        ws_n, num_ctas, per_split_items * 2, total_qo_for_ws, lse_2, wo_2)
    return bytes_2 - bytes_1

def slice_plan_workspace(buf, n, num_ctas, max_work_items, total_qo_len,
                          lse_total_size, workspace_o_elems):
    """Slice a contiguous uint8 tensor into PlanWorkspace views."""
    off = 0
    # --- int32 metadata ---
    int32_meta_count = 7 * n + 3
    meta = buf[off:off + int32_meta_count * 4].view(torch.int32)
    off += int32_meta_count * 4
    mo = 0
    qo_segment_lens    = meta[mo:mo+n]; mo += n
    kv_segment_lens    = meta[mo:mo+n]; mo += n
    qo_segment_offsets = meta[mo:mo+n+1]; mo += n+1
    kv_segment_offsets = meta[mo:mo+n+1]; mo += n+1
    qo_offset_buf      = meta[mo:mo+n]; mo += n
    kv_page_indptr     = meta[mo:mo+n+1]; mo += n+1
    plan_kv_lens       = meta[mo:mo+n]; mo += n

    # --- int32 worklist ---
    int32_wl_count = max_work_items * 3 + total_qo_len
    wl32 = buf[off:off + int32_wl_count * 4].view(torch.int32)
    off += int32_wl_count * 4
    wo = 0
    kv_tile_begin_indices = wl32[wo:wo+max_work_items]; wo += max_work_items
    kv_tile_end_indices   = wl32[wo:wo+max_work_items]; wo += max_work_items
    kv_split_indices      = wl32[wo:wo+max_work_items]; wo += max_work_items
    num_kv_splits_per_row = wl32[wo:wo+total_qo_len]; wo += total_qo_len

    # --- float32 ---
    f32_count = 2 + lse_total_size
    f32 = buf[off:off + f32_count * 4].view(torch.float32)
    off += f32_count * 4
    plan_cost   = f32[0:2]
    workspace_lse = f32[2:2+lse_total_size]

    # --- align 8 for int64 ---
    off = _align(off, 8)
    int64_count = num_ctas + max_work_items
    i64 = buf[off:off + int64_count * 8].view(torch.int64)
    off += int64_count * 8
    packed_work_range = i64[0:num_ctas]
    packed_work_info  = i64[num_ctas:num_ctas+max_work_items]

    # --- align 2 for bfloat16 ---
    off = _align(off, 2)
    workspace_o = buf[off:off + workspace_o_elems * 2].view(torch.bfloat16)
    off += workspace_o_elems * 2

    # --- uint8 cute workspace ---
    # Align to 16 bytes: C++ AlignedAllocator uses std::align(16, ...) which
    # fails if ptr is not 16-aligned even with size=0.
    off = _align(off, 16)
    cute_workspace_buffer = buf[off:off + _CUTE_WS_BYTES]
    off += _CUTE_WS_BYTES

    return PlanWorkspace(
        qo_segment_lens=qo_segment_lens,
        kv_segment_lens=kv_segment_lens,
        qo_segment_offsets=qo_segment_offsets,
        kv_segment_offsets=kv_segment_offsets,
        qo_offset=qo_offset_buf,
        kv_page_indptr=kv_page_indptr,
        plan_kv_lens=plan_kv_lens,
        packed_work_range=packed_work_range,
        packed_work_info=packed_work_info,
        kv_tile_begin_indices=kv_tile_begin_indices,
        kv_tile_end_indices=kv_tile_end_indices,
        kv_split_indices=kv_split_indices,
        num_kv_splits_per_row=num_kv_splits_per_row,
        plan_cost=plan_cost,
        workspace_lse=workspace_lse,
        workspace_o=workspace_o,
        cute_workspace_buffer=cute_workspace_buffer,
    )

_plan_workspace_cache = {}  # device_id → uint8 tensor

def _get_plan_workspace_bytes(total_bytes, device, given_workspace=None):
    """Get unified workspace. Use given_workspace if provided, else global cache."""
    if given_workspace is not None:
        if given_workspace.shape[0] < total_bytes:
            raise ValueError(
                f"given_workspace too small: {given_workspace.shape[0]} bytes < required {total_bytes} bytes. "
                f"Re-allocate with allocate_graph_workspace(...).")
        return given_workspace
    device_id = torch.device(device).index
    buf = _plan_workspace_cache.get(device_id)
    if buf is None or buf.shape[0] < total_bytes:
        buf = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        _plan_workspace_cache[device_id] = buf
    return buf

def _fill_plan_buf_from_list(buf, data, device):
    """Copy a Python list into a pre-allocated device buffer (fixed address)."""
    if data is None:
        return
    n = len(data)
    buf[:n].copy_(torch.tensor(data, dtype=torch.int32), non_blocking=True)

def _get_indexer_qotile_and_occupancy(max_qo_len, num_qo_heads, num_kv_heads, num_sm):   # return qo_tile_size, kv_tile_size, occupancy
    h_r = num_qo_heads // num_kv_heads if num_kv_heads != -1 else num_qo_heads
    qo_len = max_qo_len * h_r if (max_qo_len > 0 and max_qo_len <= 32) else max_qo_len
    if qo_len <= 16:
        return 16, 128, min(256 // num_sm, 7)   # 256 is MAX_SMS in plan.cuh
    elif qo_len <= 64:
        return 64, 128, 4
    else:
        return 128, 128, 4

def _get_fmha_qotile_and_occupancy(max_qo_len, num_qo_heads, num_kv_heads, num_sm, is_fp8=False):   # return qo_tile_size, kv_tile_size, occupancy
    if max_qo_len <= 16:
        return 16, 128 if is_fp8 else 64, 8
    elif max_qo_len <= 64:
        return 64, 128, 2
    else:
        return 128, 128, 1

def _compute_pack_factor(max_qo_len, num_qo_heads, num_kv_heads):
    if num_kv_heads == -1:
        num_kv_heads = 1   # enable packing for single head
    h_r = num_qo_heads // num_kv_heads
    if h_r <= 1 or max_qo_len <= 0 or max_qo_len > 32:
        return 1
    max_pf = 128 // max_qo_len
    for pf in reversed(_PACK_FACTORS):
        if pf <= max_pf and pf <= h_r and h_r % pf == 0:
            return pf
    return 1


# Adaptive split-KV sizing constants (must stay in sync with plan.cuh capacity).
_MAX_SPLIT_WORKSPACE_BYTES = 256 * 1024 * 1024  # 256 MB budget for split-KV workspace
_MAX_UNSPLIT = 4096  # must match MAX_UNSPLIT in plan.cuh
# Hard bound: split sub_id is stored as int8_t in plan.cuh (task_sub_id), so max_splits must stay <= 128.
_MAX_ADAPTIVE_SPLITS = 128
_MAX_FIXED_SPARSE_SPLITS = 16

def _plan_splits_and_sizes(is_sparse, qo_tile_size, num_ctas, n, ws_n,
                           total_qo_for_ws, pack_factor, num_qo_heads,
                           num_kv_splits, sparse_kv_tiles=0, graph_safe=False):
    """Shared split-KV decision + workspace sizing for _fmha_sail_plan and
    allocate_graph_workspace. Single source of truth so the pre-allocated
    fixed-address workspace can never underestimate plan's requirement.

    Returns (plan_kv_splits, effective_num_kv_splits, _explicit_nosplit,
    max_work_items, lse_total_size, workspace_o_elems, total_bytes).
    """
    if num_kv_splits >= 1:
        effective_splits = num_kv_splits
        plan_kv_splits = num_kv_splits
    else:
        # Adaptive mode: plan_kernel decides split/nosplit internally.
        # Cap max_splits to plan_kernel capacity (MAX_SMS=256 * MAX_TASKS_PER_SM=96).
        # Sparse per-token expansion means each token is an independent batch
        # with its own qo_tile(s), unlike dense where one batch may span
        # many qo_tiles.
        if is_sparse:
            qo_per_batch = pack_factor if pack_factor > 1 else 1
            tiles_per_batch = (qo_per_batch + qo_tile_size - 1) // qo_tile_size
            total_tiles = n * tiles_per_batch * num_qo_heads
        else:
            total_tiles = (total_qo_for_ws + qo_tile_size - 1) // qo_tile_size * num_qo_heads
        max_capacity = num_ctas * 96  # num_ctas (actual SMs) * MAX_TASKS_PER_SM
        if total_tiles * 2 >= max_capacity or total_tiles >= _MAX_UNSPLIT:
            if graph_safe and is_sparse and sparse_kv_tiles > 1:
                per_split_bytes = _compute_per_split_bytes(
                    ws_n, num_ctas, total_qo_for_ws, num_qo_heads)
                budget_splits = min(_MAX_ADAPTIVE_SPLITS,
                                    max(1, _MAX_SPLIT_WORKSPACE_BYTES // per_split_bytes))
                effective_splits = max(1, min(_MAX_FIXED_SPARSE_SPLITS,
                                              sparse_kv_tiles, budget_splits))
                plan_kv_splits = -effective_splits
            else:
                effective_splits = 1
                plan_kv_splits = 1
        else:
            per_split_bytes = _compute_per_split_bytes(
                ws_n, num_ctas, total_qo_for_ws, num_qo_heads)
            adaptive_splits = min(_MAX_ADAPTIVE_SPLITS, max(1, _MAX_SPLIT_WORKSPACE_BYTES // per_split_bytes))
            effective_splits = adaptive_splits
            plan_kv_splits = -adaptive_splits

    # TODO: sized by the total_qo * num_heads upper bound; can shrink once a
    # prefill-specific scheduling algorithm is supported.
    max_work_items = max(131072, total_qo_for_ws * num_qo_heads) * effective_splits
    _explicit_nosplit = (effective_splits == 1 and (is_sparse or num_kv_splits == 1))
    if _explicit_nosplit:
        lse_total_size = 0
        workspace_o_elems = 0
    else:
        lse_total_size, workspace_o_elems = _split_lse_o_workspace_sizes(total_qo_for_ws, num_qo_heads, effective_splits)

    total_bytes = compute_plan_workspace_bytes(ws_n, num_ctas, max_work_items,
                                               total_qo_for_ws, lse_total_size,
                                               workspace_o_elems)
    return (plan_kv_splits, effective_splits, _explicit_nosplit, max_work_items,
            lse_total_size, workspace_o_elems, total_bytes)


def allocate_graph_workspace(batch_size, num_qo_heads, num_kv_heads=-1,
                             max_qo_len=1, page_size=128, kv_block_num=-1,
                             num_kv_splits=1, device=None,
                             output_maxscore=False, use_fp8_kvcache=False):
    """Pre-allocate a fixed-address workspace for CUDA Graph capture.

    Must be called ONCE per (batch_size, num_qo_heads, max_qo_len) config.
    The returned tensor is passed as ``given_workspace`` to ``fmha_sail_plan``
    with ``graph_safe=True``.

    Parameters
    ----------
    batch_size : int
        Number of requests (fixed across replays).
    num_qo_heads : int
        Number of Q/O heads.
    num_kv_heads : int, optional
        Number of KV heads (for GQA). Default -1 (MHA).
    max_qo_len : int, optional
        Maximum Q/O length per request. Default 1 (decode).
    page_size : int, optional
        KV page size. Default 128.
    kv_block_num : int, optional
        Number of sparse KV blocks per token. >0 enables sparse mode.
    num_kv_splits : int, optional
        Fixed split count. Default 1 (nosplit).
    device : optional
        CUDA device. Default: current device.
    output_maxscore : bool, optional
        Must match the plan's ``output_maxscore`` flag: it switches tile
        selection to the indexer table. Default False.

    Returns
    -------
    torch.Tensor
        uint8 tensor of computed workspace size, fixed address.
    """
    device = torch.cuda.current_device() if device is None else device

    is_sparse = (kv_block_num > 0 and page_size > 0)
    _cfg_qo_len = 1 if (kv_block_num > 0 and not output_maxscore) else max_qo_len
    qo_tile_size, kv_tile_size, num_occupancy = _get_indexer_qotile_and_occupancy(max_qo_len, num_qo_heads, num_kv_heads, _get_num_cta(device)) if (output_maxscore or kv_block_num <= 0) else \
                                _get_fmha_qotile_and_occupancy(_cfg_qo_len, num_qo_heads, num_kv_heads, _get_num_cta(device), is_fp8=use_fp8_kvcache)

    # plan.cuh MAX_SMS=256 caps the number of scheduled CTA slots.
    num_ctas = min(_get_num_cta(device) * num_occupancy, 256)

    pack_factor = _compute_pack_factor(_cfg_qo_len, num_qo_heads, num_kv_heads)
    packed_heads = num_qo_heads // pack_factor if pack_factor > 1 else num_qo_heads

    # Worst-case totals (max_qo_len for every request), matching plan's
    # max_total_q-based sizing in graph_safe mode.
    max_total_q = batch_size * max_qo_len
    if is_sparse:
        n = max_total_q
        total_qo_for_ws = max_total_q * (pack_factor if pack_factor > 1 else 1)
    else:
        n = batch_size
        total_qo_for_ws = max_total_q * pack_factor

    # Sizing must match _fmha_sail_plan exactly; both use the shared helper.
    (_, _, _, _, _, _, total_bytes) = _plan_splits_and_sizes(
        is_sparse, qo_tile_size, num_ctas, n, max_total_q, total_qo_for_ws,
        pack_factor, packed_heads, num_kv_splits,
        sparse_kv_tiles=(math.ceil(kv_block_num * page_size / kv_tile_size)
                         if is_sparse else 0),
        graph_safe=True)
    return torch.empty(total_bytes, dtype=torch.uint8, device=device)

def _to_device(t, d):
    return t.to(d, non_blocking=True) if t is not None else None

def _prefill_qlen_threshold(sparse):
    return 32 if sparse else 128

def _detect_paged_kv_layout(shape, page_size_hint):
    if len(shape) != 4:
        raise ValueError(
            f"paged K/V must be 4D ([pages,Hkv,ps,D] HND or [pages,ps,Hkv,D] NHD), "
            f"got shape {tuple(shape)}")
    if page_size_hint is not None and page_size_hint > 0:
        if shape[2] == page_size_hint:
            return False, int(shape[1]), int(shape[2])      # HND
        if shape[1] == page_size_hint:
            return True, int(shape[2]), int(shape[1])       # NHD
        raise ValueError(
            f"cannot infer paged KV layout: neither dim1 nor dim2 of shape={tuple(shape)} "
            f"equals page_size={page_size_hint}; expected [pages,Hkv,ps,D] (HND) "
            f"or [pages,ps,Hkv,D] (NHD)")
    return False, int(shape[1]), int(shape[2])

def _validate_fmha_inputs(
    q, k, v, qo_segment_lens, kv_segment_lens,
    num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
    batch_size, qo_total_len, is_paged,
    kv_indices, kv_block_indexes, qo_offset, page_size,
    pack_factor=1,
    packed_work_range=None, packed_work_info=None,
    num_kv_splits=1,
    kv_tile_begin_indices=None, kv_tile_end_indices=None, kv_split_indices=None,
    qo_segment_offsets=None, kv_page_indptr=None,
    qo_tile_size=128,
    kv_tile_size=None,
    kv_is_nhd=False,
):
    assert q.dim() == 3, f"q must be [total_qo_len, num_qo_heads, head_dim], got {q.shape}"
    assert num_qo_heads % num_kv_heads == 0, (
        f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    assert qo_segment_lens.dim() == 1, f"qo_segment_lens must be 1D, got {qo_segment_lens.shape}"
    assert kv_segment_lens.dim() == 1, f"kv_segment_lens must be 1D, got {kv_segment_lens.shape}"
    assert qo_segment_lens.shape[0] == batch_size
    assert kv_segment_lens.shape[0] == batch_size

    qo_lens_cpu = qo_segment_lens.cpu()
    kv_lens_cpu = kv_segment_lens.cpu()

    expected_qo_total = qo_total_len * pack_factor if pack_factor > 1 else qo_total_len
    assert qo_lens_cpu.sum().item() == expected_qo_total, (
        f"sum(qo_segment_lens)={qo_lens_cpu.sum().item()} != expected={expected_qo_total} "
        f"(q.shape[0]={qo_total_len}, pack_factor={pack_factor})"
    )

    if is_paged:
        assert k.dim() == 4, (
            f"paged K must be 4D ([pages,Hkv,ps,D] HND or [pages,ps,Hkv,D] NHD), got {k.shape}")
        assert v.dim() == 4, (
            f"paged V must be 4D ([pages,Hkv,ps,D] HND or [pages,ps,Hkv,D] NHD), got {v.shape}")
        total_pages = k.shape[0]
        # Layout-aware dim check: (Hkv, page_size) sits at dims (1,2) for HND and
        # at dims (2,1) for NHD, so compare as an unordered pair.
        assert sorted([k.shape[1], k.shape[2]]) == sorted([num_kv_heads, page_size]), (
            f"paged K dims 1/2 must be {{Hkv={num_kv_heads}, page_size={page_size}}}, "
            f"got {k.shape} (layout={'NHD' if kv_is_nhd else 'HND'})")
        # V must mirror K's layout exactly: the binding derives one kv_is_nhd flag
        # from K and applies it to both K and V strides.
        assert v.shape[1] == k.shape[1] and v.shape[2] == k.shape[2], (
            f"paged V must use the same layout as K, got K={tuple(k.shape)} "
            f"V={tuple(v.shape)} (both must be HND or both NHD)")
        assert v.shape[0] == total_pages, (
            f"paged V must have the same page count as K, got {v.shape[0]} vs {total_pages}")
        assert k.shape[3] == v.shape[3], (
            f"paged K/V head_dim must match, got {k.shape[3]} vs {v.shape[3]}")
        # Stride guard. The kernel always reads stride(0) (page) plus, depending on
        # layout, stride(1)/stride(2). Under HND the in-page token stride is NOT read
        # -- it is assumed to equal head_dim -- so HND must additionally satisfy
        # stride(2) == head_dim. A transpose *view* passed without .contiguous()
        # violates exactly this and would otherwise be silently mis-addressed
        # (see docs/msa_sparse_nhd_kv_layout_spec.md section 2.6).
        for _name, _t in (("K", k), ("V", v)):
            assert _t.stride(3) == 1, (
                f"paged {_name} head_dim must be contiguous (stride(-1)==1), "
                f"got shape={tuple(_t.shape)} stride={tuple(_t.stride())}")
            if not kv_is_nhd:
                assert _t.stride(2) == _t.shape[3], (
                    f"paged {_name} HND requires contiguous in-page tokens "
                    f"(stride(2)==head_dim={_t.shape[3]}), got shape={tuple(_t.shape)} "
                    f"stride={tuple(_t.stride())}; if this is a transpose view of the "
                    f"other layout, add .contiguous() or pass it as NHD")
    else:
        total_kv_len = k.shape[0]
        assert kv_lens_cpu.sum().item() == total_kv_len, (
            f"sum(kv_segment_lens)={kv_lens_cpu.sum().item()} != k.shape[0]={total_kv_len}"
        )

    if qo_offset is not None:
        assert qo_offset.dim() == 1 and qo_offset.shape[0] == batch_size, (
            f"qo_offset must be [batch_size={batch_size}], got {qo_offset.shape}"
        )
        # off_cpu = qo_offset.cpu()
        # assert (off_cpu >= 0).all(), (
        #     f"qo_offset must be non-negative, got min={off_cpu.min().item()}"
        # )
        # unpacked_qo_lens = qo_lens_cpu // pack_factor if pack_factor > 1 else qo_lens_cpu
        # max_offsets = kv_lens_cpu - unpacked_qo_lens
        # violations = off_cpu > max_offsets
        # if violations.any():
        #     b = violations.nonzero()[0].item()
        #     assert False, (
        #         f"qo_offset[{b}]={off_cpu[b].item()} > kv_len-qo_len="
        #         f"{kv_lens_cpu[b].item()}-{qo_lens_cpu[b].item()}={max_offsets[b].item()}"
        #     )

    if kv_indices is not None and kv_block_indexes is None:
        assert is_paged, "kv_indices provided but K/V are not paged (4D)"
        kvi_cpu = kv_indices.cpu()
        total_pages_in_table = kvi_cpu.shape[0]
        total_pages_needed = sum(
            (kv_lens_cpu[b].item() + page_size - 1) // page_size for b in range(batch_size)
        )
        assert total_pages_in_table == total_pages_needed, (
            f"kv_indices length {total_pages_in_table} != "
            f"sum(ceil(kv_segment_lens/page_size))={total_pages_needed}"
        )
        total_pages = k.shape[0]
        assert (kvi_cpu >= 0).all() and (kvi_cpu < total_pages).all(), (
            f"kv_indices has values outside [0, {total_pages}): "
            f"min={kvi_cpu.min().item()}, max={kvi_cpu.max().item()}"
        )

    if kv_block_indexes is not None:
        assert is_paged, "sparse mode requires paged KV (kv_indices)"
        assert kv_indices is not None, "sparse mode requires kv_indices"
        assert kv_block_indexes.dim() == 3, (
            f"kv_block_indexes must be [total_qo_len, H_kv, KVBlockNum], got {kv_block_indexes.shape}"
        )
        assert kv_block_indexes.shape[0] == qo_total_len, (
            f"kv_block_indexes must be [total_qo_len={qo_total_len}, H_kv, KVBlockNum], got {kv_block_indexes.shape}"
        )
        assert kv_block_indexes.shape[1] == num_kv_heads

        bi_cpu = kv_block_indexes.cpu()
        valid_mask = (bi_cpu != -1)
        assert ((bi_cpu >= 0) | (bi_cpu == -1)).all(), (
            "kv_block_indexes must contain only non-negative indices or -1"
        )
        assert (valid_mask[:, :, :-1] >= valid_mask[:, :, 1:]).all(), (
            "kv_block_indexes: -1 padding must be at the end"
        )
        unpacked_qo_lens = qo_lens_cpu // pack_factor if pack_factor > 1 else qo_lens_cpu
        batch_of_qtoken = torch.repeat_interleave(
            torch.arange(batch_size), unpacked_qo_lens)
        kv_lens_per_qtoken = kv_lens_cpu[batch_of_qtoken]
        num_pages = ((kv_lens_per_qtoken + page_size - 1) // page_size).view(-1, 1, 1)
        assert (bi_cpu[valid_mask] < num_pages.expand_as(bi_cpu)[valid_mask]).all(), (
            "kv_block_indexes: index out of range for batch page count"
        )
        both_valid = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]
        if both_valid.any():
            assert (bi_cpu[:, :, 1:][both_valid] > bi_cpu[:, :, :-1][both_valid]).all(), (
                "kv_block_indexes: must be strictly ascending"
            )

    if qo_segment_offsets is not None:
        off_cpu = qo_segment_offsets.cpu()
        assert off_cpu.shape[0] == batch_size + 1, (
            f"qo_segment_offsets must have {batch_size + 1} elements, got {off_cpu.shape[0]}"
        )
        assert off_cpu[0].item() == 0, f"qo_segment_offsets[0]={off_cpu[0].item()} != 0"
        assert off_cpu[-1].item() == expected_qo_total, (
            f"qo_segment_offsets[-1]={off_cpu[-1].item()} != expected={expected_qo_total}"
        )
        diffs = off_cpu[1:] - off_cpu[:-1]
        assert (diffs >= 0).all(), "qo_segment_offsets must be non-decreasing"
        assert (diffs == qo_lens_cpu).all(), "qo_segment_offsets must be cumsum of qo_segment_lens"

    if kv_page_indptr is not None:
        ip_cpu = kv_page_indptr.cpu()
        assert ip_cpu.shape[0] == batch_size + 1, (
            f"kv_page_indptr must have {batch_size + 1} elements, got {ip_cpu.shape[0]}"
        )
        assert ip_cpu[0].item() == 0, f"kv_page_indptr[0]={ip_cpu[0].item()} != 0"
        assert (ip_cpu[1:] >= ip_cpu[:-1]).all(), "kv_page_indptr must be non-decreasing"
        if kv_indices is not None:
            assert ip_cpu[-1].item() <= kv_indices.shape[0], (
                f"kv_page_indptr[-1]={ip_cpu[-1].item()} > kv_indices.size={kv_indices.shape[0]}"
            )

    if packed_work_range is not None and packed_work_info is not None:
        import math
        pwr_cpu = packed_work_range.cpu()
        pwi_cpu = packed_work_info.cpu()
        num_ctas = pwr_cpu.shape[0]
        max_work_idx = pwi_cpu.shape[0]

        packed_num_heads = num_qo_heads // pack_factor if pack_factor > 1 else num_qo_heads
        max_tiles_per_batch = [(int(l) + qo_tile_size - 1) // qo_tile_size for l in qo_lens_cpu]

        for cta in range(num_ctas):
            r = int(pwr_cpu[cta].item())
            start = r & 0xFFFFFFFF
            end = (r >> 32) & 0xFFFFFFFF
            assert start <= end, (
                f"packed_work_range[{cta}]: start={start} > end={end}"
            )
            assert end <= max_work_idx, (
                f"packed_work_range[{cta}]: end={end} > packed_work_info.size={max_work_idx}"
            )
            for wi in range(start, end):
                packed = int(pwi_cpu[wi].item())
                bi = packed & 0xFFFF
                hi = (packed >> 16) & 0xFFFF
                qo_tile = (packed >> 32) & 0xFFFFFFFF
                assert bi < batch_size, (
                    f"packed_work_info[{wi}]: batch_idx={bi} >= batch_size={batch_size}"
                )
                assert hi < packed_num_heads, (
                    f"packed_work_info[{wi}]: head_idx={hi} >= packed_num_heads={packed_num_heads}"
                )
                assert qo_tile < max_tiles_per_batch[bi], (
                    f"packed_work_info[{wi}]: qo_tile={qo_tile} >= max_tiles={max_tiles_per_batch[bi]} "
                    f"for batch {bi} (qo_len={int(qo_lens_cpu[bi])})"
                )

        # Verify completeness: every (batch, head, qo_tile) must appear exactly once
        # (for num_kv_splits=1) or exactly num_splits times (for split-KV, each with
        # a different split index covering the full KV range).
        expected = set()
        for bi in range(batch_size):
            for hi in range(packed_num_heads):
                for qt in range(max_tiles_per_batch[bi]):
                    expected.add((bi, hi, qt))

        seen = {}
        for cta in range(num_ctas):
            r = int(pwr_cpu[cta].item())
            start = r & 0xFFFFFFFF
            end = (r >> 32) & 0xFFFFFFFF
            for wi in range(start, end):
                packed = int(pwi_cpu[wi].item())
                bi = packed & 0xFFFF
                hi = (packed >> 16) & 0xFFFF
                qt = (packed >> 32) & 0xFFFFFFFF
                key = (bi, hi, qt)
                seen[key] = seen.get(key, 0) + 1

        missing = expected - set(seen.keys())
        assert not missing, (
            f"plan missing {len(missing)} work items, e.g. {list(missing)[:5]}"
        )

        if num_kv_splits <= 1:
            duplicates = {k: v for k, v in seen.items() if v > 1}
            assert not duplicates, (
                f"plan has {len(duplicates)} duplicated work items (nosplit), e.g. {list(duplicates.items())[:5]}"
            )

        if num_kv_splits > 1 and kv_tile_begin_indices is not None:
            tb_cpu = kv_tile_begin_indices.cpu()
            te_cpu = kv_tile_end_indices.cpu()
            sp_cpu = kv_split_indices.cpu()

            from collections import defaultdict
            tile_splits = defaultdict(list)

            for cta in range(num_ctas):
                r = int(pwr_cpu[cta].item())
                start = r & 0xFFFFFFFF
                end = (r >> 32) & 0xFFFFFFFF
                for wi in range(start, end):
                    tb, te = int(tb_cpu[wi].item()), int(te_cpu[wi].item())
                    assert tb <= te, (
                        f"kv_tile_begin[{wi}]={tb} > kv_tile_end[{wi}]={te}"
                    )
                    sp = int(sp_cpu[wi].item())
                    assert 0 <= sp < num_kv_splits, (
                        f"kv_split_indices[{wi}]={sp} out of range [0, {num_kv_splits})"
                    )
                    packed = int(pwi_cpu[wi].item())
                    key = (packed & 0xFFFF, (packed >> 16) & 0xFFFF,
                           (packed >> 32) & 0xFFFFFFFF)
                    tile_splits[key].append((tb, te, sp))

            is_sparse = kv_block_indexes is not None
            if kv_tile_size is None:
                kv_tile_size = 64 if (qo_tile_size <= 16 and is_sparse) else 128
            for key, splits in tile_splits.items():
                bi, hi, qt = key
                if is_sparse:
                    kv_block_num = kv_block_indexes.shape[2]
                    kl = kv_block_num * page_size
                    off_q = kl - int(qo_lens_cpu[bi])
                else:
                    kl = int(kv_lens_cpu[bi])
                    off_q = int(qo_offset[bi].cpu()) if qo_offset is not None else (kl - int(qo_lens_cpu[bi]))
                packed_q_end = (qt + 1) * qo_tile_size
                q_end = (packed_q_end - 1) // pack_factor + 1 if pack_factor > 1 else packed_q_end
                eff_kv = min(q_end + off_q, kl)
                expected_iters = max(0, (eff_kv + kv_tile_size - 1) // kv_tile_size)
                splits_sorted = sorted(splits, key=lambda x: x[0])
                if expected_iters > 0:
                    assert splits_sorted[0][0] == 0, (
                        f"tile {key}: first split begins at {splits_sorted[0][0]}, expected 0"
                    )
                    assert splits_sorted[-1][1] == expected_iters, (
                        f"tile {key}: last split ends at {splits_sorted[-1][1]}, expected {expected_iters}"
                    )
                for i in range(1, len(splits_sorted)):
                    prev_end = splits_sorted[i - 1][1]
                    curr_begin = splits_sorted[i][0]
                    assert prev_end == curr_begin, (
                        f"tile {key}: gap/overlap at split boundary: "
                        f"prev_end={prev_end}, curr_begin={curr_begin}"
                    )
                sub_ids = sorted(s[2] for s in splits_sorted)
                expected_ids = list(range(len(splits_sorted)))
                assert sub_ids == expected_ids, (
                    f"tile {key}: sub_ids {sub_ids} not contiguous 0..{len(splits_sorted)-1}"
                )


def _expand_for_per_token_sparse(qo_lens, kv_lens, qo_offset, page_size, pack_factor=1):
    B = len(qo_lens)
    total_q = sum(qo_lens)

    if qo_offset is None:
        qo_offset = [kv_lens[i] - qo_lens[i] for i in range(B)]

    kv_page_indptr = [0]
    acc = 0
    for k in kv_lens:
        acc += (k + page_size - 1) // page_size
        kv_page_indptr.append(acc)

    expanded_qo_lens = [pack_factor] * total_q
    expanded_kv_lens = [0] * total_q
    expanded_qo_offset = [0] * total_q
    expanded_kv_page_indptr = [0] * (total_q + 1)

    idx = 0
    for b in range(B):
        kv_len_b = kv_lens[b]
        qo_offset_b = qo_offset[b]
        kv_page_indptr_b = kv_page_indptr[b]
        for j in range(qo_lens[b]):
            expanded_kv_lens[idx] = kv_len_b
            expanded_qo_offset[idx] = qo_offset_b + j
            expanded_kv_page_indptr[idx] = kv_page_indptr_b
            idx += 1

    expanded_kv_page_indptr[total_q] = kv_page_indptr[B]

    return expanded_qo_lens, expanded_kv_lens, expanded_qo_offset, expanded_kv_page_indptr

class PlanInfo(dict):
    """Execution plan returned by the internal dense FMHA planner.

    Users normally receive the public tuple returned by ``fmha_sail_plan`` and
    pass it unchanged to ``fmha_sail``.  The dictionary stores CUDA worklists,
    sequence metadata, split-KV workspaces, and cached buffers owned by the
    plan.
    """

    def __del__(self):
        pass
        # print("del")
        # _workspace_cache_per_plan.append(self["_ws_cache"])

def _make_plan_info(
    qo_tile_size, kv_tile_size,
    packed_work_range, packed_work_info,
    kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
    num_kv_splits, workspace_o, workspace_lse,
    max_qo_len, predicted_speedup, num_kv_splits_per_row,
    qo_segment_offsets, kv_segment_offsets, kv_page_indptr, max_k_tiles,
    qo_segment_lens, kv_segment_lens, qo_offset,
    pack_factor, orig_num_qo_heads,
    qo_len_uniform, cute_workspace_buffer,
    graph_safe=False, page_size=-1
):
    # ws = _workspace_cache_per_plan.pop()
    # print("pop")
    return PlanInfo({
        # "_ws_cache" : ws,
        "qo_tile_size": qo_tile_size,
        "kv_tile_size": kv_tile_size,
        "packed_work_range": packed_work_range,
        "packed_work_info": packed_work_info,
        "kv_tile_begin_indices": kv_tile_begin_indices,
        "kv_tile_end_indices": kv_tile_end_indices,
        "kv_split_indices": kv_split_indices,
        "num_kv_splits": num_kv_splits,
        "workspace_o": workspace_o,
        "workspace_lse": workspace_lse,
        "max_qo_len": max_qo_len,
        "predicted_speedup": predicted_speedup,
        "num_kv_splits_per_row": num_kv_splits_per_row,
        "qo_segment_offsets": qo_segment_offsets,
        "kv_segment_offsets": kv_segment_offsets,
        "kv_page_indptr": kv_page_indptr,
        "max_k_tiles": max_k_tiles,
        "qo_segment_lens": qo_segment_lens,
        "kv_segment_lens": kv_segment_lens,
        "qo_offset": qo_offset,
        "pack_factor": pack_factor,
        "orig_num_qo_heads": orig_num_qo_heads,
        "qo_len_uniform": qo_len_uniform,
        "cute_workspace_buffer": cute_workspace_buffer,
        "graph_safe": graph_safe,
        "page_size": page_size,
    })


def _call_plan(qo_segment_offsets, qo_segment_lens, kv_segment_lens,
               packed_work_range, packed_work_info,
               qo_tile_size, kv_tile_size, num_qo_heads, num_ctas, causal,
               qo_offset, num_kv_splits,
               kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
               chunk_size, out_max_sm_cost, num_kv_splits_per_row, workspace_lse, lse_total_size,
               pack_factor, is_sparse, cuda_stream=None,
               ):

    plan_module = get_plan_fn()

    if cuda_stream is None:
        cuda_stream = torch.cuda.current_stream().cuda_stream
    plan_module.plan(
        qo_segment_offsets, qo_segment_lens, kv_segment_lens,
        packed_work_range, packed_work_info,
        qo_tile_size, kv_tile_size, num_qo_heads, num_ctas, causal,
        qo_offset, num_kv_splits,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        chunk_size, out_max_sm_cost, num_kv_splits_per_row,
        cuda_stream,
        workspace_lse, lse_total_size,
        pack_factor,
        is_sparse,
    )

def _fmha_sail_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int = -1,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    num_kv_splits: int = -1,
    page_size: int = -1,
    output_maxscore: bool = False,
    kv_block_num: int = -1,
    usable_SM_count: int = -1,
    causal: bool = True,
    sparse_kernel_mode: str = 'auto',
    use_fp8_kvcache: bool = False,
    device = None,
    stream = None,
    given_workspace = None,
    max_total_q: int = -1,
    graph_safe: bool = False,
    max_qo_len_override: int = -1,
    qo_len_uniform_override: Optional[bool] = None,
    max_kv_len_override: int = -1,
):
    device = torch.cuda.current_device() if device is None else device
    # _reset_np_staging()

    # --- Bucket constants ---
    if graph_safe:
        if max_qo_len_override <= 0:
            raise ValueError("graph_safe=True requires max_qo_len_override > 0")
        if qo_len_uniform_override is None:
            raise ValueError("graph_safe=True requires qo_len_uniform_override")
        if given_workspace is None:
            raise ValueError("graph_safe=True requires given_workspace (pre-allocated)")
        if max_total_q <= 0:
            raise ValueError("graph_safe=True requires max_total_q > 0")
        if (output_maxscore or not causal) and max_kv_len_override <= 0:
            raise ValueError(
                "graph_safe=True requires max_kv_len_override > 0 "
                "when output_maxscore=True or causal=False")
        if not qo_segment_lens.is_cuda:
            raise ValueError("graph_safe=True requires qo_segment_lens on CUDA")
        if not kv_segment_lens.is_cuda:
            raise ValueError("graph_safe=True requires kv_segment_lens on CUDA")
        if isinstance(qo_offset, int):
            raise ValueError(
                "graph_safe=True requires qo_offset as a CUDA tensor; an int "
                "triggers an in-capture allocation")
        if isinstance(qo_offset, torch.Tensor) and not qo_offset.is_cuda:
            raise ValueError("graph_safe=True requires qo_offset tensor on CUDA")
        max_qo_len_orig = max_qo_len_override
        qo_len_uniform = qo_len_uniform_override
        batch_size = qo_segment_lens.shape[0]
    else:
        qo_lens_list = qo_segment_lens.tolist()
        max_qo_len_orig = max(qo_lens_list) if qo_lens_list else 0
        qo_len_uniform = len(qo_lens_list) > 0 and min(qo_lens_list) == max_qo_len_orig
        if max_qo_len_override > 0:
            assert max_qo_len_override == max_qo_len_orig, (
                f"max_qo_len_override={max_qo_len_override} != actual max(qo_lens)={max_qo_len_orig}")
        if qo_len_uniform_override is not None:
            assert qo_len_uniform_override == qo_len_uniform, (
                f"qo_len_uniform_override={qo_len_uniform_override} != actual={qo_len_uniform}")
        batch_size = qo_segment_lens.shape[0]

    _new_ws_cache()

    cuda_stream = stream

    _cfg_qo_len = 1 if (kv_block_num > 0 and not output_maxscore) else max_qo_len_orig
    qo_tile_size, kv_tile_size, num_occupancy = _get_indexer_qotile_and_occupancy(max_qo_len_orig, num_qo_heads, num_kv_heads, _get_num_cta(device)) if (output_maxscore or kv_block_num <= 0) else \
                                _get_fmha_qotile_and_occupancy(_cfg_qo_len, num_qo_heads, num_kv_heads, _get_num_cta(device), is_fp8=use_fp8_kvcache)
    # plan.cuh MAX_SMS=256 caps the number of scheduled CTA slots.
    num_ctas = min(_get_num_cta(device) * num_occupancy, 256)
    if usable_SM_count > 0:
        num_ctas = min(usable_SM_count, num_ctas)

    orig_num_qo_heads = num_qo_heads
    pack_factor = _compute_pack_factor(_cfg_qo_len, num_qo_heads, num_kv_heads)
    if pack_factor > 1:
        num_qo_heads = num_qo_heads // pack_factor

    # --- Determine num_kv_splits (deferred until total_qo_for_ws is known) ---
    # if qo_tile_size == 256:
    #     num_kv_splits = 1

    # ===================================================================
    # Device-side metadata computation (replaces .tolist() + Python loops)
    # ===================================================================
    if graph_safe:
        # CUDA residency already validated in the early guard block
        dev_qo_lens = qo_segment_lens
        dev_kv_lens = kv_segment_lens
    else:
        cuda_dev = torch.device('cuda', device if isinstance(device, int) else 0)
        dev_qo_lens = qo_segment_lens if qo_segment_lens.is_cuda else qo_segment_lens.to(cuda_dev)
        dev_kv_lens = kv_segment_lens if kv_segment_lens.is_cuda else kv_segment_lens.to(cuda_dev)

    # --- Workspace sizing ---
    is_sparse = (kv_block_num > 0 and page_size > 0)
    if is_sparse:
        # sparse: n = total_q (after per-token expansion)
        if max_total_q > 0:
            n = max_total_q
        else:
            n = int(dev_qo_lens.sum().item())
    else:
        n = batch_size

    ws_n = max_total_q if max_total_q > 0 else n
    # total_qo_len for workspace_o/lse sizing (sum of all qo tokens, AFTER pack_factor)
    if max_total_q > 0:
        total_qo_for_ws = max_total_q * pack_factor
    elif is_sparse:
        total_qo_for_ws = n * (pack_factor if pack_factor > 1 else 1)
    else:
        total_qo_for_ws = sum(qo_lens_list) * pack_factor

    # --- Determine num_kv_splits (adaptive or fixed) + workspace sizing ---
    # Shared with allocate_graph_workspace via _plan_splits_and_sizes so the
    # pre-allocated graph workspace always matches plan's requirement.
    # The budget-based formula derives the split ceiling from workspace cost,
    # giving a flexible cap (up to _MAX_ADAPTIVE_SPLITS) without D2H data reads.
    (plan_kv_splits, effective_num_kv_splits, _explicit_nosplit, max_work_items,
     lse_total_size, workspace_o_elems, total_bytes) = _plan_splits_and_sizes(
        is_sparse, qo_tile_size, num_ctas, n, ws_n, total_qo_for_ws,
        pack_factor, num_qo_heads, num_kv_splits,
        sparse_kv_tiles=(math.ceil(kv_block_num * page_size / kv_tile_size)
                         if is_sparse else 0),
        graph_safe=graph_safe)

    raw_ws = _get_plan_workspace_bytes(total_bytes, device, given_workspace)
    ws = slice_plan_workspace(raw_ws, ws_n, num_ctas, max_work_items,
                              total_qo_for_ws, lse_total_size, workspace_o_elems)

    # --- Compute metadata on device, write into workspace ---
    if True:  # unified kernel path (supports arbitrary n)
        # === Single kernel path ===
        dev_qo_off = None
        if qo_offset is not None:
            dev_qo_off = qo_offset if isinstance(qo_offset, torch.Tensor) else \
                torch.full((batch_size,), qo_offset, dtype=torch.int32, device=dev_qo_lens.device)
            if not dev_qo_off.is_cuda:
                dev_qo_off = dev_qo_off.to(dev_qo_lens.device)
        # dev_qo_off=None → kernel computes default (kv - qo) internally
        if max_kv_len_override > 0:
            max_kv_len_val = max_kv_len_override
        elif causal and not output_maxscore:
            max_kv_len_val = 0  # unused: kernel uses kv-qo, max_k_tiles=-1
        else:
            # non-causal or output_maxscore: need real max_kv_len (D2H, eager only)
            if graph_safe:
                raise ValueError("graph_safe=True requires max_kv_len_override when causal=False or output_maxscore=True")
            max_kv_len_val = int(dev_kv_lens.max().item())
        prepare_mod = get_prepare_metadata_fn()
        prepare_mod.prepare_plan_metadata(
            dev_qo_lens, dev_kv_lens, dev_qo_off,
            batch_size, n, page_size if page_size > 0 else 0, kv_block_num,
            pack_factor, max_kv_len_val, causal,
            ws.qo_segment_lens, ws.kv_segment_lens, ws.qo_offset,
            ws.plan_kv_lens, ws.kv_page_indptr,
            ws.qo_segment_offsets, ws.kv_segment_offsets,
            torch.cuda.current_stream().cuda_stream)
        kv_page_indptr_valid = (page_size > 0)
        plan_qo_offset = None if is_sparse else ws.qo_offset[:n]
        max_kv_len = max_kv_len_val
        total_qo_len = max_total_q if max_total_q > 0 else int(ws.qo_segment_offsets[n].item())
        max_qo_len = max_qo_len_orig
    else:
        # === Torch ops path (backward compatible) ===
        if is_sparse:
            # Sparse per-token expansion
            actual_n = int(dev_qo_lens.sum().item()) if max_total_q <= 0 else n
            ws.qo_segment_lens[:actual_n].fill_(pack_factor if pack_factor > 1 else 1)
            ws.kv_segment_lens[:actual_n].copy_(torch.repeat_interleave(dev_kv_lens, dev_qo_lens))

            # qo_offset: base[b] + within_batch_idx
            if qo_offset is not None:
                dev_qo_off = qo_offset if isinstance(qo_offset, torch.Tensor) else \
                    torch.full((batch_size,), qo_offset, dtype=torch.int32, device=device)
                if not dev_qo_off.is_cuda:
                    dev_qo_off = dev_qo_off.to(cuda_dev)
                base_offset = torch.repeat_interleave(dev_qo_off, dev_qo_lens)
            else:
                base_offset = torch.repeat_interleave(dev_kv_lens - dev_qo_lens, dev_qo_lens)
            cumsum_qo = torch.cumsum(dev_qo_lens, dim=0)
            starts = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev_qo_lens.device), cumsum_qo[:-1]])
            within_idx = torch.arange(actual_n, dtype=torch.int32, device=dev_qo_lens.device) - \
                torch.repeat_interleave(starts, dev_qo_lens)
            ws.qo_offset[:actual_n].copy_(base_offset + within_idx)

            # plan_kv_lens: constant
            ws.plan_kv_lens[:actual_n].fill_(kv_block_num * page_size)

            # kv_page_indptr (expanded): per-batch page starts repeated
            per_batch_pages = (dev_kv_lens + page_size - 1) // page_size
            batch_page_cumsum = torch.cumsum(per_batch_pages, dim=0)
            batch_page_starts = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev_kv_lens.device), batch_page_cumsum[:-1]])
            ws.kv_page_indptr[:actual_n].copy_(torch.repeat_interleave(batch_page_starts, dev_qo_lens))
            ws.kv_page_indptr[actual_n] = batch_page_cumsum[-1]  # total pages
            kv_page_indptr_valid = True
            plan_qo_offset = None
            n = actual_n
        else:
            # Non-sparse
            if pack_factor > 1:
                ws.qo_segment_lens[:n].copy_(dev_qo_lens * pack_factor)
            else:
                ws.qo_segment_lens[:n].copy_(dev_qo_lens)
            ws.kv_segment_lens[:n].copy_(dev_kv_lens)

            if causal:
                ws.qo_offset[:n].copy_(dev_kv_lens - dev_qo_lens)
            else:
                max_kv_val = int(dev_kv_lens.max().item()) if max_kv_len_override <= 0 else max_kv_len_override
                ws.qo_offset[:n].fill_(max_kv_val)

            ws.plan_kv_lens[:n].copy_(dev_kv_lens)

            if page_size > 0:
                ws.kv_page_indptr[0] = 0
                torch.cumsum((dev_kv_lens + page_size - 1) // page_size, dim=0,
                             out=ws.kv_page_indptr[1:n+1])
                kv_page_indptr_valid = True
            else:
                kv_page_indptr_valid = False
            plan_qo_offset = ws.qo_offset[:n]

        # --- cumsum → segment_offsets (device) ---
        ws.qo_segment_offsets[0] = 0
        torch.cumsum(ws.qo_segment_lens[:n], dim=0, out=ws.qo_segment_offsets[1:n+1])
        ws.kv_segment_offsets[0] = 0
        torch.cumsum(ws.kv_segment_lens[:n], dim=0, out=ws.kv_segment_offsets[1:n+1])

        # --- Python scalars (override or .item()) ---
        max_kv_len = max_kv_len_override if max_kv_len_override > 0 else int(dev_kv_lens.max().item())
        total_qo_len = int(ws.qo_segment_offsets[n].item()) if max_total_q <= 0 else max_total_q
        max_qo_len = max_qo_len_orig

    # --- Slice to actual size for kernel passing ---
    qo_segment_offsets = ws.qo_segment_offsets[:n+1]
    kv_segment_offsets = ws.kv_segment_offsets[:n+1]
    qo_offset = ws.qo_offset[:n]
    kv_segment_lens = ws.kv_segment_lens[:n]
    kv_page_indptr = ws.kv_page_indptr[:n+1] if kv_page_indptr_valid else None
    qo_segment_lens_gpu = ws.qo_segment_lens[:n]
    plan_kv_lens_gpu = ws.plan_kv_lens[:n]

    max_k_tiles = math.ceil(math.ceil(max_kv_len / 128) / 128) * 128 if output_maxscore else -1
    maxscore_elems = num_qo_heads * max_k_tiles * total_qo_len
    if output_maxscore and maxscore_elems > (1 << 31):
        print(f"Too huge setting to output maxscore!")
        max_k_tiles = -1

    cute_workspace_buffer = ws.cute_workspace_buffer

    # --- Adaptive split path (disabled for graph-safety) ---
    if False:  # adaptive_mode always False
        _call_plan(qo_segment_offsets,
            qo_segment_lens_gpu, plan_kv_lens_gpu,
            ws.packed_work_range, ws.packed_work_info,
            qo_tile_size, kv_tile_size,
            num_qo_heads, num_ctas, causal, plan_qo_offset,
            -max_kv_splits,
            ws.kv_tile_begin_indices, ws.kv_tile_end_indices, ws.kv_split_indices,
            chunk_size, ws.plan_cost, ws.num_kv_splits_per_row, ws.workspace_lse, lse_total_size,
            pack_factor, is_sparse, cuda_stream)

        _cost = ws.plan_cost.tolist()
        do_split = _cost[0] > 0
        predicted_speedup = _cost[1]

        if do_split:
            return _make_plan_info(
                qo_tile_size=qo_tile_size, kv_tile_size=kv_tile_size,
                packed_work_range=ws.packed_work_range, packed_work_info=ws.packed_work_info,
                kv_tile_begin_indices=ws.kv_tile_begin_indices, kv_tile_end_indices=ws.kv_tile_end_indices,
                kv_split_indices=ws.kv_split_indices, num_kv_splits=max_kv_splits, workspace_o=ws.workspace_o,
                workspace_lse=ws.workspace_lse, max_qo_len=max_qo_len, predicted_speedup=predicted_speedup,
                num_kv_splits_per_row=ws.num_kv_splits_per_row, qo_segment_offsets=qo_segment_offsets,
                kv_segment_offsets=kv_segment_offsets, kv_page_indptr=kv_page_indptr, max_k_tiles=max_k_tiles,
                qo_segment_lens=qo_segment_lens_gpu, kv_segment_lens=kv_segment_lens, qo_offset=qo_offset,
                pack_factor=pack_factor, orig_num_qo_heads=orig_num_qo_heads,
                qo_len_uniform=qo_len_uniform, cute_workspace_buffer=cute_workspace_buffer,
                page_size=page_size)
        else:
            # Re-run planner in fixed mode (num_kv_splits=1)
            _call_plan(qo_segment_offsets,
                qo_segment_lens_gpu, plan_kv_lens_gpu,
                ws.packed_work_range, ws.packed_work_info,
                qo_tile_size, kv_tile_size,
                num_qo_heads, num_ctas, causal, plan_qo_offset,
                1,
                None, None, None,
                0, None, None, None, 0,
                pack_factor, is_sparse, cuda_stream)
            return _make_plan_info(
                qo_tile_size=qo_tile_size, kv_tile_size=kv_tile_size,
                packed_work_range=ws.packed_work_range, packed_work_info=ws.packed_work_info,
                kv_tile_begin_indices=None, kv_tile_end_indices=None,
                kv_split_indices=None, num_kv_splits=1, workspace_o=None,
                workspace_lse=None, max_qo_len=max_qo_len, predicted_speedup=predicted_speedup,
                num_kv_splits_per_row=None, qo_segment_offsets=qo_segment_offsets,
                kv_segment_offsets=kv_segment_offsets, kv_page_indptr=kv_page_indptr, max_k_tiles=max_k_tiles,
                qo_segment_lens=qo_segment_lens_gpu, kv_segment_lens=kv_segment_lens, qo_offset=qo_offset,
                pack_factor=pack_factor, orig_num_qo_heads=orig_num_qo_heads,
                qo_len_uniform=qo_len_uniform, cute_workspace_buffer=cute_workspace_buffer,
                page_size=page_size)

    # --- Unified plan call (adaptive or fixed) ---
    _call_plan(qo_segment_offsets, qo_segment_lens_gpu, plan_kv_lens_gpu,
        ws.packed_work_range, ws.packed_work_info,
        qo_tile_size, kv_tile_size,
        num_qo_heads, num_ctas, causal, plan_qo_offset,
        plan_kv_splits,
        ws.kv_tile_begin_indices, ws.kv_tile_end_indices, ws.kv_split_indices,
        0, None, ws.num_kv_splits_per_row, ws.workspace_lse, lse_total_size,
        pack_factor, is_sparse, cuda_stream)
    return _make_plan_info(
        qo_tile_size=qo_tile_size, kv_tile_size=kv_tile_size,
        packed_work_range=ws.packed_work_range, packed_work_info=ws.packed_work_info,
        kv_tile_begin_indices=ws.kv_tile_begin_indices, kv_tile_end_indices=ws.kv_tile_end_indices,
        kv_split_indices=ws.kv_split_indices, num_kv_splits=effective_num_kv_splits,
        workspace_o=None if _explicit_nosplit else ws.workspace_o,
        workspace_lse=None if _explicit_nosplit else ws.workspace_lse,
        max_qo_len=max_qo_len, predicted_speedup=1.0,
        num_kv_splits_per_row=ws.num_kv_splits_per_row, qo_segment_offsets=qo_segment_offsets,
        kv_segment_offsets=kv_segment_offsets, kv_page_indptr=kv_page_indptr, max_k_tiles=max_k_tiles,
        qo_segment_lens=qo_segment_lens_gpu, kv_segment_lens=kv_segment_lens, qo_offset=qo_offset,
        pack_factor=pack_factor, orig_num_qo_heads=orig_num_qo_heads,
        qo_len_uniform=qo_len_uniform, cute_workspace_buffer=cute_workspace_buffer,
        graph_safe=graph_safe, page_size=page_size)



def _fmha_sail(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    o_scale: Optional[float] = None,
    output_maxscore: bool = True,
    output_o: bool = True,
    check_input_valid: bool = False,
    max_score_dtype: Optional[torch.dtype] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    nnz_qo, num_qo_heads, head_dim_qk = q.shape
    if kv_indices is None:
        nnz_kv, num_kv_heads, head_dim_vo = v.shape
        is_paged = False
        kv_is_nhd = False
        page_size = 0
    else:
        # HND/NHD detection anchored on the plan's page_size, which is
        # layout-independent. Deriving it from v.shape alone (the historical
        # behaviour) cannot tell HND from NHD. See _detect_paged_kv_layout.
        kv_is_nhd, num_kv_heads, page_size = _detect_paged_kv_layout(
            k.shape, plan_info.get("page_size", -1))
        nnz_kv = k.shape[0]
        head_dim_vo = v.shape[3]
        is_paged = True

    # fp8 kernels compute only max_score/sparse; q/k must both be float8_e4m3fn.
    max_k_tiles = plan_info["max_k_tiles"]
    if q.dtype == torch.float8_e4m3fn or k.dtype == torch.float8_e4m3fn:
        if q.dtype != k.dtype:
            raise ValueError(
                f"fmha_sail fp8 mode requires q/k to share the same dtype, "
                f"got q={q.dtype}, k={k.dtype}")
        if kv_block_indexes is not None:
            if v.dtype != q.dtype:
                raise ValueError("Sparse FP8 requires Q/K/V to use float8_e4m3fn")
            if not is_paged or page_size != 128 or head_dim_qk != 128 or head_dim_vo != 128:
                raise ValueError("Sparse FP8 requires paged KV with page_size=128 and head_dim=128")
            if not output_o or (output_maxscore and max_k_tiles != -1) or max_score is not None:
                raise ValueError("Sparse FP8 only supports O output: output_o=True and no max_score request")
        elif (output_o or not output_maxscore or max_k_tiles <= 0
                or (max_score is not None and max_score.dtype != torch.bfloat16)):
            raise ValueError(
                "fmha_sail fp8 e4m3fn inputs require output_o=False, "
                "output_maxscore=True, and max_k_tiles > 0")

    # max_score output is float32 by default; bf16 only when explicitly
    # requested via max_score_dtype or a preallocated bf16 buffer.
    if max_score_dtype is None:
        max_score_dtype = max_score.dtype if max_score is not None else torch.float32
    if max_score_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"fmha_sail max_score_dtype must be torch.float32 or "
            f"torch.bfloat16, got {max_score_dtype}")
    if max_score is not None and max_score.dtype != max_score_dtype:
        raise ValueError(
            f"max_score buffer dtype {max_score.dtype} does not match "
            f"max_score_dtype {max_score_dtype}")

    qo_total_len = nnz_qo

    qo_tile_size = plan_info["qo_tile_size"]
    kv_tile_size = plan_info["kv_tile_size"]
    packed_work_range = plan_info["packed_work_range"]
    packed_work_info = plan_info["packed_work_info"]
    kv_tile_begin_indices = plan_info["kv_tile_begin_indices"]
    kv_tile_end_indices = plan_info["kv_tile_end_indices"]
    kv_split_indices = plan_info["kv_split_indices"]
    num_kv_splits = plan_info["num_kv_splits"]
    workspace_o = plan_info["workspace_o"]
    workspace_lse = plan_info["workspace_lse"]
    plan_max_qo_len = plan_info["max_qo_len"]
    num_kv_splits_per_row = plan_info["num_kv_splits_per_row"]
    qo_segment_offsets = plan_info["qo_segment_offsets"]
    kv_segment_offsets = plan_info["kv_segment_offsets"]
    kv_page_indptr = plan_info["kv_page_indptr"]
    qo_segment_lens = plan_info["qo_segment_lens"]
    kv_segment_lens = plan_info["kv_segment_lens"]
    qo_offset = plan_info["qo_offset"] if q_offset_override is None else q_offset_override
    pack_factor = plan_info["pack_factor"]
    orig_num_qo_heads = plan_info["orig_num_qo_heads"]
    qo_len_uniform = plan_info["qo_len_uniform"]
    workspace_buffer = plan_info["cute_workspace_buffer"]

    batch_size = qo_segment_lens.shape[0]

    if isinstance(qo_offset, int):
        if plan_info.get("graph_safe", False):
            raise ValueError(
                "graph_safe=True requires q_offset_override as a CUDA tensor; "
                "an int triggers an in-capture torch.full_like allocation")
        qo_offset = torch.full_like(qo_segment_lens, qo_offset)
    elif qo_offset is not None:
        assert qo_offset.device == qo_segment_lens.device

    if check_input_valid:
        _validate_fmha_inputs(
            q, k, v, qo_segment_lens, kv_segment_lens,
            num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
            batch_size, qo_total_len, is_paged,
            kv_indices, kv_block_indexes, qo_offset,
            page_size if is_paged else 0,
            pack_factor=pack_factor,
            packed_work_range=packed_work_range,
            packed_work_info=packed_work_info,
            num_kv_splits=num_kv_splits,
            kv_tile_begin_indices=kv_tile_begin_indices,
            kv_tile_end_indices=kv_tile_end_indices,
            kv_split_indices=kv_split_indices,
            qo_segment_offsets=qo_segment_offsets,
            kv_page_indptr=kv_page_indptr,
            qo_tile_size=qo_tile_size,
            kv_tile_size=kv_tile_size,
            kv_is_nhd=kv_is_nhd,
        )

    if pack_factor > 1 and orig_num_qo_heads is not None:
        num_qo_heads = orig_num_qo_heads // pack_factor
        qo_total_len = nnz_qo * pack_factor

    max_qo_len = plan_max_qo_len

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim_qk)
    if q_scale is None:
        q_scale = 1.0
    if k_scale is None:
        k_scale = 1.0
    if v_scale is None:
        v_scale = 1.0
    if o_scale is None:
        o_scale = 1.0

    assert output_o or output_maxscore

    # workspace_o is None when the plan used explicit nosplit (num_kv_splits=1):
    # run then takes the non-split variant and skips the reduction kernel.
    use_split_kv = (workspace_o is not None)

    if not output_maxscore or max_k_tiles == -1:
        max_score = None
        output_o = True
    elif max_score is None and max_k_tiles > 0:
        max_score = torch.full(
            (orig_num_qo_heads, max_k_tiles, nnz_qo),
            -float("inf"),
            dtype=max_score_dtype,
            device=q.device)

    if not output_o:
        out = None
    elif out is None:
        out_dtype = torch.bfloat16 if q.dtype.itemsize == 1 else q.dtype
        out = torch.empty(
            nnz_qo,
            orig_num_qo_heads, head_dim_vo,
            device=q.device, dtype=out_dtype,
        )

    # Determine variant dispatch parameters
    dtype_code = _dlpack_dtype_code(q.dtype)

    kv_block_indexes_ptr_exists = kv_block_indexes is not None
    max_score_exists = max_score is not None
    o_exists = out is not None
    if kv_block_indexes_ptr_exists:
        sparse_mode = 0
    elif max_score_exists and o_exists:
        sparse_mode = 1
    elif max_score_exists:
        # OnlyScoreBF16: identical compute, max_score stored as bf16.
        sparse_mode = 3 if max_score_dtype == torch.bfloat16 else 2
    else:
        sparse_mode = None  # Off (matches the catch-all dispatch entry)

    if max_score_dtype == torch.bfloat16 and sparse_mode != 3:
        raise ValueError(
            "max_score_dtype=torch.bfloat16 requires OnlyScore mode "
            "(output_maxscore=True, output_o=False, no kv_block_indexes)")

    variant_page_size = (page_size if is_paged else -1)

    variant_module = get_fmha_variant(
        dtype_code, qo_tile_size, kv_tile_size,
        (max_qo_len <= 64), sparse_mode, variant_page_size, use_split_kv, pack_factor)

    variant_module.run(
        workspace_buffer,
        q, k, v,
        qo_segment_lens, kv_segment_lens,
        qo_segment_offsets, kv_segment_offsets,
        packed_work_range, packed_work_info,
        out,
        sm_scale, q_scale, k_scale, v_scale, o_scale,
        max_qo_len,
        qo_offset,
        num_kv_splits,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        workspace_o, workspace_lse,
        num_kv_splits_per_row,
        qo_tile_size,
        kv_indices, kv_page_indptr,
        max_score,
        max_k_tiles,
        kv_block_indexes,
        pack_factor,
        bool(qo_len_uniform),
        torch.cuda.current_stream().cuda_stream,
    )

    # Split-KV reduction
    if use_split_kv and out is not None:
        log2_e = math.log2(math.exp(1.0))
        scale_softmax_log2 = float(q_scale * k_scale * sm_scale) * log2_e
        inv_scale_o = float(o_scale)

        # The plan's num_kv_splits is a workspace-budget ceiling (e.g. 101) far
        # above the device-side actual split count. Clamp it to the sparse KV
        # tile upper bound (topk * page_size / kv_tile_size) so the reduction
        # kernel dispatches to the smallest power-of-two template bucket that
        # still covers every row's actual num_kv_splits_per_row. The kernel only
        # reads num_kv_splits_per_row for real work, so tightening this bound is
        # correctness-neutral; it just picks the template instantiation.
        reduction_split_cap = num_kv_splits
        if kv_block_indexes is not None and kv_tile_size > 0:
            kv_block_num_eff = kv_block_indexes.shape[2]
            sparse_tiles = (kv_block_num_eff * page_size + kv_tile_size - 1) // kv_tile_size
            reduction_split_cap = min(num_kv_splits, sparse_tiles)

        reduction_module = get_reduction_module()
        reduction_module.reduction(
            workspace_o,
            out,
            workspace_lse,
            num_kv_splits_per_row,
            scale_softmax_log2, inv_scale_o,
            reduction_split_cap, qo_total_len, num_qo_heads, head_dim_vo,
            num_qo_heads * head_dim_vo, head_dim_vo,
            num_qo_heads * head_dim_vo, head_dim_vo,
            orig_num_qo_heads,
            num_kv_heads,
            pack_factor,
            torch.cuda.current_stream().cuda_stream,
        )

    return out, max_score


def fmha_sail_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    *args,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    # Not graph-compatible: D2H length checks + data-dependent dual sub-plans.
    split_prefill_decode = False,
    **kwargs
):
    """Build a reusable execution plan for ``fmha_sail``.

    The plan is shape-dependent and can be reused across layers or repeated
    calls that share the same sequence lengths, head counts, page size, sparse
    mode, and output mode.  Planning may run CUDA kernels and allocates
    workspaces, so it should be done outside tight per-layer loops when
    possible.

    Parameters
    ----------
    qo_segment_lens : torch.Tensor
        Shape ``[batch_size]``, dtype int32/int64.  Per-request Q/O lengths.
        CPU tensors are accepted for the dense planner; sparse prefill planning
        moves them to CUDA internally.
    kv_segment_lens : torch.Tensor
        Shape ``[batch_size]``, dtype int32/int64.  Per-request KV lengths.
    *args
        Positional arguments forwarded to the internal planner.  In normal
        usage this is ``num_qo_heads`` and optionally ``num_kv_heads``.
    qo_offset : int or torch.Tensor, optional
        Per-request causal offset.  If omitted, defaults to
        ``kv_segment_lens - qo_segment_lens`` for bottom-right causal masking.
        A tensor must have shape ``[batch_size]``.
    split_prefill_decode : bool, optional
        Default False.  If True, a mixed batch ordered as decode requests
        followed by prefill requests is split into two sub-plans.  The
        original order must already group short decode sequences before long
        prefill sequences.  Enabling it costs D2H syncs at plan/run time.

        NOT graph-compatible: the split decision reads lengths back to host
        (``.item()``), and the two sub-plans are data-dependent — both break
        CUDA Graph capture/replay.  Under ``graph_safe=True`` it is force-
        disabled; mixed batches must be split by the caller outside capture.
    **kwargs
        Planner options forwarded to ``_fmha_sail_plan``.  Common options are
        ``num_kv_heads``, ``num_kv_splits``, ``page_size``,
        ``output_maxscore``, ``kv_block_num``, ``usable_SM_count``, ``causal``,
        ``sparse_kernel_mode``, ``use_fp8_kvcache``, ``device``, and ``stream``.

    Returns
    -------
    tuple
        ``(has_mixed_prefill, split, batch_size, decode_plan, prefill_plan)``.
        Pass this tuple unchanged as ``plan_info`` to ``fmha_sail``.
    """

    # assert qo_segment_lens.device.type == 'cpu' \
    #         and kv_segment_lens.device.type == 'cpu'
    # assert qo_offset is None or isinstance(qo_offset, int) or qo_offset.device.type == 'cpu'

    if qo_offset is not None and isinstance(qo_offset, int):
        if kwargs.get("graph_safe", False):
            raise ValueError(
                "graph_safe=True requires qo_offset as a CUDA tensor; an int "
                "triggers an in-capture torch.full_like allocation")
        qo_offset = torch.full_like(qo_segment_lens, qo_offset)
    # qo_offset=None: passed through, kernel/torch ops computes default (kv - qo) on device

    batch_size = qo_segment_lens.shape[0]
    has_mixed_prefill = False
    # graph_safe mode forbids D2H sync (.item()); even if a caller explicitly
    # passes split_prefill_decode=True, force it off under graph_safe.
    # Callers needing mixed prefill/decode under graph-safe must split
    # manually outside capture and call fmha_sail_plan per sub-batch.
    if split_prefill_decode and kwargs.get("graph_safe", False):
        split_prefill_decode = False
    if split_prefill_decode:
        qmax = qo_segment_lens.max().item()
        sparse = kwargs.get("kv_block_num", -1) > 0
        split_threshold = _prefill_qlen_threshold(sparse)
        if qmax > split_threshold:
            split = (qo_segment_lens > split_threshold).nonzero(as_tuple=False)[0, 0].item()
            has_mixed_prefill = split > 0
    if has_mixed_prefill:
        # print(f"Split into 2 parts at index {split}")
        decode_qo_segment_lens = qo_segment_lens[:split]
        decode_kv_segment_lens = kv_segment_lens[:split]
        decode_qo_offset = qo_offset[:split] if qo_offset is not None else None
        decode = _fmha_sail_plan(decode_qo_segment_lens, decode_kv_segment_lens, *args,
                                    qo_offset=decode_qo_offset, **kwargs)
        decode = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in decode.items()}
        prefill_qo_segment_lens = qo_segment_lens[split:]
        prefill_kv_segment_lens = kv_segment_lens[split:]
        prefill_qo_offset = qo_offset[split:] if qo_offset is not None else None
        prefill = _fmha_sail_plan(prefill_qo_segment_lens, prefill_kv_segment_lens, *args,
                                    qo_offset=prefill_qo_offset, **kwargs)
        return (True, split, batch_size, decode, prefill)
    else:
        plan = _fmha_sail_plan(qo_segment_lens, kv_segment_lens, *args,
                                    qo_offset=qo_offset, **kwargs)
        return (False, 0, batch_size, plan, None)

def fmha_sail(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run dense, paged, or sparse PPU FMHA using a precomputed plan.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[total_qo_len, num_qo_heads, head_dim]``.  Supported dtypes are
        ``torch.bfloat16``, and ``torch.float8_e4m3fn`` (OnlyScore/Sparse).
        The head dim must be 128.
    k : torch.Tensor
        Dense layout ``[total_kv_len, num_kv_heads, head_dim]`` or paged layout
        ``[total_pages, num_kv_heads, page_size, head_dim]``.
    v : torch.Tensor
        Same layout as ``k``.  The output head dimension follows ``v.shape[-1]``.
    plan_info : tuple
        Return value from ``fmha_sail_plan`` for the same lengths, head layout,
        page size, and sparse/output mode.
    kv_indices : torch.Tensor, optional
        Paged-KV physical page table, flattened across the batch.  Required when
        ``k`` and ``v`` use paged layout.  Shape is ``[sum_pages]`` and dtype is
        int32.
    kv_block_indexes : torch.Tensor, optional
        Sparse KV block indices from ``sparse_topk_select``.  Shape
        ``[total_qo_len, num_kv_heads or num_qo_heads, kv_block_num]``, dtype
        int32, ascending per row with ``-1`` padding at the tail.
    q_offset_override : int or torch.Tensor, optional
        Runtime causal-offset override.  Tensor form must have shape
        ``[batch_size]`` on the same CUDA device as the plan metadata.  The
        override must stay within the causal range visible to the original
        plan.
    out : torch.Tensor, optional
        Preallocated output buffer with shape
        ``[total_qo_len, num_qo_heads, head_dim_v]``.
    max_score : torch.Tensor, optional
        Preallocated per-KV-tile score buffer with shape
        ``[num_qo_heads, max_k_tiles, total_qo_len]`` and dtype float32
        (bfloat16 accepted in OnlyScore mode only; see ``max_score_dtype``).
    **kwargs
        Runtime options forwarded to the kernel runner.  Common options are
        ``sm_scale``, ``q_scale``, ``k_scale``, ``v_scale``, ``o_scale``,
        ``output_maxscore``, ``output_o``, ``max_score_dtype``
        (``torch.float32`` default; ``torch.bfloat16`` selects the
        OnlyScoreBF16 variant, OnlyScore mode only), and
        ``check_input_valid``.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        ``(out, max_score)``.  Either item may be ``None`` if the corresponding
        output was disabled.  When both decode and prefill sub-plans are used,
        outputs are concatenated back into the original batch order.
    """
    has_mixed_prefill, split, batch_size, decode, prefill = plan_info
    if not has_mixed_prefill:
        return _fmha_sail(q, k, v, decode, out=out, max_score=max_score, kv_indices=kv_indices,kv_block_indexes=kv_block_indexes, q_offset_override=q_offset_override, **kwargs)
    else:

        decode_pack = decode.get("pack_factor", 1)
        decode_nnz = decode["qo_segment_offsets"][-1].item() // decode_pack
        is_paged = kv_indices is not None
        nnz_qo = q.shape[0]
        num_qo_heads = q.shape[1]

        q_decode = q[:decode_nnz]
        q_prefill = q[decode_nnz:]

        if is_paged:
            k_decode, v_decode = k, v
            k_prefill, v_prefill = k, v
            if "kv_page_indptr" in decode:
                kv_page_split = decode["kv_page_indptr"][-1].item()
            else:
                kv_page_split = decode["total_rows"]
            decode_kv_indices = kv_indices[:kv_page_split]
            prefill_kv_indices = kv_indices[kv_page_split:]
        else:
            if "kv_segment_offsets" in decode:
                decode_kv_nnz = decode["kv_segment_offsets"][-1].item()
            else:
                decode_kv_nnz = decode["cu_seqlens_k"][-1].item()
            k_decode, k_prefill = k[:decode_kv_nnz], k[decode_kv_nnz:]
            v_decode, v_prefill = v[:decode_kv_nnz], v[decode_kv_nnz:]
            decode_kv_indices = None
            prefill_kv_indices = None

        decode_block_idx = kv_block_indexes[:decode_nnz] if kv_block_indexes is not None else None
        prefill_block_idx = kv_block_indexes[decode_nnz:] if kv_block_indexes is not None else None
        if isinstance(q_offset_override, int):
            decode_qo_offset = torch.full((split,), q_offset_override, dtype=torch.int32, device=q.device)
            prefill_qo_offset = torch.full((batch_size - split,), q_offset_override, dtype=torch.int32, device=q.device)
        elif q_offset_override is not None:
            decode_qo_offset = q_offset_override[:split]
            prefill_qo_offset = q_offset_override[split:]
        else:
            decode_qo_offset = None
            prefill_qo_offset = None

        # ---- Run kernels ----
        decode_out, decode_ms = _fmha_sail(
            q_decode, k_decode, v_decode, decode,
            out=None, max_score=None,
            kv_indices=decode_kv_indices, kv_block_indexes=decode_block_idx,
            q_offset_override=decode_qo_offset,
            **kwargs)
        prefill_out, prefill_ms = _fmha_sail(
            q_prefill, k_prefill, v_prefill, prefill,
            out=None, max_score=None,
            kv_indices=prefill_kv_indices, kv_block_indexes=prefill_block_idx,
            q_offset_override=prefill_qo_offset,
            **kwargs)

        # ---- Merge out ----
        if decode_out is not None and prefill_out is not None:
            combined_out = torch.cat([decode_out, prefill_out], dim=0)
        else:
            combined_out = None

        if out is not None and combined_out is not None:
            out.copy_(combined_out)

        # ---- Merge max_score ----
        if decode_ms is not None and prefill_ms is not None:
            d_kt, p_kt = decode_ms.shape[1], prefill_ms.shape[1]
            max_kt = max(d_kt, p_kt)
            combined_ms = torch.full(
                (num_qo_heads, max_kt, nnz_qo),
                -float("inf"), dtype=decode_ms.dtype, device=q.device)
            combined_ms[:, :d_kt, :decode_nnz] = decode_ms
            combined_ms[:, :p_kt, decode_nnz:] = prefill_ms
        else:
            combined_ms = decode_ms if decode_ms is not None else prefill_ms

        if max_score is not None and combined_ms is not None:
            max_score.copy_(combined_ms)

        return (out if out is not None else combined_out,
                max_score if max_score is not None else combined_ms)


# ============================================================================
# Sparse TopK Select
# ============================================================================


def sparse_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor:
    r"""Select top-k KV-tile indices per (qo_head, token) row from the FMHA max-score tensor.

    Designed for the MQA proxy-KV sparse attention path where the dense pass uses
    ``num_kv_heads_dense=1``, so ``max_score.shape[0] == num_kv_heads_real`` and
    each head row is processed independently (no GQA reduction inside this call).

    Parameters
    ----------
    max_score : torch.Tensor
        Shape ``(num_qo_heads, max_k_tiles, total_qo_len)``, contiguous, float32.
        Slots beyond the actual KV tile count must be pre-filled with ``-inf``
        (fmha_sail does this automatically via ``torch.full``).
    topk : int
        Must be exactly 16.
    num_valid_pages : int, optional
        Actual number of KV pages in the page table, i.e. ``ceil(kv_len / page_size)``.
        ``max_k_tiles`` is round-up-aligned and always >= ``num_valid_pages``.
        The kernel may select tile indices in ``[num_valid_pages, max_k_tiles-1]``
        (all-``-inf`` padding tiles). Passing ``num_valid_pages`` replaces those
        out-of-range indices with ``-1`` and sorts them to the tail, matching the
        sparse FMHA kernel's kv_block_indexes contract.
        **Strongly recommended**: omitting this allows OOB page-table accesses in
        the sparse attention pass.
    force_begin_blocks : int
        Number of KV blocks at the beginning of the sequence (indices 0..N-1) to
        always include in the top-k result, regardless of their scores.  Useful
        for sink tokens.  Default 0.
    force_end_blocks : int
        Number of KV blocks at the end of the valid sequence (indices
        nvp-N..nvp-1, closest to the current query) to always include.  Useful
        for local-window attention.  Default 0.

    Returns
    -------
    torch.Tensor
        Shape ``(total_qo_len, num_qo_heads, topk)``, int32, ascending by tile index.
        Out-of-range entries (if any) are ``-1`` at the tail.
    """

    assert max_score.dtype == torch.float32 or max_score.dtype == torch.bfloat16,\
                    f"max_score must be float32 or bfloat16, got {max_score.dtype}"
    assert max_score.dim() == 3, f"max_score must be 3D, got {max_score.shape}"
    assert max_score.is_contiguous(), "max_score must be contiguous"
    assert topk == 16, f"topk must be 16, got {topk}"

    num_qo_heads, max_k_tiles, total_qo_len = max_score.shape

    # v2.3 kernel only supports the insertion-sort path (K < 12288).
    assert max_k_tiles < 12288, (
        f"max_k_tiles={max_k_tiles} >= 12288: v2.3 kernel only supports K < 12288 "
        f"(radix-sort path not yet implemented). kv_len must be < {12288 * 128} tokens."
    )

    if num_valid_pages is not None:
        assert 0 < num_valid_pages <= max_k_tiles, (
            f"num_valid_pages={num_valid_pages} must be in (0, max_k_tiles={max_k_tiles}]"
        )
        nvp_arg = int(num_valid_pages)
    else:
        # v2.5_oob_clamp_in_kernel: kernel takes a unified num_valid_pages arg.
        # When the caller doesn't supply one, pass max_k_tiles so the in-kernel
        # `idx >= num_valid_pages` check never triggers (idx is always in
        # [0, max_k_tiles)).
        nvp_arg = int(max_k_tiles)

    assert force_begin_blocks >= 0 and force_end_blocks >= 0, (
        f"force_begin_blocks={force_begin_blocks} and force_end_blocks={force_end_blocks} "
        f"must be non-negative"
    )
    assert force_begin_blocks + force_end_blocks <= topk, (
        f"force_begin_blocks({force_begin_blocks}) + force_end_blocks({force_end_blocks}) "
        f"= {force_begin_blocks + force_end_blocks} exceeds topk={topk}"
    )

    # Workspace = transpose_buf only: (num_qo_heads, max_k_tiles, total_qo_len) fp32.
    workspace_size = num_qo_heads * max_k_tiles * total_qo_len  # elements

    workspace_buffer = _alloc_workspace_buf(_BuffTag.sparse_topk_workspace, workspace_size, max_score.device, max_score.dtype)

    if output is not None:
        output_indices = output
    else:
        output_indices = torch.empty(
            total_qo_len, num_qo_heads, topk,
            dtype=torch.int32, device=max_score.device,
        )

    module = get_sparse_topk_module()
    # MQA dense pass: num_kv_heads_dense=1, so h_r=num_qo_heads.
    # The kernel only uses num_qo_heads = h_r * num_kv_heads as a product;
    # passing num_kv_heads=1 is equivalent to any other valid factorisation.
    #
    # v2.5_oob_clamp_in_kernel: OOB clamp is folded into the kernel — the prior
    # post-process torch.where + sort + torch.where chain (~84-101 us / call)
    # is replaced by passing num_valid_pages directly to the kernel.
    module.sparse_topk_select(
        max_score, output_indices, workspace_buffer,
        topk,
        nvp_arg,
        int(force_begin_blocks),
        int(force_end_blocks),
        torch.cuda.current_stream().cuda_stream,
    )

    return output_indices
