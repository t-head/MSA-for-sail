# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""CuTe-free runtime for sparse FMHA forward + combine (Phase C).

Provides ``sparse_fmha_plan``, ``sparse_fmha_fwd``, and ``sparse_fmha_combine``
that operate on pre-built schedule metadata from Phase B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from .jit import get_ppu_arch
from .sparse_jit import (
    get_sparse_fmha_fwd_module,
    get_sparse_fmha_combine_module,
    sparse_fwd_variant_name,
)

_SUPPORTED_HEAD_DIM = (128,)
_SUPPORTED_TOPK = (4, 8, 16, 32)
_SUPPORTED_QKV_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_TEMPERATURE_LSE_FAST_PATH_ABS_TOL = 1e-12
_PARTIAL_DTYPE = torch.bfloat16


class SparseFwdDispatchKey:
    """Integer dispatch key sent to the C++ host launcher."""

    BF16 = 0
    FP8 = 1


@dataclass(frozen=True)
class SparseDtypeConfig:
    """Resolved Element dtype for a sparse forward call."""

    element: torch.dtype
    dispatch_key: int


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype == torch.float8_e4m3fn


def resolve_sparse_dtypes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> SparseDtypeConfig:
    """Validate the Q/K/V dtype and return the kernel dispatch config.

    Q, K and V must share one Element (bf16 or fp8 e4m3fn).  That Element is
    also the in-kernel conversion target after softmax; O_partial is bf16.
    """
    q_dt, k_dt, v_dt = q.dtype, k.dtype, v.dtype

    for name, dt in (("Q", q_dt), ("K", k_dt), ("V", v_dt)):
        if dt not in _SUPPORTED_QKV_DTYPES:
            raise TypeError(
                f"{name} dtype must be bf16 or fp8 e4m3fn "
                f"(one of {_SUPPORTED_QKV_DTYPES}), got {dt}"
            )

    if k_dt != q_dt or v_dt != q_dt:
        raise TypeError(
            "sparse_fmha_fwd requires Q, K and V to share one dtype "
            f"(bf16 or fp8 e4m3fn); got Q={q_dt}, K={k_dt}, V={v_dt}. "
            "Mixed bf16/fp8 combinations are not supported."
        )

    if _is_fp8(q_dt) and get_ppu_arch() != "ppu_15":
        raise TypeError(
            "fp8 sparse FMHA requires PPU0015; this device is "
            f"{get_ppu_arch()}, use bf16 Q/K/V"
        )

    return SparseDtypeConfig(
        element=q_dt,
        dispatch_key=(
            SparseFwdDispatchKey.FP8 if _is_fp8(q_dt)
            else SparseFwdDispatchKey.BF16
        ),
    )


def _is_fast_path_temperature(scale: Optional[float]) -> bool:
    return scale is None or math.isclose(
        scale, 1.0, rel_tol=0.0, abs_tol=_TEMPERATURE_LSE_FAST_PATH_ABS_TOL
    )


def _detect_paged_kv_layout(shape, page_size: int) -> Tuple[bool, int]:
    """Return ``(is_nhd, num_kv_heads)`` for a 4D paged K/V.

    NHD ``[pages, page_size, Hkv, D]`` is the default layout (the production
    sglang KV cache is slot-major); HND ``[pages, Hkv, page_size, D]`` is
    detected when dim1 is not ``page_size``.  When ``Hkv`` equals
    ``page_size`` the shape is ambiguous and the probe resolves to NHD.  The
    probe matches the C++ host binding, which derives the same flag.
    """
    if len(shape) != 4:
        raise ValueError(
            "paged K/V must be 4D ([pages, ps, Hkv, D] NHD or "
            f"[pages, Hkv, ps, D] HND), got shape {tuple(shape)}")
    if shape[1] == page_size:
        return True, int(shape[2])
    if shape[2] == page_size:
        return False, int(shape[1])
    raise ValueError(
        f"cannot infer paged K/V layout: shape={tuple(shape)} has neither "
        f"dim1 nor dim2 equal to page_size={page_size}")


@dataclass
class SparseFmhaPlan:
    """Pre-allocated buffers and derived scalars for sparse forward + combine."""

    O_partial: torch.Tensor
    LSE_partial: torch.Tensor
    O: torch.Tensor
    LSE: torch.Tensor

    LSE_temperature_partial: Optional[torch.Tensor]
    LSE_temperature: Optional[torch.Tensor]

    head_dim: int
    num_heads_kv: int
    num_heads_q: int
    qhead_per_kv: int
    total_q: int
    topk: int
    max_seqlen_q: int
    is_paged: int
    page_size: int
    work_capacity: int
    sm_scale: float
    lse_temperature_scale: float
    return_temperature_lse: bool

    dtype_config: SparseDtypeConfig

    seqused_k: Optional[torch.Tensor] = None


def sparse_fmha_plan(
    *,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kv_indices: Optional[torch.Tensor] = None,
    kv_page_indptr: Optional[torch.Tensor] = None,
    schedule=None,
    sm_scale: float,
    head_dim: int,
    num_heads_kv: int,
    qhead_per_kv: int,
    total_q: int,
    max_seqlen_q: int,
    topk: int,
    is_paged: bool = False,
    page_size: int = 128,
    work_capacity: Optional[int] = None,
    lse_temperature_scale: Optional[float] = None,
    seqused_k: Optional[torch.Tensor] = None,
) -> SparseFmhaPlan:
    """Validate inputs and pre-allocate all output buffers."""

    dtype_config = resolve_sparse_dtypes(Q, K, V)
    if head_dim not in _SUPPORTED_HEAD_DIM:
        raise ValueError(f"head_dim must be in {_SUPPORTED_HEAD_DIM}, got {head_dim}")
    if topk not in _SUPPORTED_TOPK:
        raise ValueError(f"topk must be in {_SUPPORTED_TOPK}, got {topk}")
    if max_seqlen_q >= (1 << 24):
        raise ValueError(f"max_seqlen_q must be < 2^24, got {max_seqlen_q}")
    if not (K.is_contiguous() and V.is_contiguous()):
        raise ValueError("K, V must be contiguous")
    if Q.dim() != 3 or Q.stride(-1) != 1:
        raise ValueError(
            "Q must be 3D with a dense last axis (stride(-1)==1), got shape "
            f"{tuple(Q.shape)} strides {Q.stride()}")
    if Q.stride(0) <= 0 or Q.stride(1) <= 0:
        raise ValueError(
            f"Q token/head strides must be positive, got {Q.stride()}")
    q_align = 16 // Q.element_size()
    if Q.stride(0) % q_align or Q.stride(1) % q_align:
        raise ValueError(
            "Q token/head strides must be 16B-aligned, got "
            f"{Q.stride()[:2]} (element_size={Q.element_size()})")
    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise ValueError("Q, K, V must be CUDA tensors")
    if Q.device != K.device or Q.device != V.device:
        raise ValueError("Q, K, V must be on the same device")

    num_heads_q = num_heads_kv * qhead_per_kv
    if Q.size(1) != num_heads_q:
        raise ValueError(
            f"Q.size(1) ({Q.size(1)}) != num_heads_kv * qhead_per_kv ({num_heads_q})"
        )
    if Q.size(2) != head_dim:
        raise ValueError(f"Q.size(2) ({Q.size(2)}) != head_dim ({head_dim})")
    if Q.size(0) != total_q:
        raise ValueError(f"Q.size(0) ({Q.size(0)}) != total_q ({total_q})")

    if is_paged:
        if K.dim() != 4 or V.dim() != 4:
            raise ValueError(
                "paged K/V must be 4D ([pages, Hkv, ps, D] HND or "
                f"[pages, ps, Hkv, D] NHD), got K={tuple(K.shape)} "
                f"V={tuple(V.shape)}")
        if K.shape != V.shape:
            raise ValueError(
                f"paged K and V must share a shape, got K={tuple(K.shape)} "
                f"V={tuple(V.shape)}")
        _, k_heads = _detect_paged_kv_layout(K.shape, page_size)
        if k_heads != num_heads_kv:
            raise ValueError(
                f"paged K has {k_heads} KV heads but num_heads_kv="
                f"{num_heads_kv} (shape={tuple(K.shape)}, "
                f"page_size={page_size})")
        if K.size(3) != head_dim:
            raise ValueError(
                f"paged K/V last dim ({K.size(3)}) != head_dim ({head_dim})")
        if kv_indices is None or kv_page_indptr is None:
            raise ValueError("paged mode requires kv_indices and kv_page_indptr")
        if kv_indices.dtype != torch.int32:
            raise TypeError("kv_indices must be int32")
        if kv_page_indptr.dtype != torch.int32:
            raise TypeError("kv_page_indptr must be int32")
    else:
        if kv_indices is None:
            kv_indices = torch.zeros(1, dtype=torch.int32, device=Q.device)
        if kv_page_indptr is None:
            kv_page_indptr = torch.zeros(1, dtype=torch.int32, device=Q.device)

    batch = cu_seqlens_q.size(0) - 1
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            raise TypeError("seqused_k must be int32")
        if seqused_k.dim() != 1 or seqused_k.size(0) != batch:
            raise ValueError(f"seqused_k must have shape [batch={batch}]")
        if not seqused_k.is_cuda:
            raise ValueError("seqused_k must be a CUDA tensor")
        if seqused_k.device != Q.device:
            raise ValueError("seqused_k must be on the same device as Q")

    if work_capacity is None:
        if schedule is not None:
            work_capacity = schedule.work_capacity
        else:
            work_capacity = 0

    fast_path = _is_fast_path_temperature(lse_temperature_scale)
    need_temperature = not fast_path
    temp_scale_val = float(lse_temperature_scale) if lse_temperature_scale is not None else 0.0

    device = Q.device

    O_partial = torch.empty(
        (total_q, num_heads_q, topk, head_dim),
        dtype=_PARTIAL_DTYPE, device=device,
    )
    LSE_partial = torch.empty(
        (total_q, num_heads_q, topk),
        dtype=torch.float32, device=device,
    )
    O = torch.empty(
        (total_q, num_heads_q, head_dim),
        dtype=torch.bfloat16, device=device,
    )
    LSE = torch.empty(
        (total_q, num_heads_q),
        dtype=torch.float32, device=device,
    )

    if need_temperature:
        LSE_temperature_partial = torch.empty(
            (total_q, num_heads_q, topk),
            dtype=torch.float32, device=device,
        )
        LSE_temperature = torch.empty(
            (total_q, num_heads_q),
            dtype=torch.float32, device=device,
        )
    else:
        LSE_temperature_partial = None
        LSE_temperature = None

    return SparseFmhaPlan(
        O_partial=O_partial,
        LSE_partial=LSE_partial,
        O=O,
        LSE=LSE,
        LSE_temperature_partial=LSE_temperature_partial,
        LSE_temperature=LSE_temperature,
        head_dim=head_dim,
        num_heads_kv=num_heads_kv,
        num_heads_q=num_heads_q,
        qhead_per_kv=qhead_per_kv,
        total_q=total_q,
        topk=topk,
        max_seqlen_q=max_seqlen_q,
        is_paged=int(is_paged),
        page_size=page_size,
        work_capacity=work_capacity,
        sm_scale=float(sm_scale),
        lse_temperature_scale=temp_scale_val,
        return_temperature_lse=need_temperature,
        dtype_config=dtype_config,
        seqused_k=seqused_k,
    )


def sparse_fmha_fwd(
    *,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    qsplit_indices: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    work_count: torch.Tensor,
    split_counts: torch.Tensor,
    plan: SparseFmhaPlan,
) -> None:
    """Launch the sparse forward kernel (in-place writes to plan.O_partial / plan.LSE_partial)."""

    if plan.head_dim != 128:
        raise ValueError(
            f"sparse_fmha_fwd only supports head_dim=128, got {plan.head_dim}"
        )

    variant = sparse_fwd_variant_name(
        plan.qhead_per_kv, plan.dtype_config.dispatch_key
    )
    module = get_sparse_fmha_fwd_module(variant)
    func = getattr(module, f"sparse_fmha_fwd_{variant}")

    stream = torch.cuda.current_stream().cuda_stream

    lse_temp_partial = (
        plan.LSE_temperature_partial
        if plan.LSE_temperature_partial is not None
        else plan.LSE_partial
    )
    seqused_k_tensor = plan.seqused_k if plan.seqused_k is not None else cu_seqlens_q
    has_seqused_k = 1 if plan.seqused_k is not None else 0

    func(
        Q, K, V,
        k2q_row_ptr, k2q_q_indices, qsplit_indices,
        scheduler_metadata, work_count,
        cu_seqlens_q, cu_seqlens_k,
        kv_indices, kv_page_indptr,
        plan.O_partial, plan.LSE_partial, split_counts,
        plan.sm_scale,
        plan.num_heads_kv,
        plan.qhead_per_kv,
        plan.topk,
        plan.max_seqlen_q,
        plan.is_paged,
        plan.page_size,
        plan.work_capacity,
        plan.dtype_config.dispatch_key,
        stream,
        lse_temp_partial,
        plan.lse_temperature_scale,
        seqused_k_tensor,
        has_seqused_k,
    )


def sparse_fmha_combine(
    *,
    split_counts: torch.Tensor,
    plan: SparseFmhaPlan,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Merge split-K partials into final O (bf16) and LSE (fp32, natural domain).

    Returns (O, LSE, LSE_temperature).
    LSE_temperature is None when temperature is disabled or fast-path (scale=1.0).
    """
    module = get_sparse_fmha_combine_module()
    stream = torch.cuda.current_stream().cuda_stream

    if plan.return_temperature_lse:
        lse_temp_partial = plan.LSE_temperature_partial
        lse_temp_out = plan.LSE_temperature
    else:
        lse_temp_partial = plan.LSE_partial
        lse_temp_out = plan.LSE

    module.sparse_fmha_combine(
        plan.O_partial,
        plan.LSE_partial,
        split_counts,
        plan.O,
        plan.LSE,
        plan.sm_scale,
        plan.head_dim,
        plan.topk,
        plan.total_q,
        plan.num_heads_q,
        plan.num_heads_kv,
        plan.qhead_per_kv,
        stream,
        lse_temp_partial,
        lse_temp_out,
        plan.lse_temperature_scale,
    )

    if plan.return_temperature_lse:
        return plan.O, plan.LSE, plan.LSE_temperature
    elif _is_fast_path_temperature(plan.lse_temperature_scale) and plan.lse_temperature_scale > 0:
        return plan.O, plan.LSE, plan.LSE
    else:
        return plan.O, plan.LSE, None
