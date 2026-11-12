// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include "sparse_fmha_fwd.cuh"
#include "ppu_include.hpp"
#include "fmha_common.hpp"
#include "vec_dtypes.cuh"
#include "cutlass/arch/memory_ppu.h"

#include <type_traits>

namespace minfer {
namespace sparse {

using namespace cute;
using namespace cutlass::fmha::collective;

__device__ __forceinline__ int lower_bound_q_idx(
    const int32_t* q_indices, int head_kv_idx, int row_base,
    int row_count, int q_threshold, int64_t nnz) {
  const int32_t* row = q_indices + (int64_t)head_kv_idx * nnz + row_base;
  const int lane = (int)threadIdx.x & 31;
  int lo = 0;
  int hi = row_count;
  while (hi - lo > 32) {
    const int S = hi - lo;
    const int span = (S + 31) >> 5;
    const unsigned ge = __ballot_sync(
        0xffffffffu, row[lo + min(lane * span, S - 1)] >= q_threshold);
    if (ge == 0) {
      lo = lo + min(31 * span, S - 1) + 1;
    } else {
      const int first = __ffs(ge) - 1;
      const int p_hi = lo + min(first * span, S - 1);
      if (first > 0) {
        lo = lo + min((first - 1) * span, S - 1) + 1;
      }
      hi = p_hi + 1;
    }
  }
  const int idx = lo + lane;
  const int v = (idx < hi) ? row[idx] : 0x7fffffff;
  const unsigned ge = __ballot_sync(0xffffffffu, v >= q_threshold);
  return (ge == 0) ? hi : lo + (__ffs(ge) - 1);
}

__device__ __forceinline__ void decode_edges_to_smem(
    const auto& p, int head_kv_idx, int row_start, int q_begin,
    int q_offset, int batch_begin, int valid_m, int seqlen_q_b,
    int diag_q_count, int32_t* q_abs_smem, int32_t* edge_info_smem,
    int64_t* lse_base_smem) {
  for (int i = threadIdx.x; i < valid_m; i += blockDim.x) {
    int edge = row_start + batch_begin + i;
    int32_t packed = p.qsplit_indices[head_kv_idx * p.nnz + edge];
    int slot = decode_slot(packed);
    int q_local = decode_q_idx_local(packed);
    int abs_pos = q_begin + batch_begin + i;
    if (is_sentinel(packed) || slot >= p.topk || q_local >= seqlen_q_b) {
      q_abs_smem[i] = 0;
      edge_info_smem[i] = 0;
      lse_base_smem[i] = -1;
    } else {
      q_abs_smem[i] = q_offset + q_local;
      lse_base_smem[i] =
          (int64_t)q_abs_smem[i] * p.num_heads_q * p.topk + slot;
      edge_info_smem[i] = 1 | ((abs_pos < diag_q_count) ? 2 : 0)
                        | (q_local << 2);
    }
  }
}

template <typename Element>
constexpr int kBlockKGmem_v = 128 / sizeof(Element);

template <typename Element>
using SmemAtomPlain = Layout<Shape<_8, Int<kBlockKGmem_v<Element>>>,
                              Stride<Int<kBlockKGmem_v<Element>>, _1>>;

template <typename Element, int kTileN, int kHeadDim>
using SmemLayoutKV = decltype(
    tile_to_shape(SmemAtomPlain<Element>{},
                  make_shape(Int<kTileN>{}, Int<kHeadDim>{}, _1{})));

template <typename Element, int kTileN, int kHeadDim>
using SmemLayoutVt = decltype(composition(
    SmemLayoutKV<Element, kTileN, kHeadDim>{},
    make_ordered_layout(make_shape(Int<kHeadDim>{}, Int<kTileN>{}, _1{}),
                        Step<_2, _1, _3>{})));

template <typename Element, int kTileM, int kHeadDim>
using SmemLayoutQ = decltype(
    tile_to_shape(SmemAtomPlain<Element>{},
                  make_shape(Int<kTileM>{}, Int<kHeadDim>{})));

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
template <typename Element, int kTileN, int kHeadDim>
using SmemCopyAtomK = Copy_Atom<
    PPU0015_TSM_LD_SWZL<Element, kTileN, kBlockKGmem_v<Element>, true, false,
                        kHeadDim / kBlockKGmem_v<Element>>,
    Element>;

template <typename Element, int kTileN, int kHeadDim>
using SmemCopyAtomVt = Copy_Atom<
    PPU0015_TSM_LD_SWZL<Element, kTileN, kBlockKGmem_v<Element>, true, true,
                        kHeadDim / kBlockKGmem_v<Element>>,
    Element>;

template <typename Element, int kTileM, int kHeadDim>
using SmemCopyAtomQ = Copy_Atom<
    PPU0015_TSM_LD_SWZL<Element, kTileM, kBlockKGmem_v<Element>, false, false, 1>,
    Element>;
#else
template <typename Element, int kTileN, int kHeadDim>
using SmemCopyAtomK = Copy_Atom<
    PPU0010_TSM_LD_SWZL<Element, kTileN, kBlockKGmem_v<Element>, false, false,
                        kHeadDim / kBlockKGmem_v<Element>>,
    Element>;

template <typename Element, int kTileN, int kHeadDim>
using SmemCopyAtomVt = Copy_Atom<
    PPU0010_TSM_LD_SWZL<Element, kTileN, kBlockKGmem_v<Element>, false, true,
                        kHeadDim / kBlockKGmem_v<Element>>,
    Element>;

template <typename Element, int kTileM, int kHeadDim>
using SmemCopyAtomQ = Copy_Atom<
    PPU0010_TSM_LD_SWZL<Element, kTileM, kBlockKGmem_v<Element>, false, false, 1>,
    Element>;
#endif

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
template <typename Element>
using QK_MMA_Atom = std::conditional_t<
    std::is_same_v<Element, cutlass::float_e4m3_t>,
    MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
    MMA_Atom<PPU0015_16x16x16_F32BF16BF16F32_TN>>;

template <typename Element>
using PV_MMA_Atom = std::conditional_t<
    std::is_same_v<Element, cutlass::float_e4m3_t>,
    MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
    MMA_Atom<PPU0015_16x16x16_F32BF16BF16F32_TN>>;

template <typename Element>
constexpr int kMmaInstK_v = std::is_same_v<Element, cutlass::float_e4m3_t> ? 32 : 16;
#else
template <typename Element>
using QK_MMA_Atom = std::conditional_t<
    std::is_same_v<Element, cutlass::half_t>,
    MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
    MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>>;

template <typename Element>
using PV_MMA_Atom = std::conditional_t<
    std::is_same_v<Element, cutlass::half_t>,
    MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
    MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>>;

template <typename Element>
constexpr int kMmaInstK_v = 16;
#endif

template <typename Element, int kNWarps>
using TiledMmaQK = TiledMMA<
    QK_MMA_Atom<Element>,
    Layout<Shape<Int<kNWarps>, _1, _1>>,
    Tile<Int<16 * kNWarps>, _16, Int<kMmaInstK_v<Element>>>>;

template <typename Element, int kNWarps>
using TiledMmaPV = TiledMMA<
    PV_MMA_Atom<Element>,
    Layout<Shape<Int<kNWarps>, _1, _1>>,
    Tile<Int<16 * kNWarps>, _16, Int<kMmaInstK_v<Element>>>>;

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
template <typename Element, int kNWarps>
using TiledMmaO = std::conditional_t<
    std::is_same_v<Element, cutlass::float_e4m3_t>,
    TiledMMA<MMA_Atom<cute::MSASparseFP8EpiPerm>,
             Layout<Shape<Int<kNWarps>, _1, _1>>,
             Tile<Int<16 * kNWarps>, _16, _32>>,
    TiledMmaPV<Element, kNWarps>>;
#else
template <typename Element, int kNWarps>
using TiledMmaO = TiledMmaPV<Element, kNWarps>;
#endif

__device__ __forceinline__ void store_partial_pair(ppu_bfloat16* dst, float a, float b) {
  *reinterpret_cast<__ppu_bfloat162*>(dst) = __float22bfloat162_rn(make_float2(a, b));
}

__device__ __forceinline__ void store_partial_one(ppu_bfloat16* dst, float a) {
  *dst = __float2bfloat16_rn(a);
}

template <int kQpk, int kTileM>
__device__ __forceinline__ int mtile_row_to_edge(int m, int it) {
  if constexpr (kQpk == 1) {
    (void)it;
    return m;
  } else {
    return it * (kTileM / kQpk) + m / kQpk;
  }
}

template <int kQpk>
__device__ __forceinline__ int mtile_row_to_head(int m, int it, int head_kv_idx,
                                                 int qhead_per_kv) {
  if constexpr (kQpk == 1) {
    (void)m;
    return head_kv_idx * qhead_per_kv + it;
  } else {
    (void)it;
    (void)qhead_per_kv;
    return head_kv_idx * kQpk + m % kQpk;
  }
}

template <int kQpk, int kTileM, bool kEpiHoisted, class Params,
          class TensorOrO, class TensorOcO>
__device__ __forceinline__ void store_o_partial(
    const Params& p, int head_kv_idx, int it,
    [[maybe_unused]] int valid_edges, TensorOrO const& tOrO_rc,
    TensorOcO const& tOcO_rc,
    [[maybe_unused]] const int64_t* lse_base_smem, const float* lse_row_sum,
    const float* lse_temp, bool return_temperature_lse,
    [[maybe_unused]] const int64_t* lse_base_reg,
    [[maybe_unused]] const bool* row_ok_reg) {
  nv_bfloat16* dst_base = p.ptr_O_partial;
  const int head_dim = p.head_dim;
  const bool lse_lane = (threadIdx.x & 3) == 0;
  CUTE_UNROLL
  for (int mi = 0; mi < size<0>(tOrO_rc); ++mi) {
    int m = get<0>(tOcO_rc(mi, 0));
    int64_t lse_base;
    if constexpr (kEpiHoisted) {
      if (!row_ok_reg[mi]) continue;
      lse_base = lse_base_reg[mi];
    } else {
      int e = mtile_row_to_edge<kQpk, kTileM>(m, it);
      lse_base = lse_base_smem[e];
      if (e >= valid_edges || lse_base < 0) continue;
    }
    int head_q_idx =
        mtile_row_to_head<kQpk>(m, it, head_kv_idx, p.qhead_per_kv);
    int64_t lse_off = lse_base + (int64_t)head_q_idx * p.topk;
    int64_t row_off = lse_off * head_dim;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
    CUTE_UNROLL
    for (int ni = 0; ni < size<1>(tOrO_rc); ni += 2) {
      int d = get<1>(tOcO_rc(mi, ni));
      store_partial_pair(dst_base + row_off + d, tOrO_rc(mi, ni),
                         tOrO_rc(mi, ni + 1));
    }
#else
    CUTE_UNROLL
    for (int ni = 0; ni < size<1>(tOrO_rc); ++ni) {
      int d = get<1>(tOcO_rc(mi, ni));
      store_partial_one(dst_base + row_off + d, tOrO_rc(mi, ni));
    }
#endif
    if (lse_lane) {
      p.ptr_LSE_partial[lse_off] = lse_row_sum[mi];
      if (return_temperature_lse) {
        p.ptr_LSE_temperature_partial[lse_off] = lse_temp[mi];
      }
    }
  }
}

template <typename Element, int kHeadDim, int kTileM, int kNWarps>
__device__ __forceinline__ void issue_q_gather(
    const SparseFwdParams<Element>& p, int head_q_idx,
    int q_offset, int valid_m, const int32_t* edge_info_smem,
    [[maybe_unused]] const int32_t* q_abs_smem, Element* Q_smem) {
  static_assert(kHeadDim == 128,
                "issue_q_gather swizzle formula assumes head_dim=128");
  constexpr int kBlockKGmem = kBlockKGmem_v<Element>;
  constexpr int kVecBytes = 16;
  constexpr int kVecElems = kVecBytes / sizeof(Element);
  constexpr int kChunksPerRow = kHeadDim * sizeof(Element) / kVecBytes;
  constexpr int kCubeStride = kTileM * kBlockKGmem;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  for (int chunk = threadIdx.x; chunk < kTileM * kChunksPerRow;
       chunk += blockDim.x) {
    int m = chunk / kChunksPerRow;
    int c = (chunk % kChunksPerRow) * kVecElems;
    bool row_valid = (m < valid_m) && (edge_info_smem[m] & 1);
    int64_t q_abs = row_valid ? q_abs_smem[m] : 0;
    const Element* q_src =
        p.ptr_Q + (int64_t)q_abs * p.q_stride_n +
        (int64_t)head_q_idx * p.q_stride_h + c;
    int blk = ((c % kBlockKGmem) / kVecElems) ^ (m % 8);
    int dst = (c / kBlockKGmem) * kCubeStride + m * kBlockKGmem +
              blk * kVecElems + (c % kVecElems);
    cutlass::arch::cp_async_zfill<kVecBytes, cutlass::arch::CacheOperation::Global>(
        Q_smem + dst, q_src, row_valid);
  }
#else
  // Equivalent to the chunk loop above, but re-indexed by `line = m / 4` so that
  // every term of the swizzle formula except `line * (4 * 16)` is hoisted out of
  // the loop. Requires blockDim.x == kNWarps * 32 (as set by launch_fwd_variant).
  constexpr int kBlockSize = kNWarps * 32;
  constexpr int kRowsPerStep = kBlockSize / kChunksPerRow;
  constexpr int kLinesPerStep = kRowsPerStep / 4;
  static_assert(kBlockSize % kChunksPerRow == 0,
                "issue_q_gather hoist needs blockDim.x divisible by kChunksPerRow");
  static_assert(kRowsPerStep % 8 == 0,
                "issue_q_gather hoist needs m%4 and (m/4)&1 loop-invariant");
  static_assert(kTileM % 4 == 0, "issue_q_gather hoist needs kTileM divisible by 4");

  const int m0 = threadIdx.x / kChunksPerRow;
  const int c = (threadIdx.x % kChunksPerRow) * kVecElems;
  const int cc = c % kBlockKGmem;
  const int s = cc / 16;
  const int half = (cc % 16) / kVecElems;
  const int vec = (((m0 % 4) * 2) + half) ^ ((m0 / 4) & 1);
  const int slice_start = (((s & 1) << 1) + ((s & 2) >> 1)) * 2;
  const int vec_rot = (vec + slice_start) % 8;
  const int dst_row0 = (c / kBlockKGmem) * kCubeStride + s * (kTileM * 16) +
                       vec_rot * kVecElems + (c % kVecElems);
  const int line0 = m0 / 4;
  constexpr int kIters = kTileM / kRowsPerStep;
  static_assert(kTileM % kRowsPerStep == 0,
                "issue_q_gather unroll needs kTileM divisible by kRowsPerStep");
  // decode_edges_to_smem writes q_abs_smem[m] == q_offset + q_local alongside
  // edge_info_smem[m] == 1 | (causal << 1) | (q_local << 2), so q_local can be
  // recovered from the edge_info word the validity test already needs. Folding
  // q_offset into the base pointer drops the second smem round trip, which used
  // to be serialised behind the first one on the cp_async address chain.
  const Element* q_col =
      p.ptr_Q + (int64_t)q_offset * p.q_stride_n +
      (int64_t)head_q_idx * p.q_stride_h + c;
  const int dst_base = dst_row0 + line0 * (4 * 16);

  // line == line0 + i * kLinesPerStep and m == line * 4 + m0 % 4 collapse to
  // m == m0 + kRowsPerStep * i, so the kIters reads are mutually independent
  // and can be issued together: one wait covers the whole gather instead of one
  // per copy.
  int edge[kIters];
  CUTE_UNROLL
  for (int i = 0; i < kIters; ++i) {
    const int m = m0 + kRowsPerStep * i;
    const int w = edge_info_smem[m];
    edge[i] = (m < valid_m) ? w : 0;
  }

  CUTE_UNROLL
  for (int i = 0; i < kIters; ++i) {
    const int w = edge[i];
    const bool row_valid = (w & 1);
    const int64_t q_local = row_valid ? (w >> 2) : 0;
    cutlass::arch::cp_async_zfill<kVecBytes, cutlass::arch::CacheOperation::Global>(
        Q_smem + dst_base + i * (kLinesPerStep * 4 * 16),
        q_col + q_local * p.q_stride_n, row_valid);
  }
#endif
}

template <typename Element, int kHeadDim, int kTileM, int kNWarps, int kQpk>
__device__ __forceinline__ void issue_q_pack_aiu(
    const SparseFwdParams<Element>& p, int head_kv_idx,
    int edge_base, int valid_edges,
    const int32_t* edge_info_smem, const int32_t* q_abs_smem,
    Element* Q_smem) {
  constexpr int kBlockKGmem = kBlockKGmem_v<Element>;
  constexpr int kEdgesPerTile = kTileM / kQpk;
  constexpr int kEdgesPerWarp = kEdgesPerTile / kNWarps;
  constexpr int kCubeStride = kTileM * kBlockKGmem;
  constexpr int kBitsPerAiuQEdge = kQpk * kBlockKGmem * sizeof(Element) * 8;
  using GmemTiledCopyQEdge = decltype(make_tiled_copy(
      Copy_Atom<PPU0015_AIU_LOAD<C<kBitsPerAiuQEdge>, Element, false, kQpk,
                                kBlockKGmem>,
                Element>{},
      Layout<Shape<_1, _1>, Stride<_1, _1>>{},
      Layout<Shape<Int<kQpk>, Int<kBlockKGmem>>>{}));
  constexpr int kQColGroups = kHeadDim / kBlockKGmem;
  using EdgeLayoutQ = decltype(make_layout(
      make_shape(Int<kQpk>{}, make_shape(Int<kBlockKGmem>{}, Int<kQColGroups>{})),
      make_stride(Int<kBlockKGmem>{}, make_stride(_1{}, Int<kCubeStride>{}))));

  const int warp = threadIdx.x / 32;
  const int j0 = warp * kEdgesPerWarp;
  for (int j = j0; j < j0 + kEdgesPerWarp; ++j) {
    int e = edge_base + j;
    if (e < valid_edges && (edge_info_smem[e] & 1)) {
      const Element* src = p.ptr_Q +
          (int64_t)q_abs_smem[e] * p.q_stride_n +
          (int64_t)head_kv_idx * kQpk * p.q_stride_h;
      Tensor gQ_e = make_tensor(
          make_gmem_ptr(src),
          Shape<Int<kQpk>, Int<kHeadDim>>{},
          make_stride(static_cast<int>(p.q_stride_h), _1{}));
      Tensor gQm_e = make_mix_tensor_like(gQ_e);
      GmemTiledCopyQEdge c;
      c.desc_.init(nullptr, kQpk, kHeadDim,
                   static_cast<int>(p.q_stride_h));
      auto thr_copy = c.get_thread_slice(0);
      Tensor tQgQ = thr_copy.partition_S(gQm_e);
      Tensor sQ_e =
          make_tensor(make_smem_ptr(Q_smem + j * kQpk * kBlockKGmem), EdgeLayoutQ{});
      Tensor tQsQ = thr_copy.partition_D(sQ_e);
      cute::copy(c, tQgQ, tQsQ);
    }
  }
  cutlass::arch::cp_async_fence();
}

template <typename Element, int kHeadDim, int kTileM, int kNWarps, int kQpk>
__device__ __forceinline__ void gather_q_mtile(
    const SparseFwdParams<Element>& p, int head_kv_idx,
    [[maybe_unused]] int q_offset, int it, int valid_edges,
    const int32_t* edge_info_smem, const int32_t* q_abs_smem, Element* Q_smem) {
  if constexpr (kQpk > 1) {
    issue_q_pack_aiu<Element, kHeadDim, kTileM, kNWarps, kQpk>(
        p, head_kv_idx, it * (kTileM / kQpk), valid_edges, edge_info_smem,
        q_abs_smem, Q_smem);
  } else {
    issue_q_gather<Element, kHeadDim, kTileM, kNWarps>(
        p, head_kv_idx * p.qhead_per_kv + it, q_offset, valid_edges,
        edge_info_smem, q_abs_smem, Q_smem);
    cutlass::arch::cp_async_fence();
  }
}

template <typename Element, class SmemLayoutKV_,
          class GmemTiledCopyKV, int kHeadDim, int kTileN>
__device__ __forceinline__ void load_kv_aiu(
    const Element* gmem_base, int kv_row_stride, int valid_n,
    Element* smem_base) {
  Tensor sT = make_tensor(make_smem_ptr(smem_base), SmemLayoutKV_{});
  Tensor mT = make_tensor(
      make_gmem_ptr(gmem_base),
      Shape<Int<kTileN>, Int<kHeadDim>>{}, make_stride(kv_row_stride, _1{}));
  Tensor gT = make_mix_tensor_like(mT);
  GmemTiledCopyKV c;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  c.desc_.init(nullptr, valid_n, kHeadDim, kv_row_stride);
#else
  c.desc_ = AiuDesc{nullptr, valid_n, kv_row_stride, kTileN,
                    kBlockKGmem_v<Element>, 0};
#endif
  auto thr_copy = c.get_thread_slice(0);
  Tensor tKgK = thr_copy.partition_S(gT);
  Tensor tKsK = thr_copy.partition_D(sT(_, _, _0{}));
  if ((threadIdx.x / 32) == 0) cute::copy(c, tKgK, tKsK);
}

#ifndef SPARSE_FWD_MIN_CTA_PER_CU
#define SPARSE_FWD_MIN_CTA_PER_CU 1
#endif
#if SPARSE_FWD_MIN_CTA_PER_CU > 1
#define SPARSE_FWD_LAUNCH_BOUNDS \
  __launch_bounds__(kNWarps * 32, SPARSE_FWD_MIN_CTA_PER_CU)
#else
#define SPARSE_FWD_LAUNCH_BOUNDS __launch_bounds__(kNWarps * 32)
#endif

template <int kHeadDim, int kTileM, int kTileN, int kNWarps, int kQpk,
          typename Element>
__global__ void SPARSE_FWD_LAUNCH_BOUNDS
sparse_fmha_fwd_kernel(SparseFwdParams<Element> p) {
  static_assert(kHeadDim == 128 && kTileN == 128 &&
                    (kTileM == 64 || kTileM == 128),
                "smem layout constants are specialized for head_dim=128");
  static_assert(kQpk == 1 ||
                    (kQpk >= 8 && kQpk <= kTileM && (kQpk & (kQpk - 1)) == 0),
                "packed M-dim path requires power-of-2 qpk in [8, kTileM]");
  static_assert(kQpk == 1 || (kTileM / kQpk) % kNWarps == 0,
                "per-edge pipeline maps edges to whole warps");
  constexpr bool IsFP8 = std::is_same_v<Element, cutlass::float_e4m3_t>;
  static_assert(sizeof(Element) != 1 || IsFP8,
                "Sparse FP8 supports only E4M3FN");
#if defined(__HGGC_ARCH__)
  static_assert(!IsFP8 || __HGGC_ARCH__ == 150,
                "Sparse FP8 supports only PPU0015");
#endif
#if !(defined __HGGC_ARCH__) || (__HGGC_ARCH__ != 150)
  static_assert(kQpk == 1, "PPU0010 sparse fwd supports only kQpk == 1");
  static_assert(std::is_same_v<Element, cutlass::bfloat16_t>,
                "PPU0010 sparse fwd supports only bf16");
#endif

  constexpr bool kPacked = (kQpk > 1);
  constexpr int kEdgesPerMTile = kTileM / kQpk;

  int block_idx = blockIdx.x;
  if (block_idx >= p.work_count[0]) return;

  const int32_t* meta = p.scheduler_metadata + (int64_t)block_idx * 6;
  int head_kv_idx = meta[0];
  int row_linear = meta[1];
  int q_begin = meta[2];
  int q_count = meta[3];
  int batch_idx = meta[4];
  int kv_block_idx = meta[5];

  int tidx = threadIdx.x;

  int seqlen_q_b = p.cu_seqlens_q[batch_idx + 1] - p.cu_seqlens_q[batch_idx];
  int seqlen_k_b = p.cu_seqlens_k[batch_idx + 1] - p.cu_seqlens_k[batch_idx];
  int q_offset = p.cu_seqlens_q[batch_idx];
  int logical_len_k = (p.seqused_k != nullptr) ? p.seqused_k[batch_idx] : seqlen_k_b;
  int causal_delta = logical_len_k - seqlen_q_b;
  int kv_block_base = kv_block_idx * p.blk_kv;
  int kv_boundary = max(0, logical_len_k - kv_block_base);

  int kv_valid_cols = min(kv_boundary, p.blk_kv);
  int kv_valid_end = kv_block_idx * p.blk_kv + kv_valid_cols;
  int q_threshold = kv_valid_end - causal_delta;

  int row_base =
      p.k2q_row_ptr[head_kv_idx * (p.total_rows + 1) + row_linear];
  int full_row_count =
      p.k2q_row_ptr[head_kv_idx * (p.total_rows + 1) + row_linear + 1] -
      row_base;
  int diag_q_count = lower_bound_q_idx(p.k2q_q_indices, head_kv_idx, row_base,
                                       full_row_count, q_threshold, p.nnz);

  using Budget = SparseFwdSmemBudget<Element, kTileM, kTileN, kHeadDim>;

  extern __shared__ __align__(128) char smem_buf[];
  Element* K_smem = reinterpret_cast<Element*>(smem_buf);
  Element* V_smem = reinterpret_cast<Element*>(smem_buf + Budget::kKBytes);
  Element* Q_smem = reinterpret_cast<Element*>(
      smem_buf + Budget::kKBytes + Budget::kVBytes);
  int32_t* edge_smem = reinterpret_cast<int32_t*>(
      smem_buf + Budget::kKBytes + Budget::kVBytes + Budget::kQBytes);
  int32_t* q_abs_smem = edge_smem;
  int32_t* edge_info_smem = edge_smem + kTileM;
  int64_t* lse_base_smem = reinterpret_cast<int64_t*>(edge_smem + 2 * kTileM);

  int valid_n = min(kTileN, kv_boundary);
  const Element* k_row_base;
  const Element* v_row_base;
  int kv_row_stride;
  if (p.is_paged) {
    int phys = p.kv_indices[p.kv_page_indptr[batch_idx] + kv_block_idx];
    int64_t page_off;
    if (p.kv_is_nhd) {
      page_off = (int64_t)phys * p.page_size * p.num_heads_kv * kHeadDim
               + (int64_t)head_kv_idx * kHeadDim;
      kv_row_stride = p.num_heads_kv * kHeadDim;
    } else {
      page_off = (int64_t)phys * p.num_heads_kv * p.page_size * kHeadDim
               + (int64_t)head_kv_idx * p.page_size * kHeadDim;
      kv_row_stride = kHeadDim;
    }
    k_row_base = p.ptr_K + page_off;
    v_row_base = p.ptr_V + page_off;
  } else {
    int64_t base_off =
        (int64_t)(p.cu_seqlens_k[batch_idx] + kv_block_idx * p.blk_kv) *
            p.num_heads_kv * kHeadDim +
        (int64_t)head_kv_idx * kHeadDim;
    k_row_base = p.ptr_K + base_off;
    v_row_base = p.ptr_V + base_off;
    kv_row_stride = p.num_heads_kv * kHeadDim;
  }

  constexpr int kBlockKGmemStore = kBlockKGmem_v<Element>;
  constexpr int kBitsPerAiuKV = kTileN * kBlockKGmemStore * sizeof(Element) * 8;
  using SmemLayoutKV_ = SmemLayoutKV<Element, kTileN, kHeadDim>;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  using GmemTiledCopyKV = decltype(make_tiled_copy(
      Copy_Atom<PPU0015_AIU_LOAD<C<kBitsPerAiuKV>, Element, false,
                                kTileN, kBlockKGmemStore>,
                Element>{},
      Layout<Shape<_1, _1>, Stride<_1, _1>>{},
      Layout<Shape<Int<kTileN>, Int<kBlockKGmemStore>>>{}));
#else
  using GmemTiledCopyKV = decltype(make_tiled_copy(
      Copy_Atom<PPU0010_AIU_LOAD<C<kBitsPerAiuKV>, Element, false>,
                Element>{},
      Layout<Shape<_1, _1>, Stride<_1, _1>>{},
      Layout<Shape<Int<kTileN>, Int<kBlockKGmemStore>>>{}));
#endif

  int row_start =
      p.k2q_row_ptr[head_kv_idx * (p.total_rows + 1) + row_linear] + q_begin;
  int first_valid_m = min(kTileM, q_count);

  decode_edges_to_smem(p, head_kv_idx, row_start, q_begin, q_offset, 0,
                       first_valid_m, seqlen_q_b, diag_q_count,
                       q_abs_smem,
                       edge_info_smem, lse_base_smem);
  __syncthreads();
  if constexpr (kQpk > 1) {
    load_kv_aiu<Element, SmemLayoutKV_, GmemTiledCopyKV, kHeadDim, kTileN>(
        k_row_base, kv_row_stride, valid_n, K_smem);
    cutlass::arch::cp_async_fence();
    load_kv_aiu<Element, SmemLayoutKV_, GmemTiledCopyKV, kHeadDim, kTileN>(
        v_row_base, kv_row_stride, valid_n, V_smem);
    cutlass::arch::cp_async_fence();
    gather_q_mtile<Element, kHeadDim, kTileM, kNWarps, kQpk>(
        p, head_kv_idx, q_offset, 0, first_valid_m, edge_info_smem,
        q_abs_smem, Q_smem);
  } else {
    gather_q_mtile<Element, kHeadDim, kTileM, kNWarps, kQpk>(
        p, head_kv_idx, q_offset, 0, first_valid_m, edge_info_smem,
        q_abs_smem, Q_smem);
    load_kv_aiu<Element, SmemLayoutKV_, GmemTiledCopyKV, kHeadDim, kTileN>(
        k_row_base, kv_row_stride, valid_n, K_smem);
    cutlass::arch::cp_async_fence();
    load_kv_aiu<Element, SmemLayoutKV_, GmemTiledCopyKV, kHeadDim, kTileN>(
        v_row_base, kv_row_stride, valid_n, V_smem);
    cutlass::arch::cp_async_fence();
  }

  using TiledMmaQK_t = TiledMmaQK<Element, kNWarps>;
  using TiledMmaPV_t = TiledMmaPV<Element, kNWarps>;

  TiledMmaQK_t tiled_mma_qk;
  TiledMmaPV_t tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_slice(tidx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(tidx);

  Tensor sQ = make_tensor(make_smem_ptr(Q_smem), SmemLayoutQ<Element, kTileM, kHeadDim>{});
  Tensor sK = make_tensor(make_smem_ptr(K_smem), SmemLayoutKV<Element, kTileN, kHeadDim>{});
  Tensor sVt = make_tensor(make_smem_ptr(V_smem), SmemLayoutVt<Element, kTileN, kHeadDim>{});

  Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
  Tensor tOrO = partition_fragment_C(tiled_mma_pv, Shape<Int<kTileM>, Int<kHeadDim>>{});
  Tensor tSrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kTileM>, Int<kTileN>>{});
  Tensor tSrK = thr_mma_qk.partition_fragment_B(sK(_, _, _0{}));

  Tensor cS = make_identity_tensor(Shape<Int<kTileM>, Int<kTileN>>{});
  Tensor tScS = thr_mma_qk.partition_C(cS);
  Tensor cO = make_identity_tensor(Shape<Int<kTileM>, Int<kHeadDim>>{});
  using TiledMmaO_t = TiledMmaO<Element, kNWarps>;
  TiledMmaO_t tiled_mma_o;
  auto thr_mma_o = tiled_mma_o.get_slice(tidx);
  Tensor tOcO = thr_mma_o.partition_C(cO);

  const int tid_slice = (tidx / 32) * 32;
  auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ<Element, kTileM, kHeadDim>{}, tiled_mma_qk);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid_slice);
  auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomK<Element, kTileN, kHeadDim>{}, tiled_mma_qk);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid_slice);

  Tensor tSsQ = smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ));
  Tensor tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sK));

  Tensor tSrQ_view = smem_thr_copy_Q.retile_D(tSrQ);
  Tensor tSrS_rc = make_tensor(tSrS.data(), convert_layout_acc_rowcol(tSrS.layout()));
  Tensor tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
  Tensor tOrO_rc = make_tensor(tOrO.data(), convert_layout_acc_rowcol(tOrO.layout()));
  Tensor tOcO_rc = make_tensor(tOcO.data(), convert_layout_acc_rowcol(tOcO.layout()));

  auto nohook = []() {};

  [[maybe_unused]] bool kv_wait_done = false;

  for (int batch_begin = 0; batch_begin < q_count; batch_begin += kTileM) {
    int valid_edges = min(kTileM, q_count - batch_begin);

    if (batch_begin > 0) {
      decode_edges_to_smem(p, head_kv_idx, row_start, q_begin, q_offset,
                           batch_begin, valid_edges, seqlen_q_b, diag_q_count,
                           q_abs_smem,
                           edge_info_smem, lse_base_smem);
      __syncthreads();
    }

    int num_iters;
    if constexpr (kPacked) {
      num_iters = (valid_edges + kEdgesPerMTile - 1) / kEdgesPerMTile;
    } else {
      num_iters = p.qhead_per_kv;
    }

    // The __syncthreads() calls inside the it loop are smem barriers, so the
    // compiler cannot hoist this edge_info_smem read itself. On the kQpk == 1
    // path it is loop-invariant: mtile_row_to_edge<1, kTileM>(m, it) == m, and
    // valid_edges is fixed for the whole batch_begin tile. Hoisting only the
    // load (not the mask arithmetic, and not any of the in-loop declarations)
    // keeps the PPU0015 source textually unchanged.
#if !((defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150))
    int edge_w[2];
    if constexpr (kQpk == 1) {
      CUTE_UNROLL
      for (int mi = 0; mi < size<0>(tScS_rc); ++mi) {
        int m = get<0>(tScS_rc(mi, 0));
        edge_w[mi] = (m < valid_edges) ? edge_info_smem[m] : 0;
      }
    }
#endif

    // Same hoist applied to the epilogue. lse_base_smem[] is written only by
    // decode_edges_to_smem(), which always runs before this loop, and both
    // store_o_partial() and gather_q_mtile() take const pointers, so the read
    // is loop-invariant on the kQpk == 1 path for the same reason as above:
    // mtile_row_to_edge<1, kTileM>(m, it) == m. The validity predicate
    // (e >= valid_edges || lse_base < 0) depends only on that read and on
    // valid_edges, so it folds into the same hoist. Coordinates come from
    // tOcO_rc -- the exact tensor store_o_partial() indexes -- so this makes no
    // assumption about TiledMmaO matching TiledMmaQK.
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
    constexpr bool kEpiHoisted = false;
#else
    constexpr bool kEpiHoisted = (kQpk == 1);
#endif
    int64_t lse_base_w[2];
    bool row_store_w[2];
    if constexpr (kEpiHoisted) {
      CUTE_UNROLL
      for (int mi = 0; mi < size<0>(tOcO_rc); ++mi) {
        int m = get<0>(tOcO_rc(mi, 0));
        int64_t lb = (m < valid_edges) ? lse_base_smem[m] : (int64_t)-1;
        lse_base_w[mi] = lb;
        row_store_w[mi] = (m < valid_edges) && (lb >= 0);
      }
    }

    for (int it = 0; it < num_iters; ++it) {
      const bool q0_inline = (it == 0) && (batch_begin > 0);
      if constexpr (kQpk > 1) {
        if (q0_inline) {
          gather_q_mtile<Element, kHeadDim, kTileM, kNWarps, kQpk>(
              p, head_kv_idx, q_offset, 0, valid_edges, edge_info_smem,
              q_abs_smem, Q_smem);
        }
        cutlass::arch::cp_async_wait<0>();
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_view);
        if (it == 0 && batch_begin == 0) {
          __syncthreads();
        }
      } else {
        if (q0_inline) {
          gather_q_mtile<Element, kHeadDim, kTileM, kNWarps, kQpk>(
              p, head_kv_idx, q_offset, 0, valid_edges, edge_info_smem,
              q_abs_smem, Q_smem);
          cutlass::arch::cp_async_wait<0>();
        } else if (it == 0) {
          cutlass::arch::cp_async_wait<2>();
        } else {
          cutlass::arch::cp_async_wait<0>();
        }
        __syncthreads();

        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_view);
        __syncthreads();
      }

      if (it + 1 < num_iters) {
        gather_q_mtile<Element, kHeadDim, kTileM, kNWarps, kQpk>(
            p, head_kv_idx, q_offset, it + 1, valid_edges, edge_info_smem,
            q_abs_smem, Q_smem);
      }

      if constexpr (kQpk == 1) {
        if (!kv_wait_done) {
          if (it + 1 < num_iters) {
            cutlass::arch::cp_async_wait<2>();
          } else {
            cutlass::arch::cp_async_wait<1>();
          }
          __syncthreads();
        }
      }

      clear(tSrS);
      gemm_sm80<true>(tSrS, tSrQ, tSrK, tSsQ, tSsK(_, _, _, _0{}),
                      tiled_mma_qk, smem_tiled_copy_Q, smem_tiled_copy_K,
                      smem_thr_copy_Q, smem_thr_copy_K, nohook);

      bool row_ok[2];
      bool row_causal[2];
      int row_diag[2];
      CUTE_UNROLL
      for (int mi = 0; mi < size<0>(tSrS_rc); ++mi) {
        int m = get<0>(tScS_rc(mi, 0));
        int e = mtile_row_to_edge<kQpk, kTileM>(m, it);
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
        int w = (e < valid_edges) ? edge_info_smem[e] : 0;
#else
        int w = (kQpk == 1) ? edge_w[mi]
                            : ((e < valid_edges) ? edge_info_smem[e] : 0);
#endif
        row_ok[mi] = w & 1;
        row_causal[mi] = row_ok[mi] && (w & 2);
        row_diag[mi] = (w >> 2) + causal_delta - kv_block_base;
      }
      bool need_mask = (valid_n != kTileN);
      CUTE_UNROLL
      for (int mi = 0; mi < size<0>(tSrS_rc); ++mi) {
        need_mask = need_mask || !row_ok[mi] ||
                    (row_causal[mi] && row_diag[mi] < kTileN - 1);
      }
      if (need_mask) {
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(tSrS_rc); ++mi) {
          int n_max = row_ok[mi] ? valid_n : 0;
          int mask_from = row_causal[mi] ? min(row_diag[mi] + 1, n_max) : n_max;
          CUTE_UNROLL
          for (int ni = 0; ni < size<1>(tSrS_rc); ++ni) {
            int n = get<1>(tScS_rc(mi, ni));
            if (n >= mask_from) {
              tSrS_rc(mi, ni) = -INFINITY;
            }
          }
        }
      }

      Softmax<2> softmax(p.scale_softmax_log2);
      softmax.template max_get_scale<true, true>(tSrS);

      float lse_temp[2];
      if (p.return_temperature_lse) {
        SumOp<float> sum_op;
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(tSrS_rc); ++mi) {
          float tmax = softmax.row_max(mi);
          float sum_temp = 0.f;
          if (tmax > -INFINITY) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(tSrS_rc); ++ni) {
              float v = tSrS_rc(mi, ni);
              if (v != -INFINITY) {
                sum_temp += exp2f((v - tmax) * p.lse_temperature_scale_log2);
              }
            }
          }
          sum_temp = Allreduce<4>::run(sum_temp, sum_op);
          lse_temp[mi] = (sum_temp == 0.f)
              ? -INFINITY
              : tmax + log2f(sum_temp) / p.lse_temperature_scale_log2;
        }
      }

      softmax.template online_softmax<true, true>(tSrS);

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
      if constexpr (IsFP8) { permute_Cregs_fp8(tSrS); }
      Tensor tOrP_acc = [&]() {
        if constexpr (IsFP8) {
          return make_tensor(
              tSrS.data(),
              convert_layout_acc_AregsFP8<TiledMmaPV_t>(tSrS.layout()));
        } else {
          return make_tensor(
              tSrS.data(),
              convert_layout_acc_Aregs<TiledMmaPV_t>(tSrS.layout()));
        }
      }();
      Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
      convert_type_out(tOrP_acc, tOrP);
#else
      Tensor tOrP_acc = convert_acc<Element>(tSrS);
      Tensor tOrP = make_tensor(
          tOrP_acc.data(),
          make_layout(get<0>(tSrQ.layout()), get<1>(tSrS.layout()),
                      get<2>(tSrS.layout())));
#endif

      if constexpr (kQpk == 1) {
        if (!kv_wait_done) {
          if (it + 1 < num_iters) {
            cutlass::arch::cp_async_wait<1>();
          } else {
            cutlass::arch::cp_async_wait<0>();
          }
          __syncthreads();
          kv_wait_done = true;
        }
      }

      clear(tOrO);
      Tensor tOrV = thr_mma_pv.partition_fragment_B(sVt(_, _, _0{}));
      if constexpr (IsFP8) {
        gemm_rs_sm80_pv_fp8_vdirect<kTileN, kHeadDim,
                                    kBlockKGmem_v<Element> * sizeof(Element),
                                    true>(tOrO, tOrP, tOrV,
                                          static_cast<void*>(V_smem),
                                          tiled_mma_pv);
      } else {
        auto smem_tiled_copy_V = make_tiled_copy_B(
            SmemCopyAtomVt<Element, kTileN, kHeadDim>{}, tiled_mma_pv);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid_slice);
        Tensor tOsVt = smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt));
        gemm_rs_sm80(tOrO, tOrP, tOrV, tOsVt(_, _, _, _0{}),
                     tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V);
      }

      Tensor final_scale = softmax.finalize();
      softmax.rescale_o(tOrO, final_scale);
      if constexpr (IsFP8) { permute_output_fp8(tOrO); }

      float lse_row_sum[2];
      CUTE_UNROLL
      for (int mi = 0; mi < size<0>(tOrO_rc); ++mi) {
        lse_row_sum[mi] = softmax.row_sum(mi);
      }
      store_o_partial<kQpk, kTileM, kEpiHoisted>(
          p, head_kv_idx, it, valid_edges, tOrO_rc, tOcO_rc,
          lse_base_smem, lse_row_sum, lse_temp, p.return_temperature_lse,
          lse_base_w, row_store_w);
      if constexpr (kQpk > 1) {
        if (it + 1 == num_iters) {
          __syncthreads();
        }
      } else {
        __syncthreads();
      }
    }
  }
}

}
}
