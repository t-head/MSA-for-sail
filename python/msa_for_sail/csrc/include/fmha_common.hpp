/***************************************************************************************************
 * Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 *
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#ifndef USE_PPU
#include "cutlass/arch/reg_reconfig.h"
#endif
#include "cutlass/kernel_hardware_info.h"
#ifdef USE_PPU
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "acc_vreg_fraga.h"
#include "cute/arch/copy_ppu0015_aiu.hpp"
#include "cute/atom/mma_traits_ppu0015.hpp"


namespace cute {
struct MSASparseFP8EpiPerm : PPU0015_16x16x32_F32E4M3E4M3F32_TN {};
template <>
struct MMA_Traits<MSASparseFP8EpiPerm>
    : MMA_Traits<PPU0015_16x16x32_F32E4M3E4M3F32_TN> {
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                         Stride<Stride<_64, _1>, Stride<_16, _8, _32>>>;
};
}  // namespace cute
#endif

namespace cutlass::fmha::collective {

using namespace cute;

#ifdef USE_PPU
template <typename MMA, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_AregsFP8(Layout0 acc_layout) {
  static_assert(std::is_same_v<typename MMA::ValTypeA, cutlass::float_e4m3_t>);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
  static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);
  auto l = logical_divide(select<0, 2>(acc_layout), Layout<Shape<_4, _2, _2>>{});
  return make_layout(get<0>(l), get<1>(acc_layout), get<1>(l));
}

template <typename Fragment>
CUTLASS_DEVICE void permute_Cregs_fp8(Fragment& frag) {
  static_assert(decltype(size<0, 0>(frag))::value == 2);
  static_assert(decltype(size<0, 1>(frag))::value == 2);
  static_assert(decltype(size<0, 2>(frag))::value % 2 == 0);
  static_assert(decltype(stride<0, 0>(frag))::value == 1);
  static_assert(std::is_same_v<typename Fragment::value_type, float>);
  Tensor frag_64b = group_modes<1, 3>(recast<uint2>(frag));
  CUTLASS_PRAGMA_UNROLL
  for (int mi = 0; mi < size<1>(frag_64b); ++mi) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0, 2>(frag_64b) / 2; ++i) {
      cutlass::swap(frag_64b(make_coord(_0{}, _1{}, 2 * i), mi),
                    frag_64b(make_coord(_0{}, _0{}, 2 * i + 1), mi));
    }
  }
}

template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8(Fragment& out) {
  static_assert(decltype(size<0, 0>(out))::value == 2);
  static_assert(decltype(size<0, 1>(out))::value == 2);
  static_assert(decltype(size<0, 2>(out))::value % 2 == 0);
  static_assert(decltype(stride<0, 0>(out))::value == 1);
  static_assert(std::is_same_v<typename Fragment::value_type, float>);
  Tensor frag = group_modes<1, 3>(out);
  CUTLASS_PRAGMA_UNROLL
  for (int mi = 0; mi < size<1>(frag); ++mi) {
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size<0, 1>(frag); ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<0, 2>(frag) / 2; ++i) {
        cutlass::swap(frag(make_coord(_1{}, j, 2 * i), mi),
                      frag(make_coord(_0{}, j, 2 * i + 1), mi));
      }
    }
  }
}

template <int kBlockN, int kHeadDimV, int kCubeBytes, bool UseTsmLd = true,
          typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTLASS_DEVICE void gemm_rs_sm80_pv_fp8_vdirect(
    Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, void* smem_v, TiledMma tiled_mma) {
  static_assert(UseTsmLd, "Sparse FP8 V-direct only supports AIU/TSM readers");
  static_assert((kBlockN == 64 || kBlockN == 128) && kHeadDimV == 128 && kCubeBytes == 128);
  static_assert(std::is_same_v<typename TiledMma::ValTypeA, cutlass::float_e4m3_t>);
  static_assert(std::is_same_v<typename TiledMma::ValTypeB, cutlass::float_e4m3_t>);
  static_assert(std::is_same_v<typename Tensor0::value_type, float>);
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  static_assert(decltype(size<1>(tCrB))::value == kHeadDimV / 16);
  static_assert(decltype(size<2>(tCrB))::value == kBlockN / 32);
  static_assert(decltype(size<2>(tCrA))::value == kBlockN / 32);
  using TsmOp = PPU0015_TSM_LD_SWZL<cute::bfloat16_t, kBlockN, kCubeBytes / 2,
                                    true, true, kHeadDimV / kCubeBytes>;
  Tensor tCrB32 = cute::recast<uint32_t>(tCrB);
  CUTE_STATIC_ASSERT_V(size<0>(tCrB32) == Int<4>{});
  #pragma hggc mmatiestrictly
  {
    CUTLASS_PRAGMA_UNROLL
    for (int mk = 0; mk < kBlockN / 32; ++mk) {
      CUTLASS_PRAGMA_UNROLL
      for (int np = 0; np < kHeadDimV / 32; ++np) {
        const int j0 = 32 * np;
        const int cube = j0 / kCubeBytes;
        const int wcoord = (j0 % kCubeBytes) / 2;
        uint32_t rA[4], rB[4];
        TsmOp::copy(rA, smem_v, 32 * mk,      wcoord, cube, 0);
        TsmOp::copy(rB, smem_v, 32 * mk + 16, wcoord, cube, 0);
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
          const int mn = 2 * np + h;
          tCrB32(0, mn, mk) = __byte_perm(rA[2 * h], rA[2 * h + 1], 0x6420u);
          tCrB32(1, mn, mk) = __byte_perm(rB[2 * h], rB[2 * h + 1], 0x6420u);
          tCrB32(2, mn, mk) = __byte_perm(rA[2 * h], rA[2 * h + 1], 0x7531u);
          tCrB32(3, mn, mk) = __byte_perm(rB[2 * h], rB[2 * h + 1], 0x7531u);
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(acc); ++m) {
            cute::gemm(tiled_mma, tCrA(_, m, mk), tCrB(_, mn, mk), acc(_, m, mn));
          }
        }
      }
    }
  }
}
#endif

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_local_tile_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                          int head_idx, int offset, int seq_len) {
  // (N, D, H)
  auto g_offset = local_tile(m_tensor(_, _, head_idx), cute::make_shape(1, get<1>(tile_shape)),
                             make_coord(offset, _0{}));
  auto g_sequence =
      make_tensor(g_offset.data(),
                  make_layout(cute::make_shape(seq_len, get<1>(tile_shape)), g_offset.stride()));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_local_tile_t_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                            int head_idx, int offset, int seq_len) {
  // (D, N, H)
  auto g_offset = local_tile(m_tensor(_, _, head_idx), cute::make_shape(get<0>(tile_shape), 1),
                             make_coord(_0{}, offset));
  auto g_sequence =
      make_tensor(g_offset.data(),
                  make_layout(cute::make_shape(get<0>(tile_shape), seq_len), g_offset.stride()));
  auto g_tensor = local_tile(g_offset, tile_shape, make_coord(_0{}, _));
  return g_tensor;
}

#ifndef USE_PPU
template <typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_reset_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  constexpr int rA = decltype(rank(tA))::value;
  constexpr int rB = decltype(rank(tB))::value;
  constexpr int rC = decltype(rank(tC))::value;
  static_assert(rA == 3 && rB == 3 && rC == 3);

  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tA); k_block++) {
    cute::gemm(atom, tA(_, _, k_block), tB(_, _, k_block), tC);
    atom.accumulate_ = decltype(atom.accumulate_)::One;
  }
}

template <typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  atom.accumulate_ = decltype(atom.accumulate_)::Zero;
  gemm_reset_zero_acc(atom, tA, tB, tC);
}
#endif

template <class Layout, class Stages = _1>
CUTE_DEVICE constexpr auto unstageSmemLayout(Layout const& layout, Stages stages = {}) {
  return composition(layout, prepend<decltype(rank(layout))::value>(make_layout(stages), _));
}

template <class T>
CUTE_DEVICE T warp_uniform(T a) {
  return __shfl_sync(0xffffffff, a, 0);
}

#ifndef USE_PPU
template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr auto to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<MMA_Traits<SM100_MMA_F8F6F4_SS, a_type, b_type, c_type, cute::C<M>,
                                 cute::C<N>, cute::integral_constant<UMMA::Major, a_major>,
                                 cute::integral_constant<UMMA::Major, b_major>,
                                 cute::integral_constant<UMMA::ScaleIn, a_neg>,
                                 cute::integral_constant<UMMA::ScaleIn, b_neg>>,
                      TAs...>,
             TMs...>) {
  return TiledMMA<
      MMA_Atom<MMA_Traits<SM100_MMA_F8F6F4_TS<a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
                                              b_neg, UMMA::Saturate::False>>,
               TAs...>,
      TMs...>{};
}

template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr auto to_tiled_mma_sm100_ts(
    TiledMMA<
        MMA_Atom<SM100_MMA_F16BF16_SS<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>,
                 TAs...>,
        TMs...>) {
  return TiledMMA<MMA_Atom<SM100_MMA_F16BF16_TS<a_type, b_type, c_type, M, N, a_major, b_major,
                                                a_neg, b_neg, UMMA::Saturate::False>,
                           TAs...>,
                  TMs...>{};
}

template <uint32_t RegCount>
CUTLASS_DEVICE void warpgroup_reg_set() {
  if constexpr (RegCount < 128) {
    cutlass::arch::warpgroup_reg_dealloc<RegCount>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<RegCount>();
  }
}
#endif

// Emulated exp2 using degree-3 polynomial approximation (ported from FA4's e2e_asm2).
// Uses f32x2 packed instructions to halve instruction count vs scalar version.
// Reduces SFU pressure by replacing hardware exp2f with FMA-based computation.
__device__ __forceinline__ void exp2_emulated_2(float& x, float& y) {
  asm(
    "{\n\t"
    ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
    ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
    ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
    // Clamp to [-127, +inf)
    "ppu.max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"       // f1 = max(x, -127.0)
    "ppu.max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"       // f2 = max(y, -127.0)
    "ppu.mov.b64 l1, {f1, f2};\n\t"                  // pack into f32x2
    // Bias for floor extraction: 2^23 + 2^22 = 12582912.0
    "ppu.mov.f32 f3, 0f4B400000;\n\t"
    "ppu.mov.b64 l2, {f3, f3};\n\t"
    // Round-down add to extract floor(x) in lower bits
    "ppu.add.rm.ftz.f32x2 l7, l1, l2;\n\t"           // rounded = clamped + bias
    // Fractional part: frac = clamped - (rounded - bias)
    "ppu.sub.rn.ftz.f32x2 l8, l7, l2;\n\t"           // rounded_back = rounded - bias
    "ppu.sub.rn.ftz.f32x2 l9, l1, l8;\n\t"           // frac = clamped - rounded_back
    // Degree-3 polynomial coefficients (minimax on [0,1))
    "ppu.mov.f32 f7, 0f3D9DF09D;\n\t"                // c3 = 0.077119...
    "ppu.mov.b64 l6, {f7, f7};\n\t"
    "ppu.mov.f32 f6, 0f3E6906A4;\n\t"                // c2 = 0.227564...
    "ppu.mov.b64 l5, {f6, f6};\n\t"
    "ppu.mov.f32 f5, 0f3F31F519;\n\t"                // c1 = 0.695146...
    "ppu.mov.b64 l4, {f5, f5};\n\t"
    "ppu.mov.f32 f4, 0f3F800000;\n\t"                // c0 = 1.0
    "ppu.mov.b64 l3, {f4, f4};\n\t"
    // Horner evaluation: ((c3*f + c2)*f + c1)*f + c0
    "ppu.fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"     // t = frac * c3 + c2
    "ppu.fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"    // t = t * frac + c1
    "ppu.fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"    // t = t * frac + c0
    // Combine integer exponent with polynomial mantissa
    "ppu.mov.b64 {r1, r2}, l7;\n\t"                  // unpack floor values
    "ppu.mov.b64 {r3, r4}, l10;\n\t"                 // unpack polynomial results
    "ppu.shl.b32 r5, r1, 23;\n\t"                    // shift floor to IEEE 754 exponent
    "ppu.add.s32 r7, r5, r3;\n\t"                    // result_x = 2^floor * poly
    "ppu.shl.b32 r6, r2, 23;\n\t"
    "ppu.add.s32 r8, r6, r4;\n\t"                    // result_y = 2^floor * poly
    "ppu.mov.b32 %0, r7;\n\t"
    "ppu.mov.b32 %1, r8;\n\t"
    "}\n"
    : "=f"(x), "=f"(y)
    : "f"(x), "f"(y)
  );
}

#ifdef USE_PPU
template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

template<typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = acc_layout;
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
    } else {
        // PPU 1.0 C accumulator layout: first mode has 8 registers per thread.
        // Split it as ((2, 4), MMA_M, MMA_N) so row/col views match FA3's
        // PPU1.0 softmax and epilogue interpretation.
        static_assert(decltype(size<0>(acc_layout))::value == 8);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_4>{});
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    }
};

// SM80: convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// for m16n8k16, or to (4, MMA_M, MMA_N) for m16n8k8.
template<typename MMA_Traits, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
    auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});  // ((2, N / 16))
    return make_layout(make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)), get<1>(acc_layout), coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
}

// Convert float tensor to Element type output
template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const &tensor, Tensor<EngineOut, Layout> &out) {
    using From_type = typename Engine::value_type;
    using To_type = typename EngineOut::value_type;
    static constexpr int FragmentSize = std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
    static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
    Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
    Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
    static_assert(size(frag) == size(out_frg));
    cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
    #pragma unroll
    for (int i = 0; i < size(frag); ++i) { out_frg[i] = convert_op(frag[i]); }
}

// cp.async wait_group
template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_PPU_ENABLED)
    asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// SM80 softmax helpers

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++) {
            summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

// Pairwise variant of thread_reduce_: same per-row reduction, with dependency
// depth log2(NI) instead of NI (requires power-of-two NI, checked below).
// NI = 4 (cols per thread per row of the m16n16 atom) x (kStageSplitN/16 repeats)
//    = kStageSplitN/4, where kStageSplitN = kTileN / StageCountKV = 32 or 64
//    -> NI ranges 8..16 over all dispatch tile shapes.
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_tree_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    constexpr int NI = decltype(size<1>(tensor))::value;
    static_assert(NI >= 2 && (NI & (NI - 1)) == 0, "tree reduce requires power-of-two column count");
    using ValT = typename Engine1::value_type;
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        ValT partial[NI / 2];
        #pragma unroll
        for (int ni = 0; ni < NI / 2; ni++) {
            partial[ni] = op(tensor(mi, 2 * ni), tensor(mi, 2 * ni + 1));
        }
        #pragma unroll
        for (int s = NI / 4; s >= 1; s >>= 1) {
            #pragma unroll
            for (int ni = 0; ni < s; ni++) {
                partial[ni] = op(partial[2 * ni], partial[2 * ni + 1]);
            }
        }
        summary(mi) = zero_init ? partial[0] : op(summary(mi), partial[0]);
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max_val){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max_val, max_op);
}

// Per-stage row-max for the OnlyScore/Full max_score path (all dtypes).
// Tree-shaped thread-local reduce each stage (log2(N) dependency depth),
// with the quad allreduce deferred to the tile's last stage. Max is exact
// and order-independent, so this matches the serial-chain + per-stage-quad
// result bit-for-bit. Between stages max_val holds per-thread partial maxima;
// the only consumer (the max_score write) runs after the last stage's
// allreduce inside the same step.
template<int Stages, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max_staged(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max_val, int stage) {
    MaxOp<float> max_op;
    if (stage == 0) {
        thread_reduce_tree_</*zero_init=*/true>(tensor, max_val, max_op);
    } else {
        thread_reduce_tree_</*zero_init=*/false>(tensor, max_val, max_op);
    }
    if (stage == Stages - 1) {
        quad_allreduce_(max_val, max_val, max_op);
    }
}

template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}

// Apply exp2 to all elements: tensor = exp2(tensor * scale - max_scaled)
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max_val, const float scale) {
    static constexpr float max_offset = float(Max_offset);
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max_val) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = Check_inf
            ? (max_val(mi) == -INFINITY ? 0.f : (!Scale_max ? max_val(mi) : max_val(mi) * scale) - max_offset)
            : (!Scale_max ? max_val(mi) : max_val(mi) * scale) - max_offset;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

// SM80 online softmax state
template <int kNRows, int Max_offset=0>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;
    float const softmax_scale_log2;

    CUTLASS_DEVICE Softmax(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            reduce_max</*zero_init=*/true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            reduce_max</*zero_init=*/false>(scores, row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
        reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
    };

    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            // row_sum is a sum of exp2 values, so it is never negative;
            const bool sum_ok = sum > 0.f;
            float inv_sum = sum_ok ? 1.f / sum : 0.f;
            scores_scale(mi) = inv_sum * final_scale;
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            // LSE format must match Sm100FmhaFwdMainloop's correction_merge_ksplit:
            // lse = row_max + log2(sum) / scale_softmax_log2
            //     = (row_max * sm_scale + ln(sum)) / sm_scale = LSE_natural / sm_scale
            // The reduction kernel computes exp2f(scale_softmax_log2 * (lse_s - lse_r))
            // which gives exp(LSE_natural_s - LSE_natural_r) -- correct only with this format.
            row_sum(mi) = sum_ok ? row_max(mi) + __log2f(sum) / softmax_scale_log2 : -INFINITY;
        }
        return scores_scale;
    };

    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
        }
    };
};

// SM80 GEMM with interleaved smem->reg copy and MMA, with Hook callback
template<bool A_in_regs=false, bool B_in_regs=false, bool SwapAB=false,
         typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_sm80(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                              Tensor4 const& tCsB, TiledMma tiled_mma,
                              TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                              ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B, Hook fn) {
    if constexpr (SwapAB) {
        gemm_sm80<B_in_regs, A_in_regs>(acc, tCrB, tCrA, tCsB, tCsA, tiled_mma, smem_tiled_copy_B, smem_tiled_copy_A, smem_thr_copy_B, smem_thr_copy_A, fn);
    } else {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
        if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
                if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
            }
            if constexpr (!std::is_same_v<Hook, std::nullptr_t>) {
                if (i == 0) { fn(); }
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }
}

// SM80 register-shared PV GEMM with interleaved V smem->reg copy
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
CUTLASS_DEVICE void gemm_rs_sm80(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                 TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                 ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}
#endif

}  // namespace cutlass::fmha::collective
