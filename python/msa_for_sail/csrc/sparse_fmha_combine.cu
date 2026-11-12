// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include "sparse_fmha_combine.cuh"
#include "tvm_ffi_utils.h"

#include <cmath>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

namespace minfer {
namespace sparse {

__device__ __forceinline__ void load_partial8(
    const __ppu_bfloat16* src, float2 o[4]) {
  uint4 raw = *reinterpret_cast<const uint4*>(src);
  o[0] = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(&raw.x));
  o[1] = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(&raw.y));
  o[2] = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(&raw.z));
  o[3] = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(&raw.w));
}

template <int kHeadDim>
__global__ void sparse_fmha_combine_kernel(SparseCombineParams params) {
  constexpr int kVec = 8;
  constexpr int kLanes = kHeadDim / kVec;
  constexpr int kPairs = 128 / kLanes;
  static_assert(kHeadDim == 128, "vector layout assumes head_dim=128");
  static_assert(kLanes * kVec == kHeadDim);

  int pair = threadIdx.x / kLanes;
  int lane = threadIdx.x % kLanes;

  int q_head_idx = blockIdx.x * kPairs + pair;
  int total_pairs = params.total_q * params.num_heads_q;
  if (q_head_idx >= total_pairs) return;

  int q_abs = q_head_idx / params.num_heads_q;
  int head_q_idx = q_head_idx % params.num_heads_q;
  int d = lane * kVec;

  int head_kv_idx = head_q_idx / params.qhead_per_kv;
  int raw_count = params.split_counts[q_abs * params.num_heads_kv + head_kv_idx];
  int num_splits = raw_count;
  if (num_splits > params.topk) num_splits = params.topk;

  int64_t lse_idx = ((int64_t)q_abs * params.num_heads_q + head_q_idx)
                    * params.topk;
  int64_t o_idx = lse_idx * kHeadDim + d;
  int64_t out_o_off = (int64_t)q_abs * params.num_heads_q * kHeadDim
                    + (int64_t)head_q_idx * kHeadDim + d;

  if (num_splits == 0) {
    *reinterpret_cast<uint4*>(params.ptr_O + out_o_off) = make_uint4(0, 0, 0, 0);
    if (lane == 0) {
      params.ptr_LSE[q_abs * params.num_heads_q + head_q_idx] = -INFINITY;
      if (params.ptr_LSE_temperature != nullptr)
        params.ptr_LSE_temperature[q_abs * params.num_heads_q + head_q_idx] = -INFINITY;
    }
    return;
  }

  float running_lse = -__FLT_MAX__;
  float running_w = 0.f;
  float2 acc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) acc[j] = make_float2(0.f, 0.f);

#pragma unroll 8
  for (int s = 0; s < num_splits; ++s) {
    float lse_s = params.ptr_LSE_partial[lse_idx];
    lse_idx += 1;
    bool valid = (lse_s != -INFINITY);
    float2 o[4];
    load_partial8(params.ptr_O_partial + o_idx, o);

    float new_max = fmaxf(running_lse, lse_s);
    float r_run = exp2f(params.scale_softmax_log2 * (running_lse - new_max));
    float r_s = exp2f(params.scale_softmax_log2 * (lse_s - new_max));
    running_w = running_w * r_run + r_s;
    running_lse = new_max;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 os = valid ? o[j] : make_float2(0.f, 0.f);
      acc[j].x = acc[j].x * r_run + os.x * r_s;
      acc[j].y = acc[j].y * r_run + os.y * r_s;
    }
    o_idx += kHeadDim;
  }

  if (running_w == 0.f) {
    *reinterpret_cast<uint4*>(params.ptr_O + out_o_off) = make_uint4(0, 0, 0, 0);
    if (lane == 0) {
      params.ptr_LSE[q_abs * params.num_heads_q + head_q_idx] = -INFINITY;
      if (params.ptr_LSE_temperature != nullptr)
        params.ptr_LSE_temperature[q_abs * params.num_heads_q + head_q_idx] = -INFINITY;
    }
  } else {
    float inv_w = 1.f / running_w;
    uint4 out[1];
#pragma unroll
    for (int j = 0; j < 4; j += 4) {
      *reinterpret_cast<__ppu_bfloat162*>(&out[j / 4].x) =
          __float22bfloat162_rn(make_float2(acc[j].x * inv_w, acc[j].y * inv_w));
      *reinterpret_cast<__ppu_bfloat162*>(&out[j / 4].y) =
          __float22bfloat162_rn(make_float2(acc[j + 1].x * inv_w, acc[j + 1].y * inv_w));
      *reinterpret_cast<__ppu_bfloat162*>(&out[j / 4].z) =
          __float22bfloat162_rn(make_float2(acc[j + 2].x * inv_w, acc[j + 2].y * inv_w));
      *reinterpret_cast<__ppu_bfloat162*>(&out[j / 4].w) =
          __float22bfloat162_rn(make_float2(acc[j + 3].x * inv_w, acc[j + 3].y * inv_w));
    }
    *reinterpret_cast<uint4*>(params.ptr_O + out_o_off) = out[0];
    if (lane == 0) {
      float lse_out = params.sm_scale
          * (running_lse + log2f(running_w) / params.scale_softmax_log2);
      params.ptr_LSE[q_abs * params.num_heads_q + head_q_idx] = lse_out;
    }
  }

  if (params.ptr_LSE_temperature_partial != nullptr
      && params.ptr_LSE_temperature != nullptr && lane == 0) {
    float t_running_lse = -__FLT_MAX__;
    float t_running_w = 0.f;
    int64_t t_idx = ((int64_t)q_abs * params.num_heads_q + head_q_idx)
                    * params.topk;

    for (int s = 0; s < num_splits; ++s) {
      float t_lse_s = params.ptr_LSE_temperature_partial[t_idx];
      t_idx += 1;
      float new_max = fmaxf(t_running_lse, t_lse_s);
      float r_run = exp2f(params.lse_temperature_scale_log2
                          * (t_running_lse - new_max));
      float r_s = exp2f(params.lse_temperature_scale_log2
                        * (t_lse_s - new_max));
      t_running_w = t_running_w * r_run + r_s;
      t_running_lse = new_max;
    }

    if (t_running_w == 0.f) {
      params.ptr_LSE_temperature[q_abs * params.num_heads_q + head_q_idx] =
          -INFINITY;
    } else {
      float t_lse_out = params.scale_temp
          * (t_running_lse + log2f(t_running_w)
             / params.lse_temperature_scale_log2);
      params.ptr_LSE_temperature[q_abs * params.num_heads_q + head_q_idx] =
          t_lse_out;
    }
  }
}

void sparse_fmha_combine_forward(
    TensorView O_partial,
    TensorView LSE_partial,
    TensorView split_counts,
    TensorView O,
    TensorView LSE,
    double sm_scale,
    int64_t head_dim,
    int64_t topk,
    int64_t total_q,
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t qhead_per_kv,
    int64_t stream_ptr,
    TensorView LSE_temperature_partial,
    TensorView LSE_temperature,
    double lse_temperature_scale) {
  CHECK_INPUT_AND_TYPE(O_partial, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(LSE_partial, dl_float32);
  CHECK_INPUT_AND_TYPE(split_counts, dl_int32);
  CHECK_INPUT_AND_TYPE(O, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(LSE, dl_float32);
  CHECK_DIM(4, O_partial);
  CHECK_DIM(3, LSE_partial);
  CHECK_DIM(2, split_counts);
  CHECK_DIM(3, O);
  CHECK_DIM(2, LSE);

  TVM_FFI_ICHECK_EQ(O_partial.size(0), total_q);
  TVM_FFI_ICHECK_EQ(LSE_partial.size(0), total_q);
  TVM_FFI_ICHECK_EQ(O_partial.size(1), num_heads_q);
  TVM_FFI_ICHECK_EQ(LSE_partial.size(1), num_heads_q);
  TVM_FFI_ICHECK_EQ(O_partial.size(2), topk);
  TVM_FFI_ICHECK_EQ(LSE_partial.size(2), topk);
  TVM_FFI_ICHECK_EQ(O_partial.size(3), head_dim);
  TVM_FFI_ICHECK_EQ(O.size(0), total_q);
  TVM_FFI_ICHECK_EQ(O.size(1), num_heads_q);
  TVM_FFI_ICHECK_EQ(O.size(2), head_dim);
  TVM_FFI_ICHECK_EQ(LSE.size(0), total_q);
  TVM_FFI_ICHECK_EQ(LSE.size(1), num_heads_q);
  TVM_FFI_ICHECK_EQ(split_counts.size(0), total_q);
  TVM_FFI_ICHECK_EQ(split_counts.size(1), num_heads_kv);

  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(O_partial.data_ptr()) % 16, 0)
      << "O_partial base pointer must be aligned to one 8-element vector";

  const hggcStream_t stream = reinterpret_cast<hggcStream_t>(stream_ptr);

  float sm_scale_f = static_cast<float>(sm_scale);
  float scale_log2 = sm_scale_f * kLog2e;
  float lse_temp_scale = static_cast<float>(lse_temperature_scale);
  float lse_temp_scale_log2 = (lse_temp_scale > 0.f)
      ? scale_log2 * lse_temp_scale : 0.f;

  bool has_temperature = (lse_temp_scale > 0.f
                          && fabsf(lse_temp_scale - 1.0f) > 1e-12f);

  SparseCombineParams params;
  params.ptr_O_partial =
      static_cast<const __nv_bfloat16*>(O_partial.data_ptr());
  params.ptr_LSE_partial = static_cast<const float*>(LSE_partial.data_ptr());
  params.ptr_LSE_temperature_partial = has_temperature
      ? static_cast<const float*>(LSE_temperature_partial.data_ptr())
      : nullptr;
  params.split_counts = static_cast<const int32_t*>(split_counts.data_ptr());
  params.ptr_O = static_cast<__ppu_bfloat16*>(O.data_ptr());
  params.ptr_LSE = static_cast<float*>(LSE.data_ptr());
  params.ptr_LSE_temperature = has_temperature
      ? static_cast<float*>(LSE_temperature.data_ptr())
      : nullptr;
  params.sm_scale = sm_scale_f;
  params.scale_softmax_log2 = scale_log2;
  params.lse_temperature_scale_log2 = lse_temp_scale_log2;
  params.scale_temp = sm_scale_f * lse_temp_scale;
  params.head_dim = static_cast<int>(head_dim);
  params.topk = static_cast<int>(topk);
  params.total_q = static_cast<int>(total_q);
  params.num_heads_q = static_cast<int>(num_heads_q);
  params.num_heads_kv = static_cast<int>(num_heads_kv);
  params.qhead_per_kv = static_cast<int>(qhead_per_kv);

  constexpr int kPairsPerBlock = 8;
  int grid = static_cast<int>((total_q * num_heads_q + kPairsPerBlock - 1)
                              / kPairsPerBlock);

  if (head_dim != 128) {
    TVM_FFI_THROW(RuntimeError)
        << "sparse_fmha_combine only supports head_dim=128, got " << head_dim;
  }
  sparse_fmha_combine_kernel<128><<<grid, 128, 0, stream>>>(params);

  hggcError_t status = hggcGetLastError();
  if (status != hggcSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "sparse_fmha_combine launch failed: " << hggcGetErrorString(status);
  }
}

}
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sparse_fmha_combine,
    minfer::sparse::sparse_fmha_combine_forward);
