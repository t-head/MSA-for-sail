// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <hggc_fp8.h>
#include <cstdint>
#include <cmath>

namespace minfer {
namespace sparse {

constexpr int32_t kSentinelSlot = 0xFF;
constexpr int32_t kSentinelPacked = static_cast<int32_t>(0xFF000000);

__device__ __forceinline__ int decode_q_idx_local(int32_t packed) {
  return packed & 0x00FFFFFF;
}

__device__ __forceinline__ int decode_slot(int32_t packed) {
  return static_cast<int>((static_cast<uint32_t>(packed) >> 24) & 0xFF);
}

__device__ __forceinline__ bool is_sentinel(int32_t packed) {
  return (static_cast<uint32_t>(packed) >> 24) == 0xFFu;
}

constexpr float kLog2e = 1.4426950408889634f;

template <typename Element_>
struct SparseFwdParams {
  using Element_t = Element_;

  const Element_* ptr_Q;
  int64_t q_stride_n;
  int64_t q_stride_h;
  const Element_* ptr_K;
  const Element_* ptr_V;

  const int32_t* k2q_row_ptr;
  const int32_t* k2q_q_indices;
  const int32_t* qsplit_indices;
  const int32_t* scheduler_metadata;
  const int32_t* work_count;

  const int32_t* cu_seqlens_q;
  const int32_t* cu_seqlens_k;
  const int32_t* seqused_k;
  const int32_t* kv_indices;
  const int32_t* kv_page_indptr;

  __nv_bfloat16* ptr_O_partial;
  float* ptr_LSE_partial;
  float* ptr_LSE_temperature_partial;
  const int32_t* split_counts;

  float sm_scale;
  float scale_softmax_log2;
  float lse_temperature_scale;
  float lse_temperature_scale_log2;
  int return_temperature_lse;
  int head_dim;
  int num_heads_kv;
  int num_heads_q;
  int qhead_per_kv;
  int total_q;
  int total_rows;
  int batch;
  int blk_kv;
  int page_size;
  int topk;
  int max_seqlen_q;
  int is_paged;
  int kv_is_nhd;
  int64_t nnz;
};

template <typename Element, int kTileM, int kTileN, int kHeadDim>
struct SparseFwdSmemBudget {
  static constexpr int kKBytes = kTileN * kHeadDim * sizeof(Element);
  static constexpr int kVBytes = kTileN * kHeadDim * sizeof(Element);
  static constexpr int kQBytes = kTileM * kHeadDim * sizeof(Element);
  static constexpr int kEdgeBytes = kTileM * 2 * sizeof(int32_t)
                                  + kTileM * 2 * sizeof(int64_t);
  static constexpr int kTotal = kKBytes + kVBytes + kQBytes + kEdgeBytes;
};

struct SparseCombineParams {
  const __nv_bfloat16* ptr_O_partial;
  const float* ptr_LSE_partial;
  const float* ptr_LSE_temperature_partial;
  const int32_t* split_counts;
  __ppu_bfloat16* ptr_O;
  float* ptr_LSE;
  float* ptr_LSE_temperature;

  float sm_scale;
  float scale_softmax_log2;
  float lse_temperature_scale_log2;
  float scale_temp;
  int head_dim;
  int topk;
  int total_q;
  int num_heads_q;
  int num_heads_kv;
  int qhead_per_kv;
};

}
}
