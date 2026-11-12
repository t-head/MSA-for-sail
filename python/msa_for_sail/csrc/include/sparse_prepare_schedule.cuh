// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <hggc_runtime.h>
#include <cstdint>

namespace minfer {
namespace sparse {

constexpr int kFlatScheduleThreads = 128;
constexpr int kFlatScheduleWarps = kFlatScheduleThreads / 32;
constexpr int kFwdSplitAtomicThreads = 256;

__device__ __forceinline__ int rows_in_batch(
    const int32_t* cu_seqlens_k, int batch_idx, int blk_kv) {
  int seqlen = cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx];
  return (seqlen + blk_kv - 1) / blk_kv;
}

__device__ __forceinline__ int max_rows_per_batch(
    const int32_t* cu_seqlens_k, int batch, int blk_kv) {
  int max_rows = 0;
  for (int b = 0; b < batch; ++b) {
    max_rows = max(max_rows, rows_in_batch(cu_seqlens_k, b, blk_kv));
  }
  return max_rows;
}

__device__ __forceinline__ int rows_before_level(
    const int32_t* cu_seqlens_k, int batch, int blk_kv, int level) {
  int total = 0;
  for (int b = 0; b < batch; ++b) {
    total += min(rows_in_batch(cu_seqlens_k, b, blk_kv), level);
  }
  return total;
}

__device__ __forceinline__ void decode_sparse_row_linear(
    const int32_t* cu_seqlens_k,
    int batch,
    int blk_kv,
    int row_linear,
    int& batch_idx,
    int& kv_block_idx) {
  int max_rows = max_rows_per_batch(cu_seqlens_k, batch, blk_kv);

  int lo = 0;
  int hi = max_rows;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    int rows_before_mid_next = rows_before_level(cu_seqlens_k, batch, blk_kv, mid + 1);
    if (rows_before_mid_next <= row_linear) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  int level = lo;

  int offset = row_linear - rows_before_level(cu_seqlens_k, batch, blk_kv, level);

  int active_idx = 0;
  batch_idx = 0;
  int found = 0;
  for (int b = 0; b < batch; ++b) {
    if (found == 0) {
      int rows = rows_in_batch(cu_seqlens_k, b, blk_kv);
      if (rows > level) {
        if (active_idx == offset) {
          batch_idx = b;
          found = 1;
        }
        active_idx++;
      }
    }
  }
  kv_block_idx = level;
}

__device__ __forceinline__ void emit_work(
    int32_t* scheduler_metadata,
    int64_t work_capacity,
    int work_idx,
    int head_kv_idx,
    int row_linear,
    int q_begin,
    int q_count,
    int batch_idx,
    int kv_block_idx) {
  if (work_idx < work_capacity) {
    int32_t* meta = scheduler_metadata + static_cast<int64_t>(work_idx) * 6;
    meta[0] = head_kv_idx;
    meta[1] = row_linear;
    meta[2] = q_begin;
    meta[3] = q_count;
    meta[4] = batch_idx;
    meta[5] = kv_block_idx;
  }
}

__global__ void sparse_prepare_flat_schedule_kernel(
    const int32_t* k2q_row_ptr,
    const int32_t* cu_seqlens_k,
    int32_t* scheduler_metadata,
    int32_t* work_count,
    int target_q_per_cta,
    int64_t work_capacity,
    int num_heads_kv,
    int total_rows,
    int blk_kv,
    int batch);

__global__ void sparse_prepare_fwd_split_atomic_kernel(
    const int32_t* k2q_row_ptr,
    const int32_t* k2q_q_indices,
    const int32_t* scheduler_metadata,
    const int32_t* work_count,
    int32_t* qsplit_indices,
    int32_t* split_counts,
    const int32_t* cu_seqlens_q,
    int64_t work_capacity,
    int max_seqlen_q,
    int topk,
    int total_q,
    int total_rows,
    int num_heads_kv);

}
}
