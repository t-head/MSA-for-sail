// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include "sparse_prepare_schedule.cuh"
#include "tvm_ffi_utils.h"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

namespace minfer {
namespace sparse {

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
    int batch) {
  int tidx = threadIdx.x;
  int lane_idx = tidx % 32;
  int warp_idx = tidx / 32;
  int block_idx = blockIdx.x;
  int row_head_idx = block_idx * kFlatScheduleWarps + warp_idx;
  int total_row_heads = total_rows * num_heads_kv;

  int head_kv_idx = 0;
  int row_linear = 0;
  int row_count = 0;
  int num_chunks = 0;
  int batch_idx = 0;
  int kv_block_idx = 0;

  if (row_head_idx < total_row_heads) {
    row_linear = row_head_idx / num_heads_kv;
    head_kv_idx = row_head_idx - row_linear * num_heads_kv;
    if (lane_idx == 0) {
      int row_start = k2q_row_ptr[head_kv_idx * (total_rows + 1) + row_linear];
      int row_end = k2q_row_ptr[head_kv_idx * (total_rows + 1) + row_linear + 1];
      row_count = row_end - row_start;
      decode_sparse_row_linear(cu_seqlens_k, batch, blk_kv, row_linear, batch_idx, kv_block_idx);
      if (row_count > 0) {
        num_chunks = (row_count + target_q_per_cta - 1) / target_q_per_cta;
      }
    }
  }

  row_count = __shfl_sync(0xFFFFFFFF, row_count, 0);
  num_chunks = __shfl_sync(0xFFFFFFFF, num_chunks, 0);
  batch_idx = __shfl_sync(0xFFFFFFFF, batch_idx, 0);
  kv_block_idx = __shfl_sync(0xFFFFFFFF, kv_block_idx, 0);

  for (int chunk_idx = lane_idx; chunk_idx < num_chunks; chunk_idx += 32) {
    int work_idx = atomicAdd(work_count, 1);
    int q_begin = chunk_idx * target_q_per_cta;
    int q_count = min(target_q_per_cta, row_count - q_begin);
    emit_work(
        scheduler_metadata,
        work_capacity,
        work_idx,
        head_kv_idx,
        row_linear,
        q_begin,
        q_count,
        batch_idx,
        kv_block_idx);
  }
}

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
    int num_heads_kv) {
  __shared__ int sRow[3];

  int block_idx = blockIdx.x;
  if (block_idx >= work_count[0]) {
    return;
  }

  int tidx = threadIdx.x;
  if (tidx == 0) {
    int head_kv_idx = scheduler_metadata[block_idx * 6 + 0];
    int row_linear = scheduler_metadata[block_idx * 6 + 1];
    int q_begin = scheduler_metadata[block_idx * 6 + 2];
    int q_count = scheduler_metadata[block_idx * 6 + 3];
    int batch_idx = scheduler_metadata[block_idx * 6 + 4];

    int row_start = k2q_row_ptr[head_kv_idx * (total_rows + 1) + row_linear] + q_begin;
    sRow[0] = row_start;
    sRow[1] = q_count;
    sRow[2] = batch_idx;
  }
  __syncthreads();

  int row_start = sRow[0];
  int q_count = sRow[1];
  int batch_idx = sRow[2];
  int head_kv_idx = scheduler_metadata[block_idx * 6 + 0];

  int nnz = total_q * topk;

  for (int qi = tidx; qi < q_count; qi += blockDim.x) {
    int edge = row_start + qi;
    int q_idx = k2q_q_indices[head_kv_idx * nnz + edge];
    if (q_idx >= 0 && q_idx < max_seqlen_q) {
      int q_abs = cu_seqlens_q[batch_idx] + q_idx;
      int split_slot = atomicAdd(&split_counts[q_abs * num_heads_kv + head_kv_idx], 1);
      if (split_slot < topk) {
        qsplit_indices[head_kv_idx * nnz + edge] =
            q_idx | ((split_slot & 0xFF) << 24);
      } else {
        qsplit_indices[head_kv_idx * nnz + edge] =
            q_idx | static_cast<int32_t>(0xFF000000);
      }
    } else {
      qsplit_indices[head_kv_idx * nnz + edge] =
          static_cast<int32_t>(0xFF000000);
    }
  }
}

void sparse_prepare_flat_schedule_forward(
    TensorView k2q_row_ptr,
    TensorView cu_seqlens_k,
    TensorView scheduler_metadata,
    TensorView work_count,
    int64_t target_q_per_cta,
    int64_t work_capacity,
    int64_t num_heads_kv,
    int64_t blk_kv,
    int64_t stream_ptr) {
  CHECK_INPUT_AND_TYPE(k2q_row_ptr, dl_int32);
  CHECK_INPUT_AND_TYPE(cu_seqlens_k, dl_int32);
  CHECK_INPUT_AND_TYPE(scheduler_metadata, dl_int32);
  CHECK_INPUT_AND_TYPE(work_count, dl_int32);
  CHECK_DIM(2, k2q_row_ptr);
  CHECK_DIM(1, cu_seqlens_k);
  CHECK_DIM(2, scheduler_metadata);
  CHECK_DIM(1, work_count);
  TVM_FFI_ICHECK_EQ(scheduler_metadata.size(1), 6);
  TVM_FFI_ICHECK_EQ(work_count.size(0), 1);

  int total_rows = static_cast<int>(k2q_row_ptr.size(1) - 1);
  int batch = static_cast<int>(cu_seqlens_k.size(0) - 1);

  const hggcStream_t stream = reinterpret_cast<hggcStream_t>(stream_ptr);

  size_t meta_bytes = static_cast<size_t>(scheduler_metadata.size(0) * scheduler_metadata.size(1)) *
      sizeof(int32_t);
  hggcMemsetAsync(scheduler_metadata.data_ptr(), 0, meta_bytes, stream);
  hggcMemsetAsync(work_count.data_ptr(), 0, sizeof(int32_t), stream);

  if (total_rows <= 0 || num_heads_kv <= 0) {
    return;
  }

  int warps_per_cta = kFlatScheduleThreads / 32;
  int total_row_heads = total_rows * static_cast<int>(num_heads_kv);
  int grid_ctas = (total_row_heads + warps_per_cta - 1) / warps_per_cta;

  sparse_prepare_flat_schedule_kernel<<<grid_ctas, kFlatScheduleThreads, 0, stream>>>(
      static_cast<const int32_t*>(k2q_row_ptr.data_ptr()),
      static_cast<const int32_t*>(cu_seqlens_k.data_ptr()),
      static_cast<int32_t*>(scheduler_metadata.data_ptr()),
      static_cast<int32_t*>(work_count.data_ptr()),
      static_cast<int>(target_q_per_cta),
      work_capacity,
      static_cast<int>(num_heads_kv),
      total_rows,
      static_cast<int>(blk_kv),
      batch);
}

void sparse_prepare_fwd_split_atomic_forward(
    TensorView k2q_row_ptr,
    TensorView k2q_q_indices,
    TensorView scheduler_metadata,
    TensorView work_count,
    TensorView qsplit_indices,
    TensorView split_counts,
    TensorView cu_seqlens_q,
    int64_t work_capacity,
    int64_t max_seqlen_q,
    int64_t topk,
    int64_t total_q,
    int64_t stream_ptr) {
  CHECK_INPUT_AND_TYPE(k2q_row_ptr, dl_int32);
  CHECK_INPUT_AND_TYPE(k2q_q_indices, dl_int32);
  CHECK_INPUT_AND_TYPE(scheduler_metadata, dl_int32);
  CHECK_INPUT_AND_TYPE(work_count, dl_int32);
  CHECK_INPUT_AND_TYPE(qsplit_indices, dl_int32);
  CHECK_INPUT_AND_TYPE(split_counts, dl_int32);
  CHECK_INPUT_AND_TYPE(cu_seqlens_q, dl_int32);
  CHECK_DIM(2, k2q_row_ptr);
  CHECK_DIM(2, k2q_q_indices);
  CHECK_DIM(2, scheduler_metadata);
  CHECK_DIM(1, work_count);
  CHECK_DIM(2, qsplit_indices);
  CHECK_DIM(2, split_counts);
  CHECK_DIM(1, cu_seqlens_q);
  TVM_FFI_ICHECK_EQ(scheduler_metadata.size(1), 6);
  TVM_FFI_ICHECK_EQ(work_count.size(0), 1);

  int num_heads_kv = static_cast<int>(k2q_row_ptr.size(0));
  int total_rows = static_cast<int>(k2q_row_ptr.size(1) - 1);
  int batch = static_cast<int>(cu_seqlens_q.size(0) - 1);

  const hggcStream_t stream = reinterpret_cast<hggcStream_t>(stream_ptr);

  if (work_capacity <= 0 || total_rows <= 0 || num_heads_kv <= 0) {
    return;
  }

  sparse_prepare_fwd_split_atomic_kernel<<<work_capacity, kFwdSplitAtomicThreads, 0, stream>>>(
      static_cast<const int32_t*>(k2q_row_ptr.data_ptr()),
      static_cast<const int32_t*>(k2q_q_indices.data_ptr()),
      static_cast<const int32_t*>(scheduler_metadata.data_ptr()),
      static_cast<const int32_t*>(work_count.data_ptr()),
      static_cast<int32_t*>(qsplit_indices.data_ptr()),
      static_cast<int32_t*>(split_counts.data_ptr()),
      static_cast<const int32_t*>(cu_seqlens_q.data_ptr()),
      work_capacity,
      static_cast<int>(max_seqlen_q),
      static_cast<int>(topk),
      static_cast<int>(total_q),
      total_rows,
      num_heads_kv);
}

void sparse_prepare_schedule_forward(
    TensorView k2q_row_ptr,
    TensorView k2q_q_indices,
    TensorView cu_seqlens_k,
    TensorView cu_seqlens_q,
    TensorView scheduler_metadata,
    TensorView work_count,
    TensorView qsplit_indices,
    TensorView split_counts,
    int64_t target_q_per_cta,
    int64_t work_capacity,
    int64_t max_seqlen_q,
    int64_t topk,
    int64_t total_q,
    int64_t stream_ptr) {
  int64_t num_heads_kv = k2q_row_ptr.size(0);
  int64_t blk_kv = 128;

  sparse_prepare_flat_schedule_forward(
      k2q_row_ptr,
      cu_seqlens_k,
      scheduler_metadata,
      work_count,
      target_q_per_cta,
      work_capacity,
      num_heads_kv,
      blk_kv,
      stream_ptr);

  size_t split_counts_bytes = static_cast<size_t>(split_counts.size(0) * split_counts.size(1)) *
      sizeof(int32_t);
  const hggcStream_t stream = reinterpret_cast<hggcStream_t>(stream_ptr);
  hggcMemsetAsync(split_counts.data_ptr(), 0, split_counts_bytes, stream);

  sparse_prepare_fwd_split_atomic_forward(
      k2q_row_ptr,
      k2q_q_indices,
      scheduler_metadata,
      work_count,
      qsplit_indices,
      split_counts,
      cu_seqlens_q,
      work_capacity,
      max_seqlen_q,
      topk,
      total_q,
      stream_ptr);
}

}
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sparse_prepare_flat_schedule,
    minfer::sparse::sparse_prepare_flat_schedule_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sparse_prepare_fwd_split_atomic,
    minfer::sparse::sparse_prepare_fwd_split_atomic_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sparse_prepare_schedule,
    minfer::sparse::sparse_prepare_schedule_forward);
