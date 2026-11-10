/*
 * Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 *
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include "utils.cuh"

namespace flashinfer {

// ============================================================================
// Cost model constants
// ============================================================================

constexpr float kPerIterOverhead = 43;
constexpr float kTileGlobalOverhead = 110;
constexpr float kSMGlobalOverhead = 165;
constexpr float kSplitExtraOverhead = 150;


// ============================================================================
// Plan kernel constants
// ============================================================================

constexpr int MAX_SMS = 256;
constexpr int MAX_TASKS_PER_SM = 96;
constexpr int MAX_UNSPLIT = 4096;
constexpr int MAX_REBALANCE_ITERS = 1024;
// Per-split iteration floor. Dense always 1. Sparse: 1 by default, but the
// adaptive path coarsens it to SPARSE_COARSE (page-level: 2 tiles of kTileN=64
// = 1 page of 128) ONLY for the tiniest workloads (avg_iters <= thresh, where
// budget_iters would otherwise collapse to 1 = tile-level = 32 splits, doubling
// reduction depth vs Triton's page-level 16). A FLAT SPARSE=2 was tried and
// reverted: it also raised min_piece_cost and phase2's fit threshold for
// avg_iters>=2 cases, tipping the Gate-B split/nosplit decision to nosplit
// (e.g. [D][72,498712]) and collapsing KV parallelism. Gating COARSE on
// avg_iters<=thresh confines it to the regime that needs it. See
// docs/msa_sparse_decode_page_split_coarsen_spec.md.
constexpr int MIN_ITERS_PER_SPLIT_DENSE         = 1;
constexpr int MIN_ITERS_PER_SPLIT_SPARSE        = 1;
constexpr int MIN_ITERS_PER_SPLIT_SPARSE_COARSE = 2;
constexpr int SPARSE_COARSE_AVG_ITERS_THRESH    = 1;

// ============================================================================
// Shared memory state
// ============================================================================

struct PlanSharedState {
  int     sm_cost[MAX_SMS];
  int16_t sm_task_count[MAX_SMS];

  int16_t task_kv_begin[MAX_SMS][MAX_TASKS_PER_SM];
  int16_t task_kv_end[MAX_SMS][MAX_TASKS_PER_SM];
  int16_t task_unsplit_id[MAX_SMS][MAX_TASKS_PER_SM];
  int8_t  task_sub_id[MAX_SMS][MAX_TASKS_PER_SM];

  int16_t unsplit_qo_tile[MAX_UNSPLIT];
  int8_t  unsplit_head[MAX_UNSPLIT];
  int16_t unsplit_batch[MAX_UNSPLIT];
  int16_t unsplit_kv_iters[MAX_UNSPLIT];
  int     unsplit_sub_counter[MAX_UNSPLIT];
  int     num_unsplit;

  uint     warp_scratch[8];  // for block reduce
  typename cub::BlockScan<int, MAX_SMS>::TempStorage scan_temp;
};

// ============================================================================
// Parallel min/max cost SM: pack (cost, idx) into int32
// cost in upper 24 bits, idx in lower 8 bits → min/max on int32 works
// ============================================================================

__device__ __forceinline__ uint pack_cost_idx(uint cost, uint idx) {
  return (cost << 8) | (idx & 0xff);
}
__device__ __forceinline__ uint unpack_idx(uint packed) { return packed & 0xff; }

// Layout: [63:32] qo_tile_idx (32b) | [31:16] qo_head_idx (16b) | [15:0] batch_idx (16b)
// 64-bit so per-token sparse decode (batch_size = total_q, up to 65536) can address batch
// without wrap, while large prefill keeps qo_tile_idx headroom (max_qo_len / qo_tile_size
// can exceed 256 for q ≥ 64K). See bug_sparse_decode_qo256.md.
__device__ __forceinline__ uint64_t pack_work_info(int qo_tile, int head, int batch) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(qo_tile)) << 32) |
         (static_cast<uint64_t>(head & 0xFFFF) << 16) |
         static_cast<uint64_t>(batch & 0xFFFF);
}

__device__ __forceinline__ uint64_t pack_work_range(int start, int end) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(end)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(start));
}

// Strided -INFINITY fill of workspace_lse. Called AFTER the split/nosplit
// decision: split -> full lse_total_size; nosplit -> split-0 plane only
// (reduction's num_splits==1 fast path reads only plane 0).
__device__ __forceinline__ void init_workspace_lse(float* workspace_lse, int count) {
  for (int i = static_cast<int>(threadIdx.x); i < count; i += static_cast<int>(blockDim.x))
    workspace_lse[i] = -INFINITY;
}

template <bool IsMin>
__device__ __forceinline__ int block_reduce_cost(PlanSharedState& s, int n) {
  int tid = static_cast<int>(threadIdx.x);
  uint val = (tid < n)
      ? pack_cost_idx(s.sm_cost[tid], tid)
      : pack_cost_idx(IsMin ? 0x7fffff : 0, tid);

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    uint other = __shfl_xor_sync(0xffffffff, val, offset);
    val = IsMin ? ::min(val, other) : ::max(val, other);
  }

  if (tid % 32 == 0) s.warp_scratch[tid / 32] = val;
  __syncthreads();

  if (tid < 8) {
    val = (tid * 32 < n) ? s.warp_scratch[tid]
        : pack_cost_idx(IsMin ? 0x7fffff : 0, tid * 32);
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      uint other = __shfl_xor_sync(0xff, val, offset);
      val = IsMin ? ::min(val, other) : ::max(val, other);
    }
    if (tid == 0) s.warp_scratch[0] = val;
  }
  __syncthreads();
  return s.warp_scratch[0];
}

__device__ __forceinline__ int find_min_cost_sm(PlanSharedState& s, int n) {
  return unpack_idx(block_reduce_cost<true>(s, n));
}

__device__ __forceinline__ int find_max_cost_sm(PlanSharedState& s, int n) {
  return unpack_idx(block_reduce_cost<false>(s, n));
}

// ============================================================================
// Helpers (thread 0 only)
// ============================================================================

__device__ __forceinline__ int compute_kv_iters(
    int batch_idx, int qo_tile_idx, int* qo_lens, int* kv_lens,
    int* qo_offsets, int qo_tile_size, int kv_tile_size, bool causal,
    int pack_factor = 1) {
  int kv_len = kv_lens[batch_idx];
  if (!causal) return (kv_len + kv_tile_size - 1) / kv_tile_size;
  int offset_q = qo_offsets ? qo_offsets[batch_idx] : (kv_len - qo_lens[batch_idx]);
  int packed_q_end = (qo_tile_idx + 1) * qo_tile_size;
  int q_end = (pack_factor > 1) ? (packed_q_end - 1) / pack_factor + 1 : packed_q_end;
  int eff_kv = q_end + offset_q;
  if (eff_kv > kv_len) eff_kv = kv_len;
  if (eff_kv <= 0) return 0;
  return (eff_kv + kv_tile_size - 1) / kv_tile_size;
}

__device__ __forceinline__ void add_task(
    PlanSharedState& s, int sm,
    int16_t kv_begin, int16_t kv_end, int16_t unsplit_id, int8_t sub_id) {
  int idx = s.sm_task_count[sm]++;
  if (idx < MAX_TASKS_PER_SM) {
    s.task_kv_begin[sm][idx] = kv_begin;
    s.task_kv_end[sm][idx] = kv_end;
    s.task_unsplit_id[sm][idx] = unsplit_id;
    s.task_sub_id[sm][idx] = sub_id;
  }
}

__device__ __forceinline__ bool split_task(
    PlanSharedState& s, int sm, int task_idx, int keep_iters, int target_sm, int max_splits) {
  int16_t uid = s.task_unsplit_id[sm][task_idx];

  int old_count = atomicAdd(&s.unsplit_sub_counter[uid], 1);
  if (old_count >= max_splits) {
    atomicAdd(&s.unsplit_sub_counter[uid], -1);
    return false;
  }

  int16_t orig_begin = s.task_kv_begin[sm][task_idx];
  int16_t split_point = orig_begin + keep_iters;
  int16_t orig_end = s.task_kv_end[sm][task_idx];
  s.task_kv_end[sm][task_idx] = split_point;

  add_task(s, target_sm, split_point, orig_end, uid, (int8_t)old_count);
  return true;
}

__device__ __forceinline__ int find_best_fit_task(
    PlanSharedState& s, int sm, int target_iters, int max_splits, int min_iters) {
  int n = s.sm_task_count[sm];
  int best_fit = -1, best_fit_iters = 0x7fff;
  int largest = -1, largest_iters = 0;
  int min_needed = target_iters + min_iters;

  for (int t = 0; t < n; t++) {
    int iters = s.task_kv_end[sm][t] - s.task_kv_begin[sm][t];
    int16_t uid = s.task_unsplit_id[sm][t];
    if (s.unsplit_sub_counter[uid] >= max_splits) continue;
    if (iters < 2 * min_iters) continue;

    if (iters >= min_needed && iters < best_fit_iters) {
      best_fit = t;
      best_fit_iters = iters;
    }
    if (iters > largest_iters) {
      largest = t;
      largest_iters = iters;
    }
  }
  return best_fit >= 0 ? best_fit : largest;
}

// ============================================================================
// Parallel output writing: each thread writes its own SM's tasks
// All threads participate (prefix sum for write offsets)
// ============================================================================

__device__ void parallel_write_output(
    PlanSharedState& s, int num_buckets,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int* kv_tile_begin_indices = nullptr,
    int* kv_tile_end_indices = nullptr,
    int* kv_split_indices = nullptr,
    int* split_bounds = nullptr) {

  int tid = static_cast<int>(threadIdx.x);
  int my_count = (tid < num_buckets) ? min(static_cast<int>(s.sm_task_count[tid]), MAX_TASKS_PER_SM) : 0;

  if (split_bounds) {
    for (int u = tid; u < s.num_unsplit; u += static_cast<int>(blockDim.x))
      split_bounds[u] = 0;
  }
  int my_offset;
  cub::BlockScan<int, MAX_SMS>(s.scan_temp).ExclusiveSum(my_count, my_offset);
  __syncthreads();

  if (tid < num_buckets) {
    packed_work_range[tid] = pack_work_range(my_offset, my_offset + my_count);
    for (int t = 0; t < my_count; t++) {
      int pos = my_offset + t;
      int16_t uid = s.task_unsplit_id[tid][t];
      packed_work_info[pos] = pack_work_info(
          s.unsplit_qo_tile[uid], s.unsplit_head[uid], s.unsplit_batch[uid]);
      if (kv_tile_begin_indices) kv_tile_begin_indices[pos] = s.task_kv_begin[tid][t];
      if (kv_tile_end_indices) kv_tile_end_indices[pos] = s.task_kv_end[tid][t];
      int sub_id = static_cast<int>(s.task_sub_id[tid][t]);
      if (kv_split_indices) kv_split_indices[pos] = sub_id;
      if (split_bounds) atomicMax(&split_bounds[uid], sub_id + 1);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ int exact_cost_rank(PlanSharedState& s, int num_buckets);

__device__ __forceinline__ bool direct_nosplit_group_is_uniform(
    PlanSharedState& s, int num_buckets, int group_batches, int first_batch, int qt,
    int expected_iters, int tile_cost, int* qo_lens, int* kv_lens, int* qo_offsets,
    int qo_tile_size, int kv_tile_size, bool causal, int pack_factor) {
  int tid = static_cast<int>(threadIdx.x);
  int base_cost = s.sm_cost[0];
  uint invalid = (tid < num_buckets && s.sm_cost[tid] != base_cost) ? 1u : 0u;
  if (tile_cost <= 0 || static_cast<int64_t>(base_cost) + tile_cost > 0x7fffffffLL)
    invalid = 1;
  if (tid < group_batches) {
    int b = first_batch + tid;
    if (qt >= ceil_div(qo_lens[b], qo_tile_size)) {
      invalid = 1;
    } else if (compute_kv_iters(b, qt, qo_lens, kv_lens, qo_offsets,
                                qo_tile_size, kv_tile_size, causal, pack_factor) != expected_iters) {
      invalid = 1;
    }
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    invalid |= __shfl_xor_sync(0xffffffff, invalid, offset);
  if ((tid & 31) == 0) s.warp_scratch[tid >> 5] = invalid;
  __syncthreads();
  if (tid < 8) {
    invalid = s.warp_scratch[tid];
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
      invalid |= __shfl_xor_sync(0xff, invalid, offset);
    if (tid == 0) s.warp_scratch[0] = invalid;
  }
  __syncthreads();
  return s.warp_scratch[0] == 0;
}

// ============================================================================
// Direct nosplit: enumerate tiles + greedy assign + write output in two passes
// No smem unsplit table needed — for large tile counts (>MAX_UNSPLIT)
// ============================================================================

__device__ int direct_greedy(
    PlanSharedState& s, int num_buckets,
    int* qo_lens, int* kv_lens, int* qo_offsets,
    int qo_tile_size, int kv_tile_size, int batch_size, int num_heads, bool causal,
    int num_kv_splits,
    int min_iters,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int* kv_tile_begin_indices = nullptr, int* kv_tile_end_indices = nullptr,
    int* kv_split_indices = nullptr,
    int* num_kv_splits_per_row = nullptr, int* qo_segment_offsets = nullptr,
    int pack_factor = 1) {

  int tid = static_cast<int>(threadIdx.x);

  __shared__ int _max_qo_tiles;
  if (tid == 0) {
    int mq = 0;
    for (int b = 0; b < batch_size; b++) {
      int nt = ceil_div(qo_lens[b], qo_tile_size);
      if (nt > mq) mq = nt;
    }
    _max_qo_tiles = mq;
  }
  __syncthreads();
  int max_qo_tiles = _max_qo_tiles;
  int group_batches = 1;
  if (num_kv_splits == 1 && num_heads > 0 && num_heads < num_buckets &&
      num_buckets % num_heads == 0)
    group_batches = num_buckets / num_heads;

  if (tid < num_buckets) {
    s.sm_cost[tid] = kSMGlobalOverhead;
    s.sm_task_count[tid] = 0;
  }
  __syncthreads();

  // Pass 1: batch-assign heads in reverse QO tile order
  for (int qt = max_qo_tiles - 1; qt >= 0; qt--) {
    for (int b = 0; b < batch_size; b++) {
      if (qt >= ceil_div(qo_lens[b], qo_tile_size)) continue;
      int total_ki = compute_kv_iters(b, qt, qo_lens, kv_lens, qo_offsets,
                                       qo_tile_size, kv_tile_size, causal, pack_factor);
      if (total_ki <= 0) continue;
      int nosplit_tile_cost = static_cast<int>(kPerIterOverhead * total_ki + kTileGlobalOverhead);
      if (group_batches > 1 && group_batches <= batch_size - b &&
          direct_nosplit_group_is_uniform(s, num_buckets, group_batches, b, qt,
              total_ki, nosplit_tile_cost, qo_lens, kv_lens, qo_offsets,
              qo_tile_size, kv_tile_size, causal, pack_factor)) {
        int my_rank = tid;
        __syncthreads();
        if (tid < num_buckets && my_rank < num_buckets) {
          s.sm_cost[tid] += nosplit_tile_cost;
          s.sm_task_count[tid]++;
        }
        __syncthreads();
        b += group_batches - 1;
        continue;
      }
      int actual_splits = (num_kv_splits <= 1) ? 1
          : ::min(num_kv_splits, ::max(1, total_ki / min_iters));
      int split_size = (total_ki + actual_splits - 1) / actual_splits;
      if (actual_splits > 1)
        actual_splits = (total_ki + split_size - 1) / split_size;
      for (int sp = 0; sp < actual_splits; sp++) {
        int kb = sp * split_size;
        int ke = ::min(kb + split_size, total_ki);
        int piece_ki = ke - kb;
        int tile_cost = static_cast<int>(kPerIterOverhead * piece_ki + kTileGlobalOverhead);

        for (int h_off = 0; h_off < num_heads; ) {
          int batch_count = num_heads - h_off;
          if (batch_count > num_buckets) batch_count = num_buckets;

          int my_rank = exact_cost_rank(s, num_buckets);
          bool i_get = (tid < num_buckets) && (my_rank < batch_count);
          __syncthreads();

          if (i_get) {
            s.sm_cost[tid] += tile_cost;
            s.sm_task_count[tid]++;
          }
          __syncthreads();
          h_off += batch_count;
        }
      }
    }
  }

  int max_sm = find_max_cost_sm(s, num_buckets);
  int max_cost = s.sm_cost[max_sm];

  // Compute work_indptr, reset for pass 2
  __shared__ int _work_offsets[MAX_SMS + 1];
  if (threadIdx.x == 0) {
    int pos = 0;
    for (int sm = 0; sm < num_buckets; sm++) {
      _work_offsets[sm] = pos;
      pos += s.sm_task_count[sm];
    }
    _work_offsets[num_buckets] = pos;
    for (int sm = 0; sm < num_buckets; sm++)
      s.sm_task_count[sm] = 0;
  }
  __syncthreads();

  // Pass 2: same order, batch-assign + write tile info
  if (tid < num_buckets) s.sm_cost[tid] = kSMGlobalOverhead;
  __syncthreads();

  for (int qt = max_qo_tiles - 1; qt >= 0; qt--) {
    for (int b = 0; b < batch_size; b++) {
      if (qt >= ceil_div(qo_lens[b], qo_tile_size)) continue;
      int total_ki = compute_kv_iters(b, qt, qo_lens, kv_lens, qo_offsets,
                                       qo_tile_size, kv_tile_size, causal, pack_factor);
      if (total_ki <= 0) continue;
      int nosplit_tile_cost = static_cast<int>(kPerIterOverhead * total_ki + kTileGlobalOverhead);
      if (group_batches > 1 && group_batches <= batch_size - b &&
          direct_nosplit_group_is_uniform(s, num_buckets, group_batches, b, qt,
              total_ki, nosplit_tile_cost, qo_lens, kv_lens, qo_offsets,
              qo_tile_size, kv_tile_size, causal, pack_factor)) {
        int my_rank = tid;
        __syncthreads();
        if (tid < num_buckets && my_rank < num_buckets) {
          int task_b = b + my_rank / num_heads;
          int head = my_rank % num_heads;
          s.sm_cost[tid] += nosplit_tile_cost;
          int pos = _work_offsets[tid] + s.sm_task_count[tid]++;
          packed_work_info[pos] = pack_work_info(qt, head, task_b);
          if (kv_tile_begin_indices) kv_tile_begin_indices[pos] = 0;
          if (kv_tile_end_indices) kv_tile_end_indices[pos] = total_ki;
          if (kv_split_indices) kv_split_indices[pos] = 0;
        }
        if (tid < group_batches && num_kv_splits_per_row && qo_segment_offsets) {
          int task_b = b + tid;
          int seg_off = qo_segment_offsets[task_b];
          int row_start = qt * qo_tile_size;
          int row_end = ::min(row_start + qo_tile_size, qo_lens[task_b]);
          for (int r = row_start; r < row_end; r++)
            num_kv_splits_per_row[seg_off + r] = 1;
        }
        __syncthreads();
        b += group_batches - 1;
        continue;
      }
      int actual_splits = (num_kv_splits <= 1) ? 1
          : ::min(num_kv_splits, ::max(1, total_ki / min_iters));
      int split_size = (total_ki + actual_splits - 1) / actual_splits;
      if (actual_splits > 1)
        actual_splits = (total_ki + split_size - 1) / split_size;
      if (threadIdx.x == 0 && num_kv_splits_per_row && qo_segment_offsets) {
        int seg_off = qo_segment_offsets[b];
        int qo_len = qo_lens[b];
        int row_start = qt * qo_tile_size;
        int row_end = row_start + qo_tile_size < qo_len ? row_start + qo_tile_size : qo_len;
        for (int r = row_start; r < row_end; r++)
          num_kv_splits_per_row[seg_off + r] = actual_splits;
      }
      for (int sp = 0; sp < actual_splits; sp++) {
        int kb = sp * split_size;
        int ke = ::min(kb + split_size, total_ki);
        int piece_ki = ke - kb;
        int tile_cost = static_cast<int>(kPerIterOverhead * piece_ki + kTileGlobalOverhead);

        for (int h_off = 0; h_off < num_heads; ) {
          int batch_count = num_heads - h_off;
          if (batch_count > num_buckets) batch_count = num_buckets;

          int my_rank = exact_cost_rank(s, num_buckets);
          bool i_get = (tid < num_buckets) && (my_rank < batch_count);
          __syncthreads();

          if (i_get) {
            s.sm_cost[tid] += tile_cost;
            int pos = _work_offsets[tid] + s.sm_task_count[tid]++;
            packed_work_info[pos] = pack_work_info(qt, h_off + my_rank, b);
            if (kv_tile_begin_indices) kv_tile_begin_indices[pos] = kb;
            if (kv_tile_end_indices) kv_tile_end_indices[pos] = ke;
            if (kv_split_indices) kv_split_indices[pos] = sp;
          }
          __syncthreads();
          h_off += batch_count;
        }
      }
    }
  }
  __syncthreads();
  if (tid < num_buckets) {
    packed_work_range[tid] = pack_work_range(
        _work_offsets[tid], _work_offsets[tid] + s.sm_task_count[tid]);
  }
  __syncthreads();
  return max_cost;
}

__device__ void enumerate_unsplit_tiles(
    PlanSharedState& s, int* qo_lens, int* kv_lens, int* qo_offsets,
    int qo_tile_size, int kv_tile_size, int batch_size, int num_heads, bool causal,
    int pack_factor = 1) {
  int tid = static_cast<int>(threadIdx.x);
  int max_qt = 0;
  for (uint b = static_cast<uint>(tid); b < static_cast<uint>(batch_size); b += MAX_SMS)
    max_qt = ::max(max_qt, ceil_div(qo_lens[b], qo_tile_size));
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    max_qt = ::max(max_qt, __shfl_xor_sync(0xffffffff, max_qt, offset));
  if ((tid & 31) == 0) s.warp_scratch[tid >> 5] = static_cast<uint>(max_qt);
  __syncthreads();
  if (tid < 8) {
    max_qt = static_cast<int>(s.warp_scratch[tid]);
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
      max_qt = ::max(max_qt, __shfl_xor_sync(0xff, max_qt, offset));
    if (tid == 0) s.warp_scratch[0] = static_cast<uint>(max_qt);
  }
  __syncthreads();

  int qt_base = static_cast<int>(s.warp_scratch[0]) - 1;
  uint batch_base = 0;
  int uid = 0;
  __syncthreads();
  if (tid == 0) s.warp_scratch[1] = 0;
  __syncthreads();
  while (qt_base >= 0 && uid < MAX_UNSPLIT) {
    uint batch_pos = batch_base + static_cast<uint>(tid);
    int qt = qt_base - static_cast<int>(batch_pos / static_cast<uint>(batch_size));
    int b = static_cast<int>(batch_pos % static_cast<uint>(batch_size));
    int ki = 0;
    if (qt >= 0 && qt < ceil_div(qo_lens[b], qo_tile_size))
      ki = compute_kv_iters(b, qt, qo_lens, kv_lens, qo_offsets,
                            qo_tile_size, kv_tile_size, causal, pack_factor);
    int count = ki > 0 ? ::min(num_heads, MAX_UNSPLIT) : 0;
    int prefix, total;
    cub::BlockScan<int, MAX_SMS>(s.scan_temp).ExclusiveSum(count, prefix, total);
    int begin = uid + prefix;
    int kept = ::min(count, ::max(0, MAX_UNSPLIT - begin));
    if (kept < count) atomicOr(&s.warp_scratch[1], 1u);
    for (int h = 0; h < kept; h++) {
      s.unsplit_qo_tile[begin + h] = qt;
      s.unsplit_head[begin + h] = h;
      s.unsplit_batch[begin + h] = b;
      s.unsplit_kv_iters[begin + h] = ki;
      s.unsplit_sub_counter[begin + h] = 0;
    }
    __syncthreads();
    uid += ::min(total, MAX_UNSPLIT - uid);
    uint next_batch = batch_base + MAX_SMS;
    qt_base -= static_cast<int>(next_batch / static_cast<uint>(batch_size));
    batch_base = next_batch % static_cast<uint>(batch_size);
  }
  if (qt_base >= 0) atomicOr(&s.warp_scratch[1], 1u);
  if (tid == 0) s.num_unsplit = uid;
}

// ============================================================================
// LPT sort: insertion sort unsplit tiles by kv_iters descending (thread 0 only)
// After reverse-order enumeration the table is nearly sorted, so O(n) expected.
// ============================================================================

__device__ void sort_unsplit_descending(PlanSharedState& s) {
  int n = s.num_unsplit;
  for (int i = 1; i < n; i++) {
    int16_t ki = s.unsplit_kv_iters[i];
    if (ki <= s.unsplit_kv_iters[i - 1]) continue;
    int16_t qt = s.unsplit_qo_tile[i];
    int8_t  h  = s.unsplit_head[i];
    int16_t b  = s.unsplit_batch[i];
    int j = i - 1;
    while (j >= 0 && s.unsplit_kv_iters[j] < ki) {
      s.unsplit_kv_iters[j + 1] = s.unsplit_kv_iters[j];
      s.unsplit_qo_tile[j + 1] = s.unsplit_qo_tile[j];
      s.unsplit_head[j + 1]    = s.unsplit_head[j];
      s.unsplit_batch[j + 1]   = s.unsplit_batch[j];
      j--;
    }
    s.unsplit_kv_iters[j + 1] = ki;
    s.unsplit_qo_tile[j + 1]  = qt;
    s.unsplit_head[j + 1]     = h;
    s.unsplit_batch[j + 1]    = b;
  }
}

__device__ __forceinline__ int exact_cost_rank(PlanSharedState& s, int num_buckets) {
  using RankSort = cub::BlockRadixSort<uint64_t, MAX_SMS, 1>;
  __shared__ typename RankSort::TempStorage rank_sort_temp;
  __shared__ int rank_by_bucket[MAX_SMS];
  static_assert(MAX_SMS == 256, "The low 8 rank-key bits encode the bucket ID");
  int tid = static_cast<int>(threadIdx.x);
  uint64_t keys[1];
  keys[0] = ~uint64_t(0);
  if (tid < num_buckets) {
    uint ordered_cost = static_cast<uint>(s.sm_cost[tid]) ^ 0x80000000u;
    keys[0] = (static_cast<uint64_t>(ordered_cost) << 8) | static_cast<uint64_t>(tid);
  }
  RankSort(rank_sort_temp).Sort(keys, 8, 40);
  if (tid < num_buckets) rank_by_bucket[keys[0] & 0xff] = tid;
  __syncthreads();
  return tid < num_buckets ? rank_by_bucket[tid] : 0;
}

// ============================================================================
// Phase 0: Nosplit greedy with batch assignment
// Tiles with same kv_iters are assigned in bulk via parallel rank computation.
// AssignTasks: populate smem task arrays (for subsequent rebalance)
// WriteGlobal: write final output to global memory
// ============================================================================

template <bool AssignTasks, bool WriteGlobal>
__device__ int phase0_nosplit_greedy(
    PlanSharedState& s, int num_buckets, int num_heads,
    uint64_t* packed_work_range, uint64_t* packed_work_info) {

  int tid = static_cast<int>(threadIdx.x);
  if (tid < num_buckets) {
    s.sm_cost[tid] = kSMGlobalOverhead;
    s.sm_task_count[tid] = 0;
  }
  __syncthreads();

  int n = s.num_unsplit;
  int step = num_heads > 0 ? num_heads : 1;
  int u = 0;
  while (u < n) {
    int ki = s.unsplit_kv_iters[u];
    int group_end = u + step;
    while (group_end < n && s.unsplit_kv_iters[group_end] == ki)
      group_end += step;
    if (group_end > n) group_end = n;
    int tile_cost = static_cast<int>(kPerIterOverhead * ki + kTileGlobalOverhead);

    while (u < group_end) {
      int batch_count = group_end - u;
      if (batch_count > num_buckets) batch_count = num_buckets;

      int my_rank = exact_cost_rank(s, num_buckets);

      bool i_get_tile = (tid < num_buckets) && (my_rank < batch_count);
      __syncthreads();

      if (i_get_tile) {
        s.sm_cost[tid] += tile_cost;
        if constexpr (AssignTasks) {
          int tile_u = u + my_rank;
          add_task(s, tid, 0, ki, tile_u, 0);
          s.unsplit_sub_counter[tile_u] = 1;
        }
      }
      __syncthreads();
      u += batch_count;
    }
  }

  int max_sm = find_max_cost_sm(s, num_buckets);
  int max_cost = s.sm_cost[max_sm];

  if constexpr (WriteGlobal) {
    parallel_write_output(s, num_buckets,
        packed_work_range, packed_work_info);
  }
  __syncthreads();
  return max_cost;
}

// ============================================================================
// Phase 1: Linear scan fill with LPT-sorted input (thread 0 only)
// ============================================================================

__device__ void phase1_linear_fill(
    PlanSharedState& s, int num_buckets, int max_splits, int min_iters) {

  int tid = static_cast<int>(threadIdx.x);
  if (tid < num_buckets) {
    s.sm_cost[tid] = kSMGlobalOverhead;
    s.sm_task_count[tid] = 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int n = s.num_unsplit;
    int total_iters = 0;
    for (int u = 0; u < n; u++)
      total_iters += s.unsplit_kv_iters[u];

    int avg_iters = (total_iters + num_buckets - 1) / num_buckets;
    int target_budget = kTileGlobalOverhead + avg_iters * kPerIterOverhead * 1.02;
    int cur_sm = 0;
    int sm_budget = target_budget;
    int min_piece_cost = kTileGlobalOverhead + min_iters * kPerIterOverhead;

    for (int u = 0; u < n; u++) {
      int tile_iters = s.unsplit_kv_iters[u];
      if (tile_iters <= 0) { s.unsplit_sub_counter[u] = 1; continue; }

      int tile_pos = 0;
      int sub_id = 0;

      while (tile_pos < tile_iters) {
        while (cur_sm < num_buckets && s.sm_task_count[cur_sm] >= MAX_TASKS_PER_SM) {
          cur_sm++;
          sm_budget = target_budget;
        }
        if (cur_sm >= num_buckets) {
          int best_sm = 0;
          for (int i = 1; i < num_buckets; i++)
            if (s.sm_task_count[i] < s.sm_task_count[best_sm]) best_sm = i;
          if (s.sm_task_count[best_sm] >= MAX_TASKS_PER_SM) break;
          cur_sm = best_sm;
          sm_budget = 0x7fffffff;
        }

        int avail = tile_iters - tile_pos;
        int budget_iters = (sm_budget - kTileGlobalOverhead) / kPerIterOverhead;
        if (budget_iters < min_iters) budget_iters = min_iters;
        // Check whether budget-based sizing would create more pieces than
        // max_splits allows. If so, the old code took ALL remaining iters as
        // one giant piece (line: if (sub_id + 1 >= max_splits) max_iters = avail),
        // producing severe span imbalance. Instead, distribute the remaining
        // iters evenly across the remaining allowed splits via floor division
        // so that every piece — including the last — has comparable size.
        int budget_pieces = (tile_iters + budget_iters - 1) / budget_iters;
        int max_iters;
        if (budget_pieces > max_splits) {
          int remaining_splits = max_splits - sub_id;
          max_iters = (remaining_splits > 0) ? (avail / remaining_splits) : avail;
          if (max_iters < min_iters) max_iters = min_iters;
        } else {
          max_iters = budget_iters;
        }
        int take = avail < max_iters ? avail : max_iters;

        add_task(s, cur_sm, tile_pos, tile_pos + take, u, sub_id++);
        int piece_cost = kPerIterOverhead * take + kTileGlobalOverhead;
        s.sm_cost[cur_sm] += piece_cost;
        sm_budget -= piece_cost;
        tile_pos += take;

        if (sm_budget < min_piece_cost) {
          cur_sm++;
          sm_budget = target_budget;
        }
      }
      s.unsplit_sub_counter[u] = sub_id;
    }
  }
  __syncthreads();
}

// ============================================================================
// Phase 2 (parallel): all threads participate, each handles its own SM
// Excess/deficit SMs paired via prefix sum, all splits happen simultaneously
// ============================================================================

// Returns true iff any task was actually moved/split (task tables modified).
__device__ bool phase2_rebalance_parallel(
    PlanSharedState& s, int num_buckets, int max_splits, int min_iters) {

  int tid = static_cast<int>(threadIdx.x);
  __shared__ int16_t _deficit_sms[MAX_SMS];
  __shared__ int _n_deficit;
  __shared__ int _gap;
  __shared__ int _avg_cost;
  __shared__ uint _warp_min[8];
  __shared__ uint _warp_max[8];
  __shared__ int _warp_sum[8];
  __shared__ int _dirty;
  if (tid == 0) _dirty = 0;
  __syncthreads();

  constexpr int MAX_PARALLEL_ITERS = 8;
  for (int iter = 0; iter < MAX_PARALLEL_ITERS; iter++) {
    int my_cost = (tid < num_buckets) ? s.sm_cost[tid] : 0;
    uint min_key = pack_cost_idx(tid < num_buckets ? my_cost : 0x7fffff, tid);
    uint max_key = pack_cost_idx(my_cost, tid);
    int sum_cost = my_cost;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      min_key = ::min(min_key, __shfl_xor_sync(0xffffffff, min_key, offset));
      max_key = ::max(max_key, __shfl_xor_sync(0xffffffff, max_key, offset));
      sum_cost += __shfl_xor_sync(0xffffffff, sum_cost, offset);
    }
    if ((tid & 31) == 0) {
      _warp_min[tid >> 5] = min_key;
      _warp_max[tid >> 5] = max_key;
      _warp_sum[tid >> 5] = sum_cost;
    }
    __syncthreads();
    if (tid < 8) {
      min_key = (tid * 32 < num_buckets) ? _warp_min[tid]
                                           : pack_cost_idx(0x7fffff, tid * 32);
      max_key = (tid * 32 < num_buckets) ? _warp_max[tid]
                                           : pack_cost_idx(0, tid * 32);
      sum_cost = (tid * 32 < num_buckets) ? _warp_sum[tid] : 0;
      #pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1) {
        min_key = ::min(min_key, __shfl_xor_sync(0xff, min_key, offset));
        max_key = ::max(max_key, __shfl_xor_sync(0xff, max_key, offset));
        sum_cost += __shfl_xor_sync(0xff, sum_cost, offset);
      }
      if (tid == 0) {
        _gap = s.sm_cost[unpack_idx(max_key)] - s.sm_cost[unpack_idx(min_key)];
        _avg_cost = sum_cost / num_buckets;
      }
    }
    __syncthreads();
    if (_gap <= kTileGlobalOverhead) break;
    int avg_cost = _avg_cost;

    static_assert(MAX_SMS < (1 << 15), "Packed prefix counts must fit in a positive int");
    int is_deficit = (tid < num_buckets && my_cost < avg_cost) ? 1 : 0;
    int is_excess = (tid < num_buckets && my_cost > avg_cost) ? 1 : 0;
    int packed_prefix;
    cub::BlockScan<int, MAX_SMS>(s.scan_temp).ExclusiveSum(
        is_deficit | (is_excess << 16), packed_prefix);
    int deficit_idx = packed_prefix & 0xffff;
    int excess_idx = packed_prefix >> 16;
    __syncthreads();
    if (is_deficit) _deficit_sms[deficit_idx] = tid;
    if (tid == num_buckets - 1) _n_deficit = deficit_idx + is_deficit;
    __syncthreads();
    if (_n_deficit == 0) break;

    // Each excess thread independently splits/moves to its unique deficit target
    if (is_excess && excess_idx < _n_deficit) {
      int sm = tid;
      int target_sm = _deficit_sms[excess_idx];
      int n_tasks = s.sm_task_count[sm];
      int sm_gap = s.sm_cost[sm] - s.sm_cost[target_sm];

      // Strategy 1: move a whole task if it reduces the pair gap
      int best_move = -1;
      int best_new_gap = sm_gap;
      for (int t = 0; t < n_tasks; t++) {
        int iters = s.task_kv_end[sm][t] - s.task_kv_begin[sm][t];
        int task_cost = static_cast<int>(kPerIterOverhead * iters + kTileGlobalOverhead);
        int new_src = s.sm_cost[sm] - task_cost;
        int new_dst = s.sm_cost[target_sm] + task_cost;
        if (new_src >= new_dst && (new_src - new_dst) < best_new_gap) {
          best_move = t;
          best_new_gap = new_src - new_dst;
        }
      }

      if (best_move >= 0) {
        int16_t kb = s.task_kv_begin[sm][best_move];
        int16_t ke = s.task_kv_end[sm][best_move];
        int16_t uid = s.task_unsplit_id[sm][best_move];
        int8_t sid = s.task_sub_id[sm][best_move];
        int task_cost = static_cast<int>(kPerIterOverhead * (ke - kb) + kTileGlobalOverhead);
        int last = s.sm_task_count[sm] - 1;
        if (best_move != last) {
          s.task_kv_begin[sm][best_move] = s.task_kv_begin[sm][last];
          s.task_kv_end[sm][best_move] = s.task_kv_end[sm][last];
          s.task_unsplit_id[sm][best_move] = s.task_unsplit_id[sm][last];
          s.task_sub_id[sm][best_move] = s.task_sub_id[sm][last];
        }
        s.sm_task_count[sm]--;
        add_task(s, target_sm, kb, ke, uid, sid);
        s.sm_cost[sm] -= task_cost;
        s.sm_cost[target_sm] += task_cost;
        _dirty = 1;
      } else {
        // Strategy 2: split
        int move_iters = static_cast<int>((sm_gap - kTileGlobalOverhead) / (2 * kPerIterOverhead));
        if (move_iters >= min_iters) {
          int best = find_best_fit_task(s, sm, move_iters, max_splits, min_iters);
          if (best >= 0) {
            int task_iters = s.task_kv_end[sm][best] - s.task_kv_begin[sm][best];
            int keep = task_iters - move_iters;
            if (keep < min_iters) keep = min_iters;
            int actual_move = task_iters - keep;
            if (actual_move >= min_iters) {
              if (split_task(s, sm, best, keep, target_sm, max_splits)) {
              s.sm_cost[sm] -= static_cast<int>(actual_move * kPerIterOverhead);
              s.sm_cost[target_sm] += static_cast<int>(actual_move * kPerIterOverhead + kTileGlobalOverhead);
              _dirty = 1;
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }
  return _dirty != 0;
}

// ============================================================================
// Phase 3: Write split output (all threads for parallel write + split recount)
//
// num_kv_splits_per_row must cover every workspace slot the mainloop writes.
// The mainloop writes lse/O at slot = task's sub_id VALUE (kv_split_indices),
// so the reduction's read bound is max(sub_id)+1 over the uid's KEPT tasks.
// The phase0/1/2 sub_counter cannot be trusted: add_task silently drops tasks
// once an SM hits MAX_TASKS_PER_SM (counter already bumped), and phase2
// rebalance moves/splits reshuffle sub_ids — counting tasks UNDER-covers a
// kept task with a high sub_id (its contribution would be silently dropped),
// while the old counter max-update OVER-covers (declared-but-never-written
// ghost slots whose stale bytes poison the merge when the workspace is
// reused). Recounting max(sub_id)+1 from the FINAL task table (the same
// table parallel_write_output just emitted) is the tight exact bound.
// ============================================================================

__device__ void phase3_write_output(
    PlanSharedState& s, int num_buckets, int num_heads,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int* kv_tile_begin_indices, int* kv_tile_end_indices,
    int* kv_split_indices, int* num_kv_splits_per_row,
    int* qo_segment_offsets, int* qo_lens, int qo_tile_size) {

  bool recount_nk = num_kv_splits_per_row && qo_segment_offsets && qo_lens;
  parallel_write_output(s, num_buckets,
      packed_work_range, packed_work_info,
      kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
      recount_nk ? s.unsplit_sub_counter : nullptr);

  if (recount_nk) {
    int tid = static_cast<int>(threadIdx.x);

    // --- Block-reduce max bound: skip the row write entirely when nosplit -
    // The prologue already initialized every row to 1 at kernel start (and
    // re-inits on every graph replay), so max<=1 means nothing to write.
    __shared__ int s_warp_max2[8];
    __shared__ int s_max_bound;
    int my_max2 = 0;
    for (int u = tid; u < s.num_unsplit; u += static_cast<int>(blockDim.x))
      my_max2 = ::max(my_max2, s.unsplit_sub_counter[u]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      my_max2 = ::max(my_max2, __shfl_xor_sync(0xffffffff, my_max2, offset));
    if ((tid & 31) == 0) s_warp_max2[tid >> 5] = my_max2;
    __syncthreads();
    if (tid == 0) {
      int nw = (static_cast<int>(blockDim.x) + 31) >> 5;
      int mx = s_warp_max2[0];
      for (int w = 1; w < nw; w++) mx = ::max(mx, s_warp_max2[w]);
      s_max_bound = mx;
    }
    __syncthreads();

    // --- Write nk from the recount (max across heads sharing a row) ----
    // Falls back to the serial max-merge when grouping is not guaranteed
    // (short final group from the MAX_UNSPLIT clamp).
    // Holes below the bound keep lse == -INFINITY (plan prologue memset),
    // so the reduction weights them 0 and its w==0 guard skips their stale O.
    if (s_max_bound > 1) {
      int H = num_heads;
      if (H > 0 && s.num_unsplit % H == 0 && s.num_unsplit < MAX_UNSPLIT) {
        int n_groups = s.num_unsplit / H;
        int lane = tid & 31;
        int warp = tid >> 5;
        int num_warps = static_cast<int>(blockDim.x) >> 5;
        for (int g = warp; g < n_groups; g += num_warps) {
          int u0 = g * H;
          int bound = 1;
          if (lane == 0) {
            for (int h = 0; h < H; h++)
              bound = ::max(bound, s.unsplit_sub_counter[u0 + h]);
          }
          bound = __shfl_sync(0xffffffff, bound, 0);
          int b  = s.unsplit_batch[u0];
          int qt = s.unsplit_qo_tile[u0];
          int seg_off = qo_segment_offsets[b];
          int qo_len = qo_lens[b];
          int row_start = qt * qo_tile_size;
          int row_end = ::min(row_start + qo_tile_size, qo_len);
          for (int r = row_start + lane; r < row_end; r += 32)
            num_kv_splits_per_row[seg_off + r] = bound;
        }
      } else if (tid == 0) {
        int n = s.num_unsplit;
        for (int u = 0; u < n; u++) {
          int b = s.unsplit_batch[u];
          int qt = s.unsplit_qo_tile[u];
          int splits = s.unsplit_sub_counter[u];
          int seg_off = qo_segment_offsets[b];
          int qo_len = qo_lens[b];
          int row_start = qt * qo_tile_size;
          int row_end = ::min(row_start + qo_tile_size, qo_len);
          for (int r = row_start; r < row_end; r++) {
            int cur = num_kv_splits_per_row[seg_off + r];
            if (splits > cur) num_kv_splits_per_row[seg_off + r] = splits;
          }
        }
      }
    }
  }
  __syncthreads();
}

// ============================================================================
// Main plan kernel
// ============================================================================

__global__ void plan_kernel(
    int* qo_segment_offsets, int* qo_lens, int* kv_lens,
    uint64_t* packed_work_range, uint64_t* packed_work_info,
    int qo_tile_size, int kv_tile_size, int batch_size, int num_heads,
    int num_buckets, bool causal, int* qo_offsets,
    int num_kv_splits, int* kv_tile_begin_indices, int* kv_tile_end_indices,
    int* kv_split_indices, int adaptive_chunk_size, float* out_max_sm_cost,
    int* num_kv_splits_per_row,
    float* workspace_lse = nullptr, int lse_total_size = 0,
    int pack_factor = 1, bool is_sparse = false) {

  extern __shared__ char __plan_smem[];
  PlanSharedState& state = *reinterpret_cast<PlanSharedState*>(__plan_smem);

  if (num_kv_splits_per_row && qo_segment_offsets) {
    int total_rows = qo_segment_offsets[batch_size];
    for (int i = static_cast<int>(threadIdx.x); i < total_rows; i += static_cast<int>(blockDim.x))
      num_kv_splits_per_row[i] = 1;
    __syncthreads();
  }
  // workspace_lse init is deferred to AFTER the split/nosplit decision
  // (nosplit initializes only the split-0 plane).

  bool adaptive_mode = (num_kv_splits < 0);
  int max_splits = adaptive_mode ? -num_kv_splits : num_kv_splits;
  // Base floor (dense=1, sparse=1). The sparse adaptive path below may coarsen
  // it to page-level for the tiniest workloads only; dense never changes.
  int min_iters = is_sparse ? MIN_ITERS_PER_SPLIT_SPARSE : MIN_ITERS_PER_SPLIT_DENSE;

  if (!adaptive_mode) {
    int cost = direct_greedy(state, num_buckets,
        qo_lens, kv_lens, qo_offsets, qo_tile_size, kv_tile_size,
        batch_size, num_heads, causal, num_kv_splits, min_iters,
        packed_work_range, packed_work_info,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        num_kv_splits_per_row, qo_segment_offsets, pack_factor);
    if (workspace_lse && num_kv_splits > 1) init_workspace_lse(workspace_lse, lse_total_size);
    if (threadIdx.x == 0 && out_max_sm_cost) {
      out_max_sm_cost[0] = static_cast<float>(cost);
      out_max_sm_cost[1] = static_cast<float>(cost);
    }
  } else {
    // Adaptive split path
    enumerate_unsplit_tiles(state, qo_lens, kv_lens, qo_offsets,
                            qo_tile_size, kv_tile_size, batch_size, num_heads, causal,
                            pack_factor);
    __syncthreads();

    if (state.warp_scratch[1] != 0u) {
      int cost = direct_greedy(state, num_buckets,
          qo_lens, kv_lens, qo_offsets, qo_tile_size, kv_tile_size,
          batch_size, num_heads, causal, /*num_kv_splits=*/1, min_iters,
          packed_work_range, packed_work_info,
          kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
          num_kv_splits_per_row, qo_segment_offsets, pack_factor);
      if (workspace_lse) init_workspace_lse(workspace_lse, lse_total_size);
      if (threadIdx.x == 0 && out_max_sm_cost) {
        out_max_sm_cost[0] = 0.f;
        out_max_sm_cost[1] = 1.f;
      }
      (void)cost;
      return;
    }

    // Device-side adaptive min_iters, SPARSE-ONLY: coarsen split to page-level
    // (2 tiles of 64 = 1 page of 128 -> 16 splits) only for the tiniest
    // workloads (avg_iters <= thresh, where budget_iters would collapse to 1 =
    // 32 splits, doubling reduction depth vs Triton's page-level 16). dense
    // skips this block entirely (min_iters stays DENSE=1). avg_iters and the
    // decision are device-side -> CUDA-graph-safe (plan kernel is captured in
    // the graph; phase1 clamps pieces to max_splits so the capacity-sized
    // workspace never overflows). See
    // docs/msa_sparse_decode_page_split_coarsen_spec.md.
    if (is_sparse) {  // uniform kernel arg -> conditional __syncthreads is safe
      __shared__ int s_min_iters;
      if (threadIdx.x == 0) {
        int total_iters = 0;
        for (int u = 0; u < state.num_unsplit; u++) total_iters += state.unsplit_kv_iters[u];
        int avg_iters = (total_iters + num_buckets - 1) / num_buckets;
        s_min_iters = (avg_iters <= SPARSE_COARSE_AVG_ITERS_THRESH)
                    ? MIN_ITERS_PER_SPLIT_SPARSE_COARSE
                    : MIN_ITERS_PER_SPLIT_SPARSE;
      }
      __syncthreads();
      min_iters = s_min_iters;
    }

    // Greedy nosplit: assign tasks + return cost. The gmem write is dropped:
    // phase3_write_output always (re)writes the final output, so the
    // WriteGlobal=true pass here was never consumed.
    int nosplit_cost = phase0_nosplit_greedy<true, false>(
        state, num_buckets, num_heads,
        packed_work_range, packed_work_info);

    // Quick check: can split possibly beat nosplit?
    // Best-case split cost = total_cost/M + kSplitExtraOverhead
    // If that already >= nosplit_cost * threshold, skip rebalance entirely
    int min_sm = find_min_cost_sm(state, num_buckets);
    int min_cost = state.sm_cost[min_sm];
    int gap = nosplit_cost - min_cost;

    bool skip_rebalance = (gap <= kSplitExtraOverhead);
    bool tables_dirty = false;
    if (!skip_rebalance) {
      if (state.num_unsplit < num_buckets * 0.8) {
        phase1_linear_fill(state, num_buckets, max_splits, min_iters);
        tables_dirty = true;  // phase1 always rewrites the task tables
      }
      tables_dirty = phase2_rebalance_parallel(state, num_buckets, max_splits, min_iters)
                     || tables_dirty;
    }

    int split_max_sm = find_max_cost_sm(state, num_buckets);
    int split_cost = state.sm_cost[split_max_sm] + kSplitExtraOverhead;

    bool split = false;
    if (split_cost < nosplit_cost * 0.99) {
      split = true;
    }
    if (!split && tables_dirty) {
      // Nosplit decided AND the task tables were modified: re-run phase0 to
      // restore nosplit task assignment. Skipped when phase1 did not run and
      // phase2 made no move/split — the tables still hold the phase0 result.
      phase0_nosplit_greedy<true, false>(state, num_buckets, num_heads,
                                         nullptr, nullptr);
    }
    // Deferred workspace_lse init: full when split, split-0 plane when nosplit.
    if (workspace_lse) {
      if (split) init_workspace_lse(workspace_lse, lse_total_size);
      else init_workspace_lse(workspace_lse, qo_segment_offsets[batch_size] * num_heads);
    }
    // Always write output (kv_tile_begin/end/split_indices + num_kv_splits_per_row)
    // so that the split variant run kernel can read them regardless of split/nosplit decision.
    phase3_write_output(state, num_buckets, num_heads,
                        packed_work_range, packed_work_info,
                        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
                        num_kv_splits_per_row, qo_segment_offsets, qo_lens, qo_tile_size);
    if (threadIdx.x == 0 && out_max_sm_cost) {
      out_max_sm_cost[0] = split ? 1 : 0;
      out_max_sm_cost[1] = nosplit_cost / float(split_cost);
    }
  }
}

// ============================================================================
// Host wrapper (unchanged signature)
// ============================================================================

hggcError_t plan_kernel_wrapper(int* qo_segment_offsets, int* qo_lens,
                                int* kv_lens, uint64_t* packed_work_range,
                                uint64_t* packed_work_info, int qo_tile_size,
                                int kv_tile_size, int batch_size, int num_heads, int num_buckets,
                                bool causal, int* qo_offsets, bool enable_pdl,
                                hggcStream_t stream,
                                int num_kv_splits = 1,
                                int* kv_tile_begin_indices = nullptr,
                                int* kv_tile_end_indices = nullptr,
                                int* kv_split_indices = nullptr,
                                int adaptive_chunk_size = 0,
                                float* out_max_sm_cost = nullptr,
                                int* num_kv_splits_per_row = nullptr,
                                float* workspace_lse = nullptr,
                                int lse_total_size = 0,
                                int pack_factor = 1, bool is_sparse = false) {
  int smem_size = sizeof(PlanSharedState);
  static bool smem_attr_set = false;
  if (!smem_attr_set && smem_size >= (48 << 10)) {
    hggcFuncSetAttribute(plan_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    smem_attr_set = true;
  }
  plan_kernel<<<1, MAX_SMS, smem_size, stream>>>(
      qo_segment_offsets, qo_lens, kv_lens,
      packed_work_range, packed_work_info,
      qo_tile_size, kv_tile_size, batch_size, num_heads,
      num_buckets, causal, qo_offsets, num_kv_splits,
      kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
      adaptive_chunk_size, out_max_sm_cost, num_kv_splits_per_row,
      workspace_lse, lse_total_size, pack_factor, is_sparse);
  FLASHINFER_CUDA_CALL(hggcGetLastError());
  return hggcSuccess;
}

}  // namespace flashinfer
