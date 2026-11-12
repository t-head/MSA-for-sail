// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include "sparse_fmha_fwd_impl.cuh"
#include "tvm_ffi_utils.h"

#include <cmath>
#include <limits>
#include <type_traits>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

namespace minfer {
namespace sparse {

enum SparseFwdDispatchKey : int {
  kSparseFwdDispatch_BF16 = 0,
  kSparseFwdDispatch_FP8  = 1,
  kSparseFwdDispatch_Count,
};

static_assert(kSparseFwdDispatch_Count == 2,
              "dispatch key count must match python SparseFwdDispatchKey");

#define SPARSE_FWD_ENTRY_PARAMS                          \
  tvm::ffi::TensorView Q,                                \
  tvm::ffi::TensorView K,                                \
  tvm::ffi::TensorView V,                                \
  tvm::ffi::TensorView k2q_row_ptr,                      \
  tvm::ffi::TensorView k2q_q_indices,                    \
  tvm::ffi::TensorView qsplit_indices,                   \
  tvm::ffi::TensorView scheduler_metadata,               \
  tvm::ffi::TensorView work_count,                       \
  tvm::ffi::TensorView cu_seqlens_q,                     \
  tvm::ffi::TensorView cu_seqlens_k,                     \
  tvm::ffi::TensorView kv_indices,                       \
  tvm::ffi::TensorView kv_page_indptr,                   \
  tvm::ffi::TensorView O_partial,                        \
  tvm::ffi::TensorView LSE_partial,                      \
  tvm::ffi::TensorView split_counts,                     \
  double sm_scale,                                       \
  int64_t num_heads_kv,                                  \
  int64_t qhead_per_kv,                                  \
  int64_t topk,                                          \
  int64_t max_seqlen_q,                                  \
  int64_t is_paged,                                      \
  int64_t page_size,                                     \
  int64_t work_capacity,                                 \
  int64_t dispatch_key,                                  \
  int64_t stream_ptr,                                    \
  tvm::ffi::TensorView LSE_temperature_partial,          \
  double lse_temperature_scale,                          \
  tvm::ffi::TensorView seqused_k,                        \
  int64_t has_seqused_k

#define SPARSE_FWD_ENTRY_ARGS                                                      \
  Q, K, V, k2q_row_ptr, k2q_q_indices, qsplit_indices, scheduler_metadata,         \
      work_count, cu_seqlens_q, cu_seqlens_k, kv_indices, kv_page_indptr,          \
      O_partial, LSE_partial, split_counts, sm_scale, num_heads_kv,                \
      qhead_per_kv, topk, max_seqlen_q, is_paged, page_size, work_capacity,        \
      dispatch_key, stream_ptr, LSE_temperature_partial,                             \
      lse_temperature_scale, seqused_k, has_seqused_k

template <int kQpk, typename Element>
static void launch_fwd_variant(SparseFwdParams<Element>& params,
                               int64_t work_capacity, hggcStream_t stream) {
  constexpr int kHeadDim = 128;
  constexpr int kTileM = (kQpk >= 8) ? 128 : 64;
  constexpr int kTileN = 128;
  constexpr int kNWarps = (kQpk >= 8) ? 8 : 4;
  constexpr int kThreads = kNWarps * 32;
  constexpr int kSmemBytes =
      SparseFwdSmemBudget<Element, kTileM, kTileN, kHeadDim>::kTotal;

  const void* kernel = (const void*)sparse_fmha_fwd_kernel<
      kHeadDim, kTileM, kTileN, kNWarps, kQpk, Element>;

  static bool attr_set = false;
  if (!attr_set) {
    hggcFuncSetAttribute(
        kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    attr_set = true;
  }

  dim3 grid(static_cast<int>(work_capacity));
  dim3 block(kThreads);
  sparse_fmha_fwd_kernel<kHeadDim, kTileM, kTileN, kNWarps, kQpk, Element>
      <<<grid, block, kSmemBytes, stream>>>(params);
}

template <int kQpk, int kDtypeKey>
void sparse_fmha_fwd_forward(SPARSE_FWD_ENTRY_PARAMS) {
  TVM_FFI_ICHECK_EQ(dispatch_key, static_cast<int64_t>(kDtypeKey))
      << "dispatch key " << dispatch_key
      << " does not match the compiled variant (kDtypeKey=" << kDtypeKey << ")";
  if constexpr (kQpk > 1) {
    TVM_FFI_ICHECK_EQ(qhead_per_kv, static_cast<int64_t>(kQpk))
        << "packed variant (kQpk=" << kQpk
        << ") requires qhead_per_kv == " << kQpk;
  }

  CHECK_INPUT_AND_TYPE(k2q_row_ptr, dl_int32);
  CHECK_INPUT_AND_TYPE(k2q_q_indices, dl_int32);
  CHECK_INPUT_AND_TYPE(qsplit_indices, dl_int32);
  CHECK_INPUT_AND_TYPE(scheduler_metadata, dl_int32);
  CHECK_INPUT_AND_TYPE(work_count, dl_int32);
  CHECK_INPUT_AND_TYPE(cu_seqlens_q, dl_int32);
  CHECK_INPUT_AND_TYPE(cu_seqlens_k, dl_int32);
  CHECK_INPUT_AND_TYPE(O_partial, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(LSE_partial, dl_float32);
  CHECK_INPUT_AND_TYPE(split_counts, dl_int32);
  CHECK_DIM(3, Q);
  CHECK_DIM(2, k2q_row_ptr);
  CHECK_DIM(2, k2q_q_indices);
  CHECK_DIM(2, qsplit_indices);
  CHECK_DIM(2, scheduler_metadata);
  CHECK_DIM(1, work_count);
  CHECK_DIM(1, cu_seqlens_q);
  CHECK_DIM(1, cu_seqlens_k);
  CHECK_DIM(4, O_partial);
  CHECK_DIM(3, LSE_partial);
  CHECK_DIM(2, split_counts);
  TVM_FFI_ICHECK_EQ(Q.size(2), 128)
      << "sparse_fmha_fwd only supports head_dim=128, got " << Q.size(2);
  TVM_FFI_ICHECK_EQ(scheduler_metadata.size(1), 6);
  TVM_FFI_ICHECK_EQ(work_count.size(0), 1);
  TVM_FFI_ICHECK_EQ(k2q_q_indices.size(0), qsplit_indices.size(0));
  TVM_FFI_ICHECK_EQ(k2q_q_indices.size(1), qsplit_indices.size(1));

  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(Q.data_ptr()) % 16, 0)
      << "Q base pointer must be 16B aligned";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(K.data_ptr()) % 16, 0)
      << "K base pointer must be 16B aligned";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(V.data_ptr()) % 16, 0)
      << "V base pointer must be 16B aligned";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(O_partial.data_ptr()) % 16, 0)
      << "O_partial base pointer must be 16B aligned";

  CHECK_LAST_DIM_CONTIGUOUS(Q);
  TVM_FFI_ICHECK(Q.stride(0) > 0 && Q.stride(1) > 0)
      << "Q token/head strides must be positive, got (" << Q.stride(0)
      << ", " << Q.stride(1) << ")";
  TVM_FFI_ICHECK(Q.stride(1) <= std::numeric_limits<int>::max())
      << "Q head stride must fit in int for the AIU descriptor, got "
      << Q.stride(1);

  if (is_paged) {
    CHECK_INPUT_AND_TYPE(kv_indices, dl_int32);
    CHECK_INPUT_AND_TYPE(kv_page_indptr, dl_int32);
    CHECK_DIM(1, kv_indices);
    CHECK_DIM(1, kv_page_indptr);
  }
  if (has_seqused_k) {
    CHECK_INPUT_AND_TYPE(seqused_k, dl_int32);
    CHECK_DIM(1, seqused_k);
    TVM_FFI_ICHECK_EQ(seqused_k.size(0), cu_seqlens_q.size(0) - 1)
        << "seqused_k must have shape [batch]";
  }

  bool kv_is_nhd = false;
  if (is_paged) {
    CHECK_DIM(4, K);
    CHECK_DIM(4, V);
    TVM_FFI_ICHECK_EQ(K.size(3), 128)
        << "paged K last dim must be head_dim=128, got " << K.size(3);
    TVM_FFI_ICHECK_EQ(V.size(3), 128)
        << "paged V last dim must be head_dim=128, got " << V.size(3);
    kv_is_nhd = (K.size(1) == page_size);
    TVM_FFI_ICHECK(kv_is_nhd || K.size(2) == page_size)
        << "paged K shape (" << K.size(0) << ", " << K.size(1) << ", "
        << K.size(2) << ", " << K.size(3) << ") has neither dim1 nor dim2 "
        << "equal to page_size=" << page_size
        << "; expected NHD [pages, ps, Hkv, D] or HND [pages, Hkv, ps, D]";
    const int64_t kv_heads = kv_is_nhd ? K.size(2) : K.size(1);
    TVM_FFI_ICHECK_EQ(kv_heads, num_heads_kv)
        << "paged K shape (" << K.size(0) << ", " << K.size(1) << ", "
        << K.size(2) << ", " << K.size(3) << ") gives " << kv_heads
        << " KV heads, but the plan was built for " << num_heads_kv
        << " (page_size=" << page_size
        << (kv_is_nhd ? ", NHD" : ", HND") << ")";
    TVM_FFI_ICHECK(V.size(0) == K.size(0) && V.size(1) == K.size(1) &&
                   V.size(2) == K.size(2) && V.size(3) == K.size(3))
        << "paged V shape (" << V.size(0) << ", " << V.size(1) << ", "
        << V.size(2) << ", " << V.size(3)
        << ") must match K shape (" << K.size(0) << ", " << K.size(1) << ", "
        << K.size(2) << ", " << K.size(3) << ")";
  }

  int64_t head_dim = 128;
  int64_t num_heads_q = Q.size(1);
  int64_t total_q = Q.size(0);
  int64_t total_rows = k2q_row_ptr.size(1) - 1;
  int64_t batch = cu_seqlens_q.size(0) - 1;
  int64_t nnz = qsplit_indices.size(1);

  const hggcStream_t stream = reinterpret_cast<hggcStream_t>(stream_ptr);

  float lse_temp_scale = static_cast<float>(lse_temperature_scale);
  bool has_temperature = (lse_temp_scale > 0.f
                          && fabsf(lse_temp_scale - 1.0f) > 1e-12f);

  if (work_capacity <= 0 || total_rows <= 0 || num_heads_kv <= 0) return;

  float sm_scale_f = static_cast<float>(sm_scale);
  float scale_log2 = sm_scale_f * kLog2e;

  auto run = [&](auto* params_ptr) {
    using ParamsType = std::remove_pointer_t<decltype(params_ptr)>;
    using Element = typename ParamsType::Element_t;
    ParamsType params;
    TVM_FFI_ICHECK_EQ(
        (Q.stride(0) * static_cast<int64_t>(sizeof(Element))) % 16, 0)
        << "Q token stride must be 16B aligned for the vectorized Q gather, got "
        << Q.stride(0) << " elements of " << sizeof(Element) << "B";
    TVM_FFI_ICHECK_EQ(
        (Q.stride(1) * static_cast<int64_t>(sizeof(Element))) % 16, 0)
        << "Q head stride must be 16B aligned for the vectorized Q gather, got "
        << Q.stride(1) << " elements of " << sizeof(Element) << "B";
    params.ptr_Q = static_cast<const Element*>(Q.data_ptr());
    params.q_stride_n = Q.stride(0);
    params.q_stride_h = Q.stride(1);
    params.ptr_K = static_cast<const Element*>(K.data_ptr());
    params.ptr_V = static_cast<const Element*>(V.data_ptr());
    params.k2q_row_ptr = static_cast<const int32_t*>(k2q_row_ptr.data_ptr());
    params.k2q_q_indices = static_cast<const int32_t*>(k2q_q_indices.data_ptr());
    params.qsplit_indices = static_cast<const int32_t*>(qsplit_indices.data_ptr());
    params.scheduler_metadata = static_cast<const int32_t*>(scheduler_metadata.data_ptr());
    params.work_count = static_cast<const int32_t*>(work_count.data_ptr());
    params.cu_seqlens_q = static_cast<const int32_t*>(cu_seqlens_q.data_ptr());
    params.cu_seqlens_k = static_cast<const int32_t*>(cu_seqlens_k.data_ptr());
    params.seqused_k = has_seqused_k
        ? static_cast<const int32_t*>(seqused_k.data_ptr()) : nullptr;
    params.kv_indices = is_paged ? static_cast<const int32_t*>(kv_indices.data_ptr()) : nullptr;
    params.kv_page_indptr = is_paged ? static_cast<const int32_t*>(kv_page_indptr.data_ptr()) : nullptr;
    params.ptr_O_partial = static_cast<__nv_bfloat16*>(O_partial.data_ptr());
    params.ptr_LSE_partial = static_cast<float*>(LSE_partial.data_ptr());
    params.split_counts = static_cast<const int32_t*>(split_counts.data_ptr());
    params.sm_scale = sm_scale_f;
    params.scale_softmax_log2 = scale_log2;
    params.head_dim = static_cast<int>(head_dim);
    params.num_heads_kv = static_cast<int>(num_heads_kv);
    params.num_heads_q = static_cast<int>(num_heads_q);
    params.qhead_per_kv = static_cast<int>(qhead_per_kv);
    params.total_q = static_cast<int>(total_q);
    params.total_rows = static_cast<int>(total_rows);
    params.batch = static_cast<int>(batch);
    params.blk_kv = 128;
    params.page_size = static_cast<int>(page_size);
    params.topk = static_cast<int>(topk);
    params.max_seqlen_q = static_cast<int>(max_seqlen_q);
    params.is_paged = static_cast<int>(is_paged);
    params.kv_is_nhd = kv_is_nhd ? 1 : 0;
    params.nnz = nnz;
    params.ptr_LSE_temperature_partial = has_temperature
        ? static_cast<float*>(LSE_temperature_partial.data_ptr()) : nullptr;
    params.lse_temperature_scale = lse_temp_scale;
    params.lse_temperature_scale_log2 = has_temperature
        ? scale_log2 * lse_temp_scale : 0.f;
    params.return_temperature_lse = has_temperature ? 1 : 0;
    return params;
  };

  if constexpr (kDtypeKey == kSparseFwdDispatch_BF16) {
    CHECK_CUDA(Q);
    CHECK_INPUT_TYPE(Q, dl_bfloat16);
    CHECK_INPUT_AND_TYPE(K, dl_bfloat16);
    CHECK_INPUT_AND_TYPE(V, dl_bfloat16);
    auto params = run(static_cast<SparseFwdParams<cutlass::bfloat16_t>*>(nullptr));
    launch_fwd_variant<kQpk, cutlass::bfloat16_t>(params, work_capacity, stream);
  } else if constexpr (kDtypeKey == kSparseFwdDispatch_FP8) {
    CHECK_CUDA(Q);
    CHECK_INPUT_TYPE(Q, dl_float8_e4m3fn);
    CHECK_INPUT_AND_TYPE(K, dl_float8_e4m3fn);
    CHECK_INPUT_AND_TYPE(V, dl_float8_e4m3fn);
    auto params = run(static_cast<SparseFwdParams<cutlass::float_e4m3_t>*>(nullptr));
    launch_fwd_variant<kQpk, cutlass::float_e4m3_t>(params, work_capacity, stream);
  } else {
    static_assert(kDtypeKey < 0 || kDtypeKey >= kSparseFwdDispatch_Count,
                  "unsupported dispatch key for this variant");
  }

  hggcError_t status = hggcGetLastError();
  if (status != hggcSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "sparse_fmha_fwd launch failed: " << hggcGetErrorString(status);
  }
}

}
}
