// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include "sparse_common.cuh"
#include <hggc_runtime.h>
#include <hggc_bf16.h>

namespace minfer {
namespace sparse {

template <int kHeadDim, int kTileM, int kTileN, int kNWarps, int kQpk,
          typename Element>
__global__ void sparse_fmha_fwd_kernel(SparseFwdParams<Element> params);

}
}
