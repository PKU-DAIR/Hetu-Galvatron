/*************************************************************************
 * Adapted from NVIDIA TransformerEngine
 * Original: transformer_engine/common/fused_rope/fused_rope.cu
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 *
 * Standalone, header-only port: kernels + templated launchers with raw-pointer
 * interface. TransformerEngine internal abstractions (NVTETensor, Tensor,
 * TYPE_SWITCH macros, logging.h) have been removed.
 ************************************************************************/

#pragma once

#include <assert.h>
#include <cuda_runtime.h>

namespace fused_rope_kernels {

constexpr int THREADS_PER_WARP = 32;

#define FRP_CHECK_CUDA(expr)                                              \
  do {                                                                    \
    cudaError_t _err = (expr);                                            \
    if (_err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
              cudaGetErrorString(_err), __FILE__, __LINE__);              \
      abort();                                                            \
    }                                                                     \
  } while (0)

template <typename scalar_t>
__device__ void fused_rope_block_forward(const scalar_t *src, const float *freqs, scalar_t *dst,
                                         const int s_id, const int offset_block,
                                         const int offset_block_dst, const int h, const int d,
                                         const int d2, const int stride_h, const int stride_d,
                                         const int o_stride_h, const int o_stride_d) {
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos, v_sin;
    sincosf(freqs[s_id * d2 + d_id], &v_sin, &v_cos);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate = (d_id + d2 / 2 < d2)
                               ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
                               : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__device__ void fused_rope_block_backward(const scalar_t *src, const float *freqs, scalar_t *dst,
                                          const int s_id, const int offset_block,
                                          const int offset_block_dst, const int h, const int d,
                                          const int d2, const int stride_h, const int stride_d,
                                          const int o_stride_h, const int o_stride_d) {
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos = cosf(freqs[s_id * d2 + d_id]);
    float v_sin = (d_id + d2 / 2 < d2) ? sinf(freqs[s_id * d2 + d_id + d2 / 2])
                                       : -sinf(freqs[s_id * d2 + d_id + d2 / 2 - d2]);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate = (d_id + d2 / 2 < d2) ? src[offset_src + (d2 / 2) * stride_d]
                                                : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // handle the tail
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d];
      }
    }
  }
}

// Map a (sbhd/bshd) local sequence position to the row in the FULL freqs table,
// applying CP zigzag + SP narrow. With cp_size = sp_size = 1 this is a no-op
// (s_full_idx == s_local), matching the original TE kernel.
//
// Equivalent to galvatron's `get_pos_emb_on_this_cp_sp_rank`:
//   1. Split full freqs into 2*cp_size chunks of size S_full/(2*cp_size).
//      CP rank gets chunks [cp_rank, 2*cp_size-cp_rank-1].
//   2. SP narrow takes a contiguous slice of size S_local from the
//      CP-sliced stream at offset sp_rank * S_local.
__device__ inline int sbhd_remap_freqs_idx(const int s_local, const int S_local,
                                           const int cp_size, const int cp_rank,
                                           const int sp_size, const int sp_rank) {
  if (cp_size == 1 && sp_size == 1) return s_local;
  const int s_cp = sp_rank * S_local + s_local;
  if (cp_size == 1) return s_cp;
  // S_full / (2*cp_size) == S_local * sp_size / 2
  const int chunk = S_local * sp_size / 2;
  if (s_cp < chunk) {
    return cp_rank * chunk + s_cp;
  } else {
    return (2 * cp_size - cp_rank - 1) * chunk + (s_cp - chunk);
  }
}

template <typename scalar_t>
__global__ void fused_rope_forward_kernel(const scalar_t *src, const float *freqs, scalar_t *dst,
                                          const int h, const int d, const int d2,
                                          const int cp_size, const int cp_rank, const int sp_size,
                                          const int sp_rank, const int stride_s,
                                          const int stride_b, const int stride_h,
                                          const int stride_d, const int o_stride_s,
                                          const int o_stride_b, const int o_stride_h,
                                          const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  int s_full_idx =
      sbhd_remap_freqs_idx(s_id, gridDim.x, cp_size, cp_rank, sp_size, sp_rank);
  fused_rope_block_forward(src, freqs, dst, s_full_idx, offset_block, offset_block_dst, h, d, d2,
                           stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_backward_kernel(const scalar_t *src, const float *freqs, scalar_t *dst,
                                           const int h, const int d, const int d2,
                                           const int cp_size, const int cp_rank, const int sp_size,
                                           const int sp_rank, const int stride_s,
                                           const int stride_b, const int stride_h,
                                           const int stride_d, const int o_stride_s,
                                           const int o_stride_b, const int o_stride_h,
                                           const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  int s_full_idx =
      sbhd_remap_freqs_idx(s_id, gridDim.x, cp_size, cp_rank, sp_size, sp_rank);
  fused_rope_block_backward(src, freqs, dst, s_full_idx, offset_block, offset_block_dst, h, d, d2,
                            stride_h, stride_d, o_stride_h, o_stride_d);
}

// THD kernels (with CP + SP support).
//
// Layout assumptions (matching galvatron's _apply_rotary_pos_emb_thd):
//   * Input `src` has shape [T_local, h, d], where T_local is the per-rank
//     token count AFTER both CP and SP slicing:
//         T_local = (cu_seqlens[-1] / cp_size) / sp_size
//   * `cu_seqlens` is the ORIGINAL (pre-CP, pre-SP) cumulative sequence lengths
//     of length batch + 1. The kernel divides by cp_size internally.
//   * `freqs` is the FULL positional-embedding table of shape [max_s, 1, 1, d2].
//     The kernel maps each output token back to its index in this table via:
//
//       g            = sp_rank * T_local + t_id_local           (CP-local global pos)
//       (sample b)   = scan cu_seqlens/cp_size for the sample containing g
//       s_local      = g - cu_seqlens[b]/cp_size                (within-sample pos after CP)
//       s_for_freqs  = CP zigzag(s_local, cp_rank, cp_size)     (index into full freqs)
//
// The grid is 1D over T_local because SP slicing cuts across sample boundaries,
// so the original 2D (max_s, batch) launch is no longer sample-local.

template <typename scalar_t>
__device__ inline void thd_locate_token(const int *cu_seqlens, const int cp_size,
                                        const int batch, const int g, int &sample_start,
                                        int &sample_end) {
  // Linear scan; batch is small (~tens) so divergence is acceptable.
  int b_id = 0;
  sample_start = cu_seqlens[0] / cp_size;
  sample_end = cu_seqlens[1] / cp_size;
  while (b_id < batch - 1 && g >= sample_end) {
    ++b_id;
    sample_start = sample_end;
    sample_end = cu_seqlens[b_id + 1] / cp_size;
  }
}

template <typename scalar_t>
__global__ void fused_rope_thd_forward_kernel(const scalar_t *src, const int *cu_seqlens,
                                              const float *freqs, scalar_t *dst, const int cp_size,
                                              const int cp_rank, const int sp_rank,
                                              const int batch, const int T_local, const int h,
                                              const int d, const int d2, const int stride_t,
                                              const int stride_h, const int stride_d,
                                              const int o_stride_t, const int o_stride_h,
                                              const int o_stride_d) {
  const int t_id = blockIdx.x;
  if (t_id >= T_local) return;

  const int g = sp_rank * T_local + t_id;
  int sample_start, sample_end;
  thd_locate_token<scalar_t>(cu_seqlens, cp_size, batch, g, sample_start, sample_end);
  if (g >= sample_end) return;  // padding / oversized T_local guard

  const int s_local = g - sample_start;
  const int cur_seqlens = sample_end - sample_start;

  int s_id_for_freqs;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_local < cur_seqlens / 2) {
      s_id_for_freqs = s_local + cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs =
          cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 + s_local - cur_seqlens / 2;
    }
  } else {
    s_id_for_freqs = s_local;
  }

  const int offset_block = t_id * stride_t;
  const int offset_block_dst = t_id * o_stride_t;
  fused_rope_block_forward(src, freqs, dst, s_id_for_freqs, offset_block, offset_block_dst, h, d,
                           d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_thd_backward_kernel(const scalar_t *src, const int *cu_seqlens,
                                               const float *freqs, scalar_t *dst, const int cp_size,
                                               const int cp_rank, const int sp_rank,
                                               const int batch, const int T_local, const int h,
                                               const int d, const int d2, const int stride_t,
                                               const int stride_h, const int stride_d,
                                               const int o_stride_t, const int o_stride_h,
                                               const int o_stride_d) {
  const int t_id = blockIdx.x;
  if (t_id >= T_local) return;

  const int g = sp_rank * T_local + t_id;
  int sample_start, sample_end;
  thd_locate_token<scalar_t>(cu_seqlens, cp_size, batch, g, sample_start, sample_end);
  if (g >= sample_end) return;

  const int s_local = g - sample_start;
  const int cur_seqlens = sample_end - sample_start;

  int s_id_for_freqs;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_local < cur_seqlens / 2) {
      s_id_for_freqs = s_local + cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs =
          cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 + s_local - cur_seqlens / 2;
    }
  } else {
    s_id_for_freqs = s_local;
  }

  const int offset_block = t_id * stride_t;
  const int offset_block_dst = t_id * o_stride_t;
  fused_rope_block_backward(src, freqs, dst, s_id_for_freqs, offset_block, offset_block_dst, h, d,
                            d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
void fused_rope_forward_launcher(const scalar_t *input, const float *freqs, scalar_t *output,
                                 const int s, const int b, const int h, const int d, const int d2,
                                 const int cp_size, const int cp_rank, const int sp_size,
                                 const int sp_rank, const int stride_s, const int stride_b,
                                 const int stride_h, const int stride_d, const int o_stride_s,
                                 const int o_stride_b, const int o_stride_h, const int o_stride_d,
                                 cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_forward_kernel<<<blocks, threads, 0, stream>>>(
      input, freqs, output, h, d, d2, cp_size, cp_rank, sp_size, sp_rank, stride_s, stride_b,
      stride_h, stride_d, o_stride_s, o_stride_b, o_stride_h, o_stride_d);
  FRP_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_rope_backward_launcher(const scalar_t *output_grads, const float *freqs,
                                  scalar_t *input_grads, const int s, const int b, const int h,
                                  const int d, const int d2, const int cp_size, const int cp_rank,
                                  const int sp_size, const int sp_rank, const int stride_s,
                                  const int stride_b, const int stride_h, const int stride_d,
                                  const int o_stride_s, const int o_stride_b, const int o_stride_h,
                                  const int o_stride_d, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_backward_kernel<<<blocks, threads, 0, stream>>>(
      output_grads, freqs, input_grads, h, d, d2, cp_size, cp_rank, sp_size, sp_rank, stride_s,
      stride_b, stride_h, stride_d, o_stride_s, o_stride_b, o_stride_h, o_stride_d);
  FRP_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_rope_thd_forward_launcher(const scalar_t *input, const int *cu_seqlens,
                                     const float *freqs, scalar_t *output, const int cp_size,
                                     const int cp_rank, const int sp_rank, const int batch,
                                     const int T_local, const int h, const int d, const int d2,
                                     const int stride_t, const int stride_h, const int stride_d,
                                     const int o_stride_t, const int o_stride_h,
                                     const int o_stride_d, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(T_local);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_thd_forward_kernel<<<blocks, threads, 0, stream>>>(
      input, cu_seqlens, freqs, output, cp_size, cp_rank, sp_rank, batch, T_local, h, d, d2,
      stride_t, stride_h, stride_d, o_stride_t, o_stride_h, o_stride_d);
  FRP_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_rope_thd_backward_launcher(const scalar_t *output_grads, const int *cu_seqlens,
                                      const float *freqs, scalar_t *input_grads, const int cp_size,
                                      const int cp_rank, const int sp_rank, const int batch,
                                      const int T_local, const int h, const int d, const int d2,
                                      const int stride_t, const int stride_h, const int stride_d,
                                      const int o_stride_t, const int o_stride_h,
                                      const int o_stride_d, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(T_local);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_thd_backward_kernel<<<blocks, threads, 0, stream>>>(
      output_grads, cu_seqlens, freqs, input_grads, cp_size, cp_rank, sp_rank, batch, T_local, h, d,
      d2, stride_t, stride_h, stride_d, o_stride_t, o_stride_h, o_stride_d);
  FRP_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fused_rope_kernels
