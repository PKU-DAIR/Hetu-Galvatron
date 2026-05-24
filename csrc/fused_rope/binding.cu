/*************************************************************************
 * PyTorch binding for the standalone FusedRoPE extraction.
 *
 * Replaces (in the TransformerEngine layout):
 *   - transformer_engine/pytorch/csrc/extensions/apply_rope.cu  (ATen <-> NVTETensor glue)
 *   - the 4 m.def lines in transformer_engine/pytorch/csrc/extensions/pybind.cpp
 *
 * Dispatch uses AT_DISPATCH_FLOATING_TYPES_AND2 to cover float/half/bfloat16
 * (the TE TYPE_SWITCH_INPUT macro also handles FP8, which is out of scope here).
 ************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "fused_rope.cuh"

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")

// Common validation for sbhd/bshd CP+SP entry points. `s` is the local seq
// length (after CP+SP slicing of the input). `freqs` must contain at least
// s * cp_size * sp_size rows so the kernel can index into the full table.
static inline void check_sbhd_cp_sp(const at::Tensor &x, const at::Tensor &freqs, int cp_size,
                                    int cp_rank, int sp_size, int sp_rank) {
  TORCH_CHECK(cp_size >= 1 && sp_size >= 1, "cp_size and sp_size must be >= 1");
  TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size, "cp_rank out of range");
  TORCH_CHECK(sp_rank >= 0 && sp_rank < sp_size, "sp_rank out of range");
  const int s_local = x.size(0);
  const int s_required = s_local * cp_size * sp_size;
  TORCH_CHECK(freqs.size(0) >= s_required,
              "freqs.size(0) must be >= input.size(0) * cp_size * sp_size; got ",
              freqs.size(0), " vs required ", s_required);
  if (cp_size > 1) {
    TORCH_CHECK((s_local * sp_size) % 2 == 0,
                "for cp_size>1, S_local * sp_size must be even (CP zigzag chunk size)");
  }
}

at::Tensor fused_rope_forward(const at::Tensor &input, const at::Tensor &freqs,
                              const bool transpose_output_memory, const int cp_size,
                              const int cp_rank, const int sp_size, const int sp_rank) {
  CHECK_CUDA(input);
  CHECK_CUDA(freqs);
  TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(input.size(3) >= freqs.size(3),
              "expected the last dim of the input tensor equals or is "
              "greater than the freqs tensor");
  CHECK_FLOAT(freqs);
  check_sbhd_cp_sp(input, freqs, cp_size, cp_rank, sp_size, sp_rank);

  const int s = input.size(0);
  const int b = input.size(1);
  const int h = input.size(2);
  const int d = input.size(3);
  const int stride_s = input.stride(0);
  const int stride_b = input.stride(1);
  const int stride_h = input.stride(2);
  const int stride_d = input.stride(3);
  const int d2 = freqs.size(3);

  auto act_options = input.options().requires_grad(false);
  at::Tensor output;
  if (transpose_output_memory) {
    output = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
  } else {
    output = torch::empty({s, b, h, d}, act_options);
  }
  const int o_stride_s = output.stride(0);
  const int o_stride_b = output.stride(1);
  const int o_stride_h = output.stride(2);
  const int o_stride_d = output.stride(3);

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "fused_rope_forward", [&] {
        fused_rope_kernels::fused_rope_forward_launcher<scalar_t>(
            input.data_ptr<scalar_t>(), freqs.data_ptr<float>(), output.data_ptr<scalar_t>(), s, b,
            h, d, d2, cp_size, cp_rank, sp_size, sp_rank, stride_s, stride_b, stride_h, stride_d,
            o_stride_s, o_stride_b, o_stride_h, o_stride_d, stream);
      });
  return output;
}

at::Tensor fused_rope_backward(const at::Tensor &output_grads, const at::Tensor &freqs,
                               const bool transpose_output_memory, const int cp_size,
                               const int cp_rank, const int sp_size, const int sp_rank) {
  CHECK_CUDA(output_grads);
  CHECK_CUDA(freqs);
  TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(output_grads.size(3) >= freqs.size(3),
              "expected the last dim of the output_grads tensor equals or is "
              "greater than the freqs tensor");
  CHECK_FLOAT(freqs);
  check_sbhd_cp_sp(output_grads, freqs, cp_size, cp_rank, sp_size, sp_rank);

  const int s = output_grads.size(0);
  const int b = output_grads.size(1);
  const int h = output_grads.size(2);
  const int d = output_grads.size(3);
  const int stride_s = output_grads.stride(0);
  const int stride_b = output_grads.stride(1);
  const int stride_h = output_grads.stride(2);
  const int stride_d = output_grads.stride(3);
  const int d2 = freqs.size(3);

  auto act_options = output_grads.options().requires_grad(false);
  at::Tensor input_grads;
  if (transpose_output_memory) {
    input_grads = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
  } else {
    input_grads = torch::empty({s, b, h, d}, act_options);
  }
  const int o_stride_s = input_grads.stride(0);
  const int o_stride_b = input_grads.stride(1);
  const int o_stride_h = input_grads.stride(2);
  const int o_stride_d = input_grads.stride(3);

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, output_grads.scalar_type(),
      "fused_rope_backward", [&] {
        fused_rope_kernels::fused_rope_backward_launcher<scalar_t>(
            output_grads.data_ptr<scalar_t>(), freqs.data_ptr<float>(),
            input_grads.data_ptr<scalar_t>(), s, b, h, d, d2, cp_size, cp_rank, sp_size, sp_rank,
            stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b, o_stride_h,
            o_stride_d, stream);
      });
  return input_grads;
}

at::Tensor fused_rope_thd_forward(const at::Tensor &input, const at::Tensor &cu_seqlens,
                                  const at::Tensor &freqs, const int cp_size, const int cp_rank,
                                  const int sp_size, const int sp_rank) {
  CHECK_CUDA(input);
  CHECK_CUDA(cu_seqlens);
  CHECK_CUDA(freqs);
  TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(cu_seqlens.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int, "cu_seqlens must be int32");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(input.size(2) >= freqs.size(3),
              "expected the last dim of the input tensor equals or is "
              "greater than the freqs tensor");
  TORCH_CHECK(cp_size >= 1 && sp_size >= 1, "cp_size and sp_size must be >= 1");
  TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size, "cp_rank out of range");
  TORCH_CHECK(sp_rank >= 0 && sp_rank < sp_size, "sp_rank out of range");
  CHECK_FLOAT(freqs);

  const int T_local = input.size(0);
  const int h = input.size(1);
  const int d = input.size(2);
  const int stride_t = input.stride(0);
  const int stride_h = input.stride(1);
  const int stride_d = input.stride(2);
  const int batch = cu_seqlens.size(0) - 1;
  const int d2 = freqs.size(3);

  auto act_options = input.options().requires_grad(false);
  auto output = torch::empty({T_local, h, d}, act_options);
  const int o_stride_t = output.stride(0);
  const int o_stride_h = output.stride(1);
  const int o_stride_d = output.stride(2);

  if (T_local == 0) return output;

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "fused_rope_thd_forward", [&] {
        fused_rope_kernels::fused_rope_thd_forward_launcher<scalar_t>(
            input.data_ptr<scalar_t>(), cu_seqlens.data_ptr<int>(), freqs.data_ptr<float>(),
            output.data_ptr<scalar_t>(), cp_size, cp_rank, sp_rank, batch, T_local, h, d, d2,
            stride_t, stride_h, stride_d, o_stride_t, o_stride_h, o_stride_d, stream);
      });
  return output;
}

at::Tensor fused_rope_thd_backward(const at::Tensor &output_grads, const at::Tensor &cu_seqlens,
                                   const at::Tensor &freqs, const int cp_size, const int cp_rank,
                                   const int sp_size, const int sp_rank) {
  CHECK_CUDA(output_grads);
  CHECK_CUDA(cu_seqlens);
  CHECK_CUDA(freqs);
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(cu_seqlens.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int, "cu_seqlens must be int32");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(output_grads.size(2) >= freqs.size(3),
              "expected the last dim of the output_grads tensor equals or is "
              "greater than the freqs tensor");
  TORCH_CHECK(cp_size >= 1 && sp_size >= 1, "cp_size and sp_size must be >= 1");
  TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size, "cp_rank out of range");
  TORCH_CHECK(sp_rank >= 0 && sp_rank < sp_size, "sp_rank out of range");
  CHECK_FLOAT(freqs);

  const int T_local = output_grads.size(0);
  const int h = output_grads.size(1);
  const int d = output_grads.size(2);
  const int stride_t = output_grads.stride(0);
  const int stride_h = output_grads.stride(1);
  const int stride_d = output_grads.stride(2);
  const int batch = cu_seqlens.size(0) - 1;
  const int d2 = freqs.size(3);

  auto act_options = output_grads.options().requires_grad(false);
  auto input_grads = torch::empty({T_local, h, d}, act_options);
  const int o_stride_t = input_grads.stride(0);
  const int o_stride_h = input_grads.stride(1);
  const int o_stride_d = input_grads.stride(2);

  if (T_local == 0) return input_grads;

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, output_grads.scalar_type(),
      "fused_rope_thd_backward", [&] {
        fused_rope_kernels::fused_rope_thd_backward_launcher<scalar_t>(
            output_grads.data_ptr<scalar_t>(), cu_seqlens.data_ptr<int>(), freqs.data_ptr<float>(),
            input_grads.data_ptr<scalar_t>(), cp_size, cp_rank, sp_rank, batch, T_local, h, d, d2,
            stride_t, stride_h, stride_d, o_stride_t, o_stride_h, o_stride_d, stream);
      });
  return input_grads;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_rope_forward", &fused_rope_forward,
        "Fused Apply RoPE FWD (sbhd/bshd, supports CP + SP)", py::arg("input"), py::arg("freqs"),
        py::arg("transpose_output_memory") = false, py::arg("cp_size") = 1,
        py::arg("cp_rank") = 0, py::arg("sp_size") = 1, py::arg("sp_rank") = 0);
  m.def("fused_rope_backward", &fused_rope_backward,
        "Fused Apply RoPE BWD (sbhd/bshd, supports CP + SP)", py::arg("output_grads"),
        py::arg("freqs"), py::arg("transpose_output_memory") = false, py::arg("cp_size") = 1,
        py::arg("cp_rank") = 0, py::arg("sp_size") = 1, py::arg("sp_rank") = 0);
  m.def("fused_rope_thd_forward", &fused_rope_thd_forward,
        "Fused Apply RoPE FWD (thd format, supports CP + SP)", py::arg("input"),
        py::arg("cu_seqlens"), py::arg("freqs"), py::arg("cp_size") = 1, py::arg("cp_rank") = 0,
        py::arg("sp_size") = 1, py::arg("sp_rank") = 0);
  m.def("fused_rope_thd_backward", &fused_rope_thd_backward,
        "Fused Apply RoPE BWD (thd format, supports CP + SP)", py::arg("output_grads"),
        py::arg("cu_seqlens"), py::arg("freqs"), py::arg("cp_size") = 1, py::arg("cp_rank") = 0,
        py::arg("sp_size") = 1, py::arg("sp_rank") = 0);
}
