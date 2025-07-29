#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <iostream>
#include <cassert>

bool is_initialized = false;
ncclComm_t nccl_comm;

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
    public:
        // TODO: create different comms for different world_size?
        ncclComm_t getcomm(int rank, int world_size) {
            ncclUniqueId ncclID;
            if (rank == 0) {
                ncclGetUniqueId(&ncclID);
            }
            broadcastUniqueNCCLID(&ncclID,
                false,
                "prefetch_all_to_all_comm",
                rank);
            ncclCommInitRank(&nccl_comm, world_size, ncclID, rank);
            return nccl_comm;
        }
};

void init_nccl_comm(c10d::ProcessGroup& p, int rank, int world_size) {
    if (is_initialized) {
        return;
    }
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)
        (p.getBackend(c10d::ProcessGroup::NCCL).get());
    nccl_comm = h->getcomm(rank, world_size);
    is_initialized = true;
}

extern "C" bool moe_nccl_forward_tensor(
    ncclComm_t comm,
    cudaStream_t stream,
    const at::Tensor& sharded_param,
    at::Tensor& padded_unsharded_param,
    const int* global_placement,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
);

extern "C" bool moe_nccl_backward_tensor(
    ncclComm_t comm,
    cudaStream_t stream,
    const at::Tensor& padded_unsharded_grad,
    at::Tensor& new_sharded_grad,
    const int* global_placement,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
);

extern "C" bool hierarchical_moe_nccl_forward_tensor(
    ncclComm_t comm,
    cudaStream_t stream,
    const at::Tensor& inter_buffer,
    at::Tensor& padded_unsharded_param,
    const int* global_placement,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
);

extern "C" bool hierarchical_moe_nccl_backward_tensor(
    ncclComm_t comm,
    cudaStream_t stream,
    const at::Tensor& padded_unsharded_grad,
    at::Tensor& inter_buffer,
    const int* global_placement,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
);

namespace py = pybind11;

void moe_nccl_forward(
    torch::Tensor sharded_param,
    torch::Tensor global_placement,
    torch::Tensor padded_unsharded_param,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
) {
    // assert(is_initialized);
    TORCH_CHECK(sharded_param.is_cuda(), "sharded_param must be on CUDA device");
    // TORCH_CHECK(global_placement.is_cuda(), "global_placement must be on CUDA device");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(sharded_param.device().index());

    bool success = moe_nccl_forward_tensor(
        nccl_comm, stream,
        sharded_param, padded_unsharded_param,
        global_placement.data_ptr<int>(),
        world_size, global_expert_num, local_expert_num, expert_shard_size
    );
    
    if (!success) {
        throw std::runtime_error("NCCL forward pass failed");
    }
    
}

void moe_nccl_backward(
    torch::Tensor padded_unsharded_grad,
    torch::Tensor global_placement,
    torch::Tensor recv_buffer, // TODO: no buffer?
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
) {
    // assert(is_initialized);
    TORCH_CHECK(padded_unsharded_grad.is_cuda(), "padded_unsharded_grad must be on CUDA device");
    //TORCH_CHECK(global_placement.is_cuda(), "global_placement must be on CUDA device");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(padded_unsharded_grad.device().index());

    bool success = moe_nccl_backward_tensor(
        nccl_comm, stream,
        padded_unsharded_grad, recv_buffer,
        global_placement.data_ptr<int>(),
        world_size, global_expert_num, local_expert_num, expert_shard_size
    );
    
    if (!success) {
        throw std::runtime_error("NCCL backward pass failed");
    }
    
}

void hierarchical_moe_nccl_forward(
    torch::Tensor inter_buffer,
    torch::Tensor global_placement,
    torch::Tensor padded_unsharded_param,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
) {
    // assert(is_initialized);
    TORCH_CHECK(inter_buffer.is_cuda(), "inter_buffer must be on CUDA device");
    // TORCH_CHECK(global_placement.is_cuda(), "global_placement must be on CUDA device");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(inter_buffer.device().index());

    bool success = hierarchical_moe_nccl_forward_tensor(
        nccl_comm, stream,
        inter_buffer, padded_unsharded_param,
        global_placement.data_ptr<int>(),
        world_size, global_expert_num, local_expert_num, expert_shard_size
    );
    
    if (!success) {
        throw std::runtime_error("NCCL forward pass failed");
    }
    
}

void hierarchical_moe_nccl_backward(
    torch::Tensor padded_unsharded_grad,
    torch::Tensor global_placement,
    torch::Tensor inter_buffer,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
) {
    // assert(is_initialized);
    TORCH_CHECK(padded_unsharded_grad.is_cuda(), "padded_unsharded_grad must be on CUDA device");
    //TORCH_CHECK(global_placement.is_cuda(), "global_placement must be on CUDA device");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(padded_unsharded_grad.device().index());

    bool success = hierarchical_moe_nccl_backward_tensor(
        nccl_comm, stream,
        padded_unsharded_grad, inter_buffer,
        global_placement.data_ptr<int>(),
        world_size, global_expert_num, local_expert_num, expert_shard_size
    );
    
    if (!success) {
        throw std::runtime_error("NCCL backward pass failed");
    }
    
}


PYBIND11_MODULE(moe_all_to_all_kernels, m) {
    m.doc() = "Optimized MoE All-to-All kernels using CUDA and NCCL";

    m.def("init_nccl_comm", &init_nccl_comm, 
          "Initialize NCCL communicator",
          py::arg("process_group"), py::arg("rank"), py::arg("world_size"));
    
    m.def("moe_nccl_forward", &moe_nccl_forward, 
          "NCCL-based MoE All-to-All forward pass",
          py::arg("sharded_param"), py::arg("global_placement"), py::arg("padded_unsharded_param"),
          py::arg("world_size"), py::arg("global_expert_num"), 
          py::arg("local_expert_num"), py::arg("expert_shard_size"));
    
    m.def("moe_nccl_backward", &moe_nccl_backward, 
          "NCCL-based MoE All-to-All backward pass",
          py::arg("padded_unsharded_grad"), py::arg("global_placement"), py::arg("recv_buffer"),
          py::arg("world_size"), py::arg("global_expert_num"), 
          py::arg("local_expert_num"), py::arg("expert_shard_size"));

    m.def("hierarchical_moe_nccl_forward", &hierarchical_moe_nccl_forward, 
          "NCCL-based hierarchical MoE All-to-All forward pass",
          py::arg("inter_buffer"), py::arg("global_placement"), py::arg("padded_unsharded_param"),
          py::arg("world_size"), py::arg("global_expert_num"), 
          py::arg("local_expert_num"), py::arg("expert_shard_size"));
    
    m.def("hierarchical_moe_nccl_backward", &hierarchical_moe_nccl_backward, 
          "NCCL-based hierarchical MoE All-to-All backward pass",
          py::arg("padded_unsharded_grad"), py::arg("global_placement"), py::arg("inter_buffer"),
          py::arg("world_size"), py::arg("global_expert_num"), 
          py::arg("local_expert_num"), py::arg("expert_shard_size"));

}