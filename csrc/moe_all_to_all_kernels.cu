#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nccl.h>
#include <iostream>
#include <torch/extension.h>

#define NCCL_SAFE_CALL(__fn__) { \
    auto __res__ = __fn__; \
    if (__res__ != ncclSuccess) { \
        fprintf(stderr, "NCCL Error at %s:%d value %d\n", __FILE__, __LINE__, __res__); \
        exit(-1); \
    } \
}

template<typename scalar_t>
ncclDataType_t get_nccl_dtype();

template<>
ncclDataType_t get_nccl_dtype<float>() {
    return ncclFloat;
}

template<>
ncclDataType_t get_nccl_dtype<double>() {
    return ncclDouble;
}

template<>
ncclDataType_t get_nccl_dtype<c10::Half>() {
    return ncclHalf;
}

template<>
ncclDataType_t get_nccl_dtype<c10::BFloat16>() {
    return ncclBfloat16;
}

template<typename scalar_t>
bool moe_nccl_alltoall_forward_impl(
    ncclComm_t comm,
    cudaStream_t stream,
    const scalar_t* input,
    scalar_t* output,
    const int* global_placement,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
) {
    const ncclDataType_t nccl_dtype = get_nccl_dtype<scalar_t>();

    NCCL_SAFE_CALL(ncclGroupStart());

    for (int rank = 0; rank < world_size; ++rank) {
        for (int local_expert = 0; local_expert < local_expert_num; ++local_expert) {
            const int placement_idx = rank * local_expert_num + local_expert;
            const int target_expert = global_placement[placement_idx];
            const size_t send_offset = target_expert * expert_shard_size;
            const size_t recv_offset = local_expert * (world_size * expert_shard_size) + 
                                    rank * expert_shard_size;

            NCCL_SAFE_CALL(ncclSend(
                input + send_offset,
                expert_shard_size,
                nccl_dtype,
                rank,
                comm,
                stream
            ));

            NCCL_SAFE_CALL(ncclRecv(
                output + recv_offset,
                expert_shard_size,
                nccl_dtype,
                rank,
                comm,
                stream
            ));
        }
    }
    
    NCCL_SAFE_CALL(ncclGroupEnd());
    
    return true;
}

template<typename scalar_t>
bool moe_nccl_alltoall_backward_impl(
    ncclComm_t comm,
    cudaStream_t stream,
    const scalar_t* input,
    scalar_t* output,
    const int* global_placement,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
) {
    const ncclDataType_t nccl_dtype = get_nccl_dtype<scalar_t>();

    NCCL_SAFE_CALL(ncclGroupStart());

    for (int rank = 0; rank < world_size; ++rank) {
        for (int local_expert = 0; local_expert < local_expert_num; ++local_expert) {
            const int placement_idx = rank * local_expert_num + local_expert;
            const int placed_expert = global_placement[placement_idx];
            const size_t send_offset = local_expert * (world_size * expert_shard_size) + 
                                        rank * expert_shard_size;
            // output size: world_size * local_expert_num * expert_shard_size
            const size_t recv_offset = placement_idx * expert_shard_size;
                    
            NCCL_SAFE_CALL(ncclSend(
                input + send_offset,
                expert_shard_size,
                nccl_dtype,
                rank,
                comm,
                stream
            ));
            
            NCCL_SAFE_CALL(ncclRecv(
                output + recv_offset,
                expert_shard_size,
                nccl_dtype,
                rank,
                comm,
                stream
            ));
        }
    }
    
    NCCL_SAFE_CALL(ncclGroupEnd());
    
    return true;
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
) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, sharded_param.scalar_type(), "moe_nccl_forward_tensor", [&]() {
        moe_nccl_alltoall_forward_impl<scalar_t>(
            comm, stream,
            sharded_param.data_ptr<scalar_t>(),
            padded_unsharded_param.data_ptr<scalar_t>(),
            global_placement, world_size, global_expert_num, local_expert_num, expert_shard_size
        );
    });
    
    return true;
}

extern "C" bool moe_nccl_backward_tensor(
    ncclComm_t comm,
    cudaStream_t stream,
    const at::Tensor& padded_unsharded_grad,
    at::Tensor& recv_buffer, // TODO: no buffer?
    const int* global_placement,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, padded_unsharded_grad.scalar_type(), "moe_nccl_backward_tensor", [&]() {
        moe_nccl_alltoall_backward_impl<scalar_t>(
            comm, stream,
            padded_unsharded_grad.data_ptr<scalar_t>(),
            recv_buffer.data_ptr<scalar_t>(),
            global_placement, world_size, global_expert_num, local_expert_num, expert_shard_size
        );
    });
    
    return true;
} 

// Only for intra node communication, make sure group init correctly before use
template<typename scalar_t>
bool hierarchical_moe_nccl_alltoall_forward_impl(
    ncclComm_t comm,
    cudaStream_t stream,
    const scalar_t* input,
    scalar_t* output,
    const int* global_placement,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
) {
    const ncclDataType_t nccl_dtype = get_nccl_dtype<scalar_t>();

    NCCL_SAFE_CALL(ncclGroupStart());
    
    int gpus_per_node = 8;
    int num_node = world_size / gpus_per_node;

    for (int node = 0; node < num_node; ++node) {
        for (int rank = 0; rank < gpus_per_node; ++rank) {
            for (int local_expert = 0; local_expert < local_expert_num; ++local_expert) {
                const int placement_idx = rank * local_expert_num + local_expert;
                const int target_expert = global_placement[placement_idx];
                const size_t send_offset = node * global_expert_num * expert_shard_size + target_expert * expert_shard_size;
                const size_t recv_offset = local_expert * (world_size * expert_shard_size) + 
                                        (node * gpus_per_node + rank) * expert_shard_size;

                NCCL_SAFE_CALL(ncclSend(
                    input + send_offset,
                    expert_shard_size,
                    nccl_dtype,
                    rank,
                    comm,
                    stream
                ));

                NCCL_SAFE_CALL(ncclRecv(
                    output + recv_offset,
                    expert_shard_size,
                    nccl_dtype,
                    rank,
                    comm,
                    stream
                ));
            }
        }
    }
    
    NCCL_SAFE_CALL(ncclGroupEnd());
    
    return true;
}

template<typename scalar_t>
bool hierarchical_moe_nccl_alltoall_backward_impl(
    ncclComm_t comm,
    cudaStream_t stream,
    const scalar_t* input,
    scalar_t* output,
    const int* global_placement,
    int world_size,
    int global_expert_num,
    int local_expert_num,
    int expert_shard_size
) {
    const ncclDataType_t nccl_dtype = get_nccl_dtype<scalar_t>();

    NCCL_SAFE_CALL(ncclGroupStart());

    int gpus_per_node = 8;
    int num_node = world_size / gpus_per_node;

    for (int node = 0; node < num_node; ++node) {
        for (int rank = 0; rank < gpus_per_node; ++rank) {
            for (int local_expert = 0; local_expert < local_expert_num; ++local_expert) {
                const int placement_idx = rank * local_expert_num + local_expert;
                const int placed_expert = global_placement[placement_idx];
                const size_t send_offset = local_expert * (world_size * expert_shard_size) + 
                                        (node * gpus_per_node + rank) * expert_shard_size;
                // output size: world_size * local_expert_num * expert_shard_size
                const size_t recv_offset = (node * gpus_per_node * local_expert_num + placement_idx) * expert_shard_size;
                        
                NCCL_SAFE_CALL(ncclSend(
                    input + send_offset,
                    expert_shard_size,
                    nccl_dtype,
                    rank,
                    comm,
                    stream
                ));
                
                NCCL_SAFE_CALL(ncclRecv(
                    output + recv_offset,
                    expert_shard_size,
                    nccl_dtype,
                    rank,
                    comm,
                    stream
                ));
            }
        }
    }
    
    NCCL_SAFE_CALL(ncclGroupEnd());
    
    return true;
}

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
) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, inter_buffer.scalar_type(), "hierarchical_moe_nccl_forward_tensor", [&]() {
        hierarchical_moe_nccl_alltoall_forward_impl<scalar_t>(
            comm, stream,
            inter_buffer.data_ptr<scalar_t>(),
            padded_unsharded_param.data_ptr<scalar_t>(),
            global_placement, world_size, global_expert_num, local_expert_num, expert_shard_size
        );
    });
    
    return true;
}

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
) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, padded_unsharded_grad.scalar_type(), "moe_nccl_backward_tensor", [&]() {
        hierarchical_moe_nccl_alltoall_backward_impl<scalar_t>(
            comm, stream,
            padded_unsharded_grad.data_ptr<scalar_t>(),
            inter_buffer.data_ptr<scalar_t>(),
            global_placement, world_size, global_expert_num, local_expert_num, expert_shard_size
        );
    });
    
    return true;
} 