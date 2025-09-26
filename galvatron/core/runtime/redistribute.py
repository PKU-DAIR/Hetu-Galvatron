import torch
from einops import rearrange

# from megatron.core.parallel_state import (
#     get_global_memory_buffer,
# )

def _zigzag_transformation(input_, cp_world_size):
    if cp_world_size == 1:
        return input_
    
    seq_dim = 0
    original_shape = input_.shape
    assert 2*cp_world_size <= original_shape[0], "sequence length must be larger than 2*cp" 
    reshaped_input = input_.view(2 * cp_world_size, -1, *original_shape[1:])
    zigzag_indices = torch.zeros(2 * cp_world_size, dtype=torch.long, device=input_.device)
    for cp_rank in range(cp_world_size):
    
        idx1 = cp_rank
        idx2 = 2 * cp_world_size - cp_rank - 1
        
        zigzag_indices[2 * cp_rank] = idx1
        zigzag_indices[2 * cp_rank + 1] = idx2
    zigzag_tensor = reshaped_input[zigzag_indices]
    output_shape = (-1, *original_shape[1:])
    output = zigzag_tensor.contiguous().view(output_shape)
    return output

def _reverse_zigzag_transformation(input_, cp_world_size):
    if cp_world_size == 1:
        return input_
    seq_dim = 0 
    original_shape = input_.shape
    reshaped_input = input_.view(2 * cp_world_size, -1, *original_shape[1:])
    reverse_indices = torch.zeros(2 * cp_world_size, dtype=torch.long, device=input_.device)
    for cp_rank in range(cp_world_size):
        idx1 = cp_rank
        idx2 = 2 * cp_world_size - cp_rank - 1
        reverse_indices[idx1] = 2 * cp_rank
        reverse_indices[idx2] = 2 * cp_rank + 1
    restored_tensor = reshaped_input[reverse_indices]
    restored_shape = (-1, *original_shape[1:])
    output = restored_tensor.contiguous().view(restored_shape)
    return output

def _split_along_first_dim_with_sequence_parallel(input_, split_tp_sp_group, split_cp_group, split_tp_sp_cp_group):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    from galvatron.core import get_args

    args = get_args()

    tp_sp_world_size = 1 if split_tp_sp_group is None else torch.distributed.get_world_size(group=split_tp_sp_group)
    cp_world_size = 1 if split_cp_group is None else torch.distributed.get_world_size(group=split_cp_group)
    tp_sp_cp_world_size = 1 if split_tp_sp_cp_group is None else torch.distributed.get_world_size(group=split_tp_sp_cp_group)

    # Bypass the function if we are using only 1 GPU.
    if tp_sp_cp_world_size == 1:
        return input_   
    if args.sequence_parallel:
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * tp_sp_cp_world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        # get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        handle = torch.distributed._all_gather_base(output, input_, group=split_tp_sp_cp_group)
    else:
        output = input_
    # Zigzag reverse transformation.
    if cp_world_size > 1:
        output = _reverse_zigzag_transformation(output, cp_world_size)

    if args.shape_order == "SBH": 
        output = rearrange(output, "s b h -> b s h")

    # Split along first dimension.
    dim_size = output.size()[0]
    assert dim_size % tp_sp_cp_world_size == 0, "First dimension of the tensor should be divisible by tp*sp*cp parallel size"
    local_dim_size = dim_size // tp_sp_cp_world_size
    rank = torch.distributed.get_rank(group=split_tp_sp_cp_group)
    dim_offset = rank * local_dim_size

    if args.shape_order == "SBH":  # [b, s, h] -> [s, b, h]
        output = output[dim_offset : dim_offset + local_dim_size].permute(1, 0, 2).contiguous()
    else:
        output = output[dim_offset : dim_offset + local_dim_size].contiguous()

    return output.contiguous()

def _gather_along_first_dim_with_sequence_parallel(input_, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group):
    """Gather tensors and concatinate along the first dimension."""
    from galvatron.core import get_args

    args = get_args()

    tp_sp_world_size = 1 if allgather_tp_sp_group is None else torch.distributed.get_world_size(group=allgather_tp_sp_group)
    cp_world_size = 1 if allgather_cp_group is None else torch.distributed.get_world_size(group=allgather_cp_group)
    tp_sp_cp_world_size = 1 if allgather_tp_sp_cp_group is None else torch.distributed.get_world_size(group=allgather_tp_sp_cp_group)
    # Bypass the function if we are using only 1 GPU.
    if tp_sp_cp_world_size == 1:
        return input_

    if args.shape_order == "SBH":  # [s, b, h] -> [b, s, h]
        input_ = rearrange(input_, "s b h -> b s h")

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * tp_sp_cp_world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())

    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=allgather_tp_sp_cp_group)

    if args.shape_order == "SBH":  # [s, b, h] -> [b, s, h]
        output = rearrange(output, "b s h -> s b h")
    # else:
    #     if args.sequence_parallel:
    #         output = rearrange(output, "b s h -> (b s) h")
    # Zigzag transformation.
    if cp_world_size > 1:
        output = _zigzag_transformation(output, cp_world_size)

    if args.sequence_parallel:
        dim_size = output.size()[0]
        assert dim_size % tp_sp_cp_world_size == 0, "First dimension of the tensor should be divisible by tp*sp*cp parallel size"
        local_dim_size = dim_size // tp_sp_cp_world_size
        #sp_rank = torch.distributed.get_rank(group=allgather_tp_sp_group)
        #print("device",torch.cuda.current_device(),"sp_rank",sp_rank)
        #cp_rank = torch.distributed.get_rank(group=allgather_cp_group)
        #print("device",torch.cuda.current_device(),"cp_rank",cp_rank)
        #dim_offset = sp_rank * local_dim_size + cp_rank * local_dim_size * tp_sp_world_size
        rank = torch.distributed.get_rank(group=allgather_tp_sp_cp_group)
        dim_offset = rank * local_dim_size
        output = output[dim_offset : dim_offset + local_dim_size].contiguous()
    return output.contiguous()

def _split_along_first_dim(input_, split_tp_sp_cp_group):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    tp_sp_cp_world_size = 1 if split_tp_sp_cp_group is None else torch.distributed.get_world_size(group=split_tp_sp_cp_group)
    # Bypass the function if we are using only 1 GPU.
    if tp_sp_cp_world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % tp_sp_cp_world_size == 0, "First dimension of the tensor should be divisible by tp*sp*cp parallel size"
    local_dim_size = dim_size // tp_sp_cp_world_size
    rank = torch.distributed.get_rank(group=split_tp_sp_cp_group)
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_first_dim(input_, allgather_tp_sp_cp_group):
    """Gather tensors and concatinate along the first dimension."""

    tp_sp_cp_world_size = 1 if allgather_tp_sp_cp_group is None else torch.distributed.get_world_size(group=allgather_tp_sp_cp_group)
    # Bypass the function if we are using only 1 GPU.
    if tp_sp_cp_world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * tp_sp_cp_world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=allgather_tp_sp_cp_group)

    return output

class _Split(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    # @staticmethod
    # def symbolic(graph, input_, group):
    #     return _split_along_first_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, split_tp_sp_group, split_cp_group, split_tp_sp_cp_group, is_input):
        ctx.split_tp_sp_group = split_tp_sp_group
        ctx.split_cp_group = split_cp_group
        ctx.split_tp_sp_cp_group = split_tp_sp_cp_group
        ctx.is_input = is_input
        if is_input is False:
            return _split_along_first_dim(input_, split_tp_sp_cp_group)
        else:
            return _split_along_first_dim_with_sequence_parallel(input_, split_tp_sp_group, split_cp_group, split_tp_sp_cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_input is False:
            return _gather_along_first_dim(grad_output, ctx.split_tp_sp_cp_group), None, None, None, None
        else:
            return _gather_along_first_dim_with_sequence_parallel(grad_output, ctx.split_tp_sp_group, ctx.split_cp_group, ctx.split_tp_sp_cp_group), None, None, None, None


class _Gather(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    # @staticmethod
    # def symbolic(graph, input_):
    #     return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, is_input):
        ctx.allgather_tp_sp_group = allgather_tp_sp_group
        ctx.allgather_cp_group = allgather_cp_group
        ctx.allgather_tp_sp_cp_group = allgather_tp_sp_cp_group
        ctx.is_input = is_input
        if is_input is False:
            return _gather_along_first_dim(input_, allgather_tp_sp_cp_group)
        else:
            return _gather_along_first_dim_with_sequence_parallel(input_, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_input is False:
            return _split_along_first_dim(grad_output, ctx.allgather_tp_sp_cp_group), None, None, None, None
        else:
            return _split_along_first_dim_with_sequence_parallel(grad_output, ctx.allgather_tp_sp_group, ctx.allgather_cp_group, ctx.allgather_tp_sp_cp_group), None, None, None, None


def split_to_group(input_, split_tp_sp_group, split_cp_group, split_tp_sp_cp_group, is_input):
    return _Split.apply(input_, split_tp_sp_group, split_cp_group, split_tp_sp_cp_group, is_input)


def gather_from_group(input_, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, is_input):
    return _Gather.apply(input_, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, is_input)

def _fused_split_allgather_along_first_dim(
    input_, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, 
    split_tp_sp_group, split_cp_group, split_tp_sp_cp_group,
    fused_allgather_group, fused_split_group
):

    if fused_split_group is not None:
        group = fused_split_group
        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_

        # Split along first dimension.
        dim_size = input_.size()[0]
        assert dim_size % world_size == 0, "First dimension of the tensor should be divisible by tensor parallel size"
        local_dim_size = dim_size // world_size
        rank = torch.distributed.get_rank(group=group)
        dim_offset = rank * local_dim_size

        output = input_[dim_offset : dim_offset + local_dim_size].contiguous()
        return output

    if fused_allgather_group is not None:
        group = fused_allgather_group
        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_

        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
        return output
    return input_

def _fused_split_allgather_along_first_dim_with_sequence_parallel(
    input_, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, 
    split_tp_sp_group, split_cp_group, split_tp_sp_cp_group,
    fused_allgather_group, fused_split_group
):
    from galvatron.core import get_args

    args = get_args()

    split_tp_sp_cp_world_size = 1 if split_tp_sp_cp_group is None else torch.distributed.get_world_size(group=split_tp_sp_cp_group)
    # Bypass the function if we are using only 1 GPU.
    # if world_size == 1:
    #     return input_
    if args.sequence_parallel and split_tp_sp_cp_group is not None and split_tp_sp_cp_world_size > 1:
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * split_tp_sp_cp_world_size
        output_ = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        # get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        torch.distributed.all_gather_into_tensor(output_, input_.contiguous(), group=split_tp_sp_cp_group)
    else:
        output_ = input_.contiguous()
    old_cp_world_size = 1 if split_cp_group is None else torch.distributed.get_world_size(group=split_cp_group)
    new_cp_world_size = 1 if allgather_cp_group is None else torch.distributed.get_world_size(group=allgather_cp_group)
    if old_cp_world_size != new_cp_world_size:
        if old_cp_world_size > 1:
            output_ = _reverse_zigzag_transformation(output_, old_cp_world_size)
        if new_cp_world_size > 1:
            output_ = _zigzag_transformation(output_, new_cp_world_size)

    if args.shape_order == "SBH":  # [s, b, h] -> [b, s, h]
        output_ = rearrange(output_, "s b h -> b s h")

    if fused_split_group is not None or fused_allgather_group is not None:
        if fused_split_group is not None:
            # Split along first dimension.
            world_size = torch.distributed.get_world_size(group=fused_split_group)
            dim_size = output_.size()[0]
            # print("dim_size", dim_size, "world_size", world_size)
            assert dim_size % world_size == 0, "First dimension of the tensor should be divisible by fused_split_group size"
            local_dim_size = dim_size // world_size
            rank = torch.distributed.get_rank(group=fused_split_group)
            dim_offset = rank * local_dim_size

            output = output_[dim_offset : dim_offset + local_dim_size].contiguous()

        if fused_allgather_group is not None:

            world_size = torch.distributed.get_world_size(group=fused_allgather_group)

            dim_size = list(output_.size())
            dim_size[0] = dim_size[0] * world_size

            output = torch.empty(dim_size, dtype=output_.dtype, device=torch.cuda.current_device())
            # print(world_size,output.shape, output_.contiguous().shape,fused_allgather_group,fused_split_group)
            # print(torch.distributed.get_rank(group=fused_allgather_group),torch.cuda.current_device(),fused_allgather_group)
            # torch.distributed.barrier(group=allgather_group)
            # print("begin!",torch.cuda.current_device())
            torch.distributed.all_gather_into_tensor(output, output_.contiguous(), group=fused_allgather_group)
            # print("end!",torch.cuda.current_device())
    else:
        output = output_
    if args.shape_order == "SBH":  # [b, s, h] -> [s, b, h]
        output = rearrange(output, "b s h -> s b h")
    # else:
    #     if args.sequence_parallel:
    #         output = rearrange(output, "b s h -> (b s) h")
    if args.sequence_parallel:
        dim_size = output.size()[0]
        tp_sp_cp_world_size = 1 if allgather_tp_sp_cp_group is None else torch.distributed.get_world_size(group=allgather_tp_sp_cp_group)
        tp_sp_world_size = 1 if allgather_tp_sp_group is None else torch.distributed.get_world_size(group=allgather_tp_sp_group)
        assert dim_size % tp_sp_cp_world_size == 0, "First dimension of the tensor should be divisible by tp*sp*cp parallel size"
        local_dim_size = dim_size // tp_sp_cp_world_size
        #sp_rank = torch.distributed.get_rank(group=allgather_tp_sp_group)
        #cp_rank = torch.distributed.get_rank(group=allgather_cp_group)
        #dim_offset = sp_rank * local_dim_size + cp_rank * local_dim_size * tp_sp_world_size
        if tp_sp_cp_world_size > 1:
            rank = torch.distributed.get_rank(group=allgather_tp_sp_cp_group)
            dim_offset = rank * local_dim_size
            output = output[dim_offset : dim_offset + local_dim_size].contiguous()
    # print(input_.shape, output.shape)
    # print(output.shape, output.stride(), torch.cuda.current_device())
    return output.contiguous()



class _Fused_split_allgather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, is_input, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, 
    split_tp_sp_group, split_cp_group, split_tp_sp_cp_group,
    fused_allgather_group, fused_split_group):
        ctx.allgather_tp_sp_group = allgather_tp_sp_group
        ctx.allgather_cp_group = allgather_cp_group
        ctx.allgather_tp_sp_cp_group = allgather_tp_sp_cp_group
        ctx.split_tp_sp_group = split_tp_sp_group
        ctx.split_cp_group = split_cp_group
        ctx.split_tp_sp_cp_group = split_tp_sp_cp_group
        ctx.fused_allgather_group = fused_allgather_group
        ctx.fused_split_group = fused_split_group
        ctx.is_input = is_input
        if is_input is False:
            return _fused_split_allgather_along_first_dim(
                input_, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, 
                split_tp_sp_group, split_cp_group, split_tp_sp_cp_group,
                fused_allgather_group, fused_split_group
            )
        else:
            return _fused_split_allgather_along_first_dim_with_sequence_parallel(
                input_, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, 
                split_tp_sp_group, split_cp_group, split_tp_sp_cp_group,
                fused_allgather_group, fused_split_group
            )

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_input is False:
            return (
                _fused_split_allgather_along_first_dim(
                    grad_output, ctx.split_tp_sp_group, ctx.split_cp_group, ctx.split_tp_sp_cp_group, 
                    ctx.fused_split_group, ctx.fused_allgather_group
                ),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            return (
                _fused_split_allgather_along_first_dim_with_sequence_parallel(
                    grad_output, ctx.split_tp_sp_group, ctx.split_cp_group, ctx.split_tp_sp_cp_group, 
                    ctx.allgather_tp_sp_group, ctx.allgather_cp_group, ctx.allgather_tp_sp_cp_group,
                    ctx.fused_split_group, ctx.fused_allgather_group
                ),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


def fused_split_allgather(input_, is_input, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, 
    split_tp_sp_group, split_cp_group, split_tp_sp_cp_group,
    fused_allgather_group, fused_split_group):
    return _Fused_split_allgather.apply(
        input_, is_input, allgather_tp_sp_group, allgather_cp_group, allgather_tp_sp_cp_group, 
        split_tp_sp_group, split_cp_group, split_tp_sp_cp_group,
        fused_allgather_group, fused_split_group
    )
