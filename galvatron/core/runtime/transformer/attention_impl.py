# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import math
import os
from typing import Optional, Any, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func as flash_attn_unpadded_func,
        )
    except ImportError:
        flash_attn_unpadded_func = None


def _galvatron_flash_deterministic() -> bool:
    """True when accuracy_alignment scripts export GALVATRON_DETERMINISTIC=1 (default)."""
    v = os.environ.get("GALVATRON_DETERMINISTIC", "").strip().lower()
    return v in ("1", "true", "yes", "on")


# --------- flash attention impl --------------
class FlashSelfOrCrossAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, (
            "Please install FlashAttention first, " "e.g., with pip install flash-attn"
        )
        assert rearrange is not None, "Please install einops first, e.g., with pip install einops"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        if flash_attn_unpadded_func is None:
            raise ImportError("FlashAttention is not installed, please install with " "pip install flash-attn")
        if rearrange is None:
            raise ImportError("einops is not installed, please install with pip install einops")


    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)

        is_causal = self.causal
        if seqlen_k == seqlen_q:
            cu_seqlens_k = cu_seqlens_q
        else:
            cu_seqlens_k = torch.arange(
                0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k.device
            )
        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0
        # if self.training:
        #     # during training q,k,v always have same seqlen
        #     assert seqlen_k == seqlen_q

        #     is_causal = self.causal
        #     cu_seqlens_k = cu_seqlens_q
        #     dropout_p = self.dropout_p
        # else:
        #     # turn off FA causal mask after first inference autoregressive iteration
        #     # only on first autoregressive step q,k,v have same seqlen
        #     is_causal = seqlen_q == seqlen_k
        #     cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
        #                 device=q.device)
        #     dropout_p = 0

        output = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
            deterministic=_galvatron_flash_deterministic(),
        )

        output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
        return output
    
# ------- ulysses --------------

def post_all2all(scatter_idx, batch_dim_idx, seq_world_size, bs, seq_len, num_head, head_dim):

    def post_func(input):
        if batch_dim_idx == 0:
            # b, s, n, h
            if scatter_idx < 2:
                output = input.permute(1, 2, 0, 3, 4).contiguous()
                output = output.reshape(bs, seq_len // seq_world_size, seq_world_size * num_head, head_dim).contiguous()
            else:
                output = input.permute(1, 0, 2, 3, 4).contiguous()
                output = output.reshape(bs, seq_world_size * seq_len, num_head // seq_world_size, head_dim).contiguous()
        else:
            # s, b, n, h
            if scatter_idx < 2:
                output = input.transpose(0, 1).transpose(1, 2).contiguous()
                # output = input.permute(1, 2, 0, 3, 4).contiguous()
                output = output.reshape(seq_len // seq_world_size, bs, seq_world_size * num_head, head_dim).contiguous()
            else:
                output = input.reshape(seq_len * seq_world_size, bs, num_head // seq_world_size, head_dim).contiguous()
        return output

    return post_func


def single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, async_op=False, handle=None, type=None):
    seq_world_size = dist.get_world_size(group)
    if batch_dim_idx == 0:
        # b, s, n, h
        if scatter_idx < 2:
            bs, global_seq_len, num_local_head, head_dim = input.shape
            input_t = input.reshape(
                [bs, seq_world_size, global_seq_len // seq_world_size, num_local_head, head_dim]
            ).contiguous()
            input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        else:
            bs, local_seq_len, num_total_head, head_dim = input.shape
            assert (
                num_total_head % seq_world_size == 0
            ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape(
                [bs, local_seq_len, seq_world_size, num_total_head // seq_world_size, head_dim]
            ).contiguous()
            input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    else:
        # s, b, n, h
        if scatter_idx < 2:
            global_seq_len, bs, num_local_head, head_dim = input.shape
            input_t = input.reshape(
                [seq_world_size, global_seq_len // seq_world_size, bs, num_local_head, head_dim]
            ).contiguous()
        else:
            local_seq_len, bs, num_total_head, head_dim = input.shape
            assert (
                num_total_head % seq_world_size == 0
            ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape(
                [local_seq_len * bs, seq_world_size, num_total_head // seq_world_size, head_dim]
            ).contiguous()
            input_t = input_t.transpose(0, 1).contiguous()
            # input_t = input.reshape([local_seq_len, bs, seq_world_size, num_total_head // seq_world_size,
            #                          head_dim]).contiguous()
            # input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()

    if scatter_idx < 2:
        post_all2all_fun = post_all2all(
            scatter_idx, batch_dim_idx, seq_world_size, bs, global_seq_len, num_local_head, head_dim
        )
    else:
        post_all2all_fun = post_all2all(
            scatter_idx, batch_dim_idx, seq_world_size, bs, local_seq_len, num_total_head, head_dim
        )

    output = torch.empty_like(input_t)
    work = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    if async_op:
        if type in ("dq", "dk"):
            handle[type + "_work"] = work
            handle[type + "_grad"] = output
            handle[type + "_post_all2all_func"] = post_all2all_fun
            return output

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        batch_dim_idx: int,
        stream=None,
        handle=None,
        type=None,
        is_fwd=True,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.stream = stream
        ctx.handle = handle
        ctx.type = type
        ctx.batch_dim_idx = batch_dim_idx
        if ctx.handle is None:
            res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)

        else:
            assert False
            # TODO: support overlap
            # overlap communication path
            if not is_fwd and type == "o":
                assert ctx.stream != None
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)
                get_accelerator().current_stream().wait_stream(ctx.stream)
                del ctx.stream.activation_buffer_list
                # The computation of d o_weight can overlap with the communication of d o_input

            elif not is_fwd and type in ("q", "k"):
                # Achieve communication overlap by pipelining the matrix computation and communication of dq, dk, and dv
                type = "d" + type
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, True, handle, type)

            elif is_fwd and type in ("q", "k"):
                # Achieve communication overlap by pipelining the matrix computation and communication of q, k, and v
                type = "fwd_" + type
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False, handle, type)

            else:
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)

        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:

        return (
            None,
            _SeqAllToAll.apply(
                ctx.group,
                *grad_output,
                ctx.gather_idx,
                ctx.scatter_idx,
                ctx.batch_dim_idx,
                ctx.stream,
                ctx.handle,
                ctx.type,
                False,
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: torch.nn.Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,
        gather_idx: int = 0,
        sp_stream=None,
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.sp_overlap_comm = False
        self.overlap_handles = None
        self.sp_stream = sp_stream
        if sp_stream is not None:
            assert False, "sp_stream is not supported"
            # TODO: support overlap
            self.overlap_handles = {}
            self.sp_overlap_comm = True
            self.dafult_stream = get_accelerator().default_stream()

    def layer_sync(self, layer):
        if self.sp_overlap_comm and hasattr(layer, "done_event"):
            self.dafult_stream.wait_event(layer.done_event)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, batch_dim_idx: int, *args: Any, **kwargs) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            batch_dim_idx (int): indicating which dim is batch
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        num_query_groups = key.shape[2]
        sp_world_size = torch.distributed.get_world_size(self.spg)
        if num_query_groups >= sp_world_size:
            assert num_query_groups % sp_world_size == 0, "num_query_groups % sp_world_size != 0"
        else:
            assert sp_world_size % num_query_groups == 0, "sp_world_size % num_query_groups != 0"
        if num_query_groups < sp_world_size:
            key = key.repeat_interleave(
                sp_world_size // num_query_groups, dim=2
            )
            value = value.repeat_interleave(
                sp_world_size // num_query_groups, dim=2
            )
            
        def bwd_hook(layer_type):

            def pre_hook_fun(grad):
                type = "d" + layer_type
                self.overlap_handles[type + "_work"].wait()
                self.sp_stream.wait_stream(self.dafult_stream)
                all2all_output = self.overlap_handles[type + "_grad"]
                grad = list(grad)
                grad[0] = self.overlap_handles[type + "_post_all2all_func"](all2all_output)
                grad = tuple(grad)

            return pre_hook_fun

        if torch.distributed.get_world_size(self.spg) > 1:
            self.layer_sync(query)
            query_layer = _SeqAllToAll.apply(
                self.spg, query, self.scatter_idx, self.gather_idx, batch_dim_idx, None, self.overlap_handles, "q"
            )
            self.layer_sync(key)
            key_layer = _SeqAllToAll.apply(
                self.spg, key, self.scatter_idx, self.gather_idx, batch_dim_idx, None, self.overlap_handles, "k"
            )
            if self.sp_overlap_comm:
                self.dafult_stream.wait_stream(self.sp_stream)
            value_layer = _SeqAllToAll.apply(
                self.spg, value, self.scatter_idx, self.gather_idx, batch_dim_idx, None, self.overlap_handles, "v"
            )
            if self.sp_overlap_comm:
                # Register a hook to synchronize dq and dk after the all-to-all
                # operation when the gradient data is used.
                # Place this logic after the q, k, v all-to-all operation to
                # improve interpreter speed to
                # call and launch of the forward all-to-all communication.
                grad_fn_q = query.grad_fn.next_functions[0][0]
                grad_fn_q.register_prehook(bwd_hook(layer_type="q"))
                grad_fn_k = key.grad_fn.next_functions[0][0]
                grad_fn_k.register_prehook(bwd_hook(layer_type="k"))
        else:
            query_layer, key_layer, value_layer = query, key, value

        # out shape : e.g., [s:h/p:]
        head_dim = query_layer.shape[-1]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], -1, head_dim)
        if torch.distributed.get_world_size(self.spg) > 1:
            output = _SeqAllToAll.apply(
                self.spg,
                context_layer,
                self.gather_idx,
                self.scatter_idx,
                batch_dim_idx,
                self.sp_stream,
                self.overlap_handles,
                "o",
            )
        else:
            output = context_layer
        # out e.g., [s/p::h]
        return output


# --------- Zigzag Ring Flash Attention --------------
# Reference: https://github.com/zhuzilin/ring-flash-attention/
# We make some modifications to the original code to adapt to make computation and communication overlap better.
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import inspect
from functools import cache

@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args

def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse

#TODO：for other nccl version，we can use different nccl stream to overlap communication and computation
class RingComm:
    def __init__(self, process_group: dist.ProcessGroup, batch_comm = True):
        self.batch_comm = batch_comm
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self._send_reqs = []
        self._recv_reqs = []

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor
        if self.batch_comm:
            send_op = dist.P2POp(
                dist.isend, to_send, self.send_rank, group=self._process_group
            )
            recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
            self._ops.append(send_op)
            self._ops.append(recv_op)
        else:
            if self.rank % 2 == 0:
                send_req = dist.isend(to_send, self.send_rank, group=self._process_group)
                recv_req = dist.irecv(res, self.recv_rank, group=self._process_group)
            else:
                recv_req = dist.irecv(res, self.recv_rank, group=self._process_group)
                send_req = dist.isend(to_send, self.send_rank, group=self._process_group)
            self._recv_reqs.append(recv_req)
            self._send_reqs.append(send_req)
        return res

    def commit(self):
        if self.batch_comm:
            if self._reqs is not None:
                raise RuntimeError("commit called twice")
            self._reqs = dist.batch_isend_irecv(self._ops)
        else:
            pass

    def wait(self):
        if self.batch_comm:
            if self._reqs is None:
                raise RuntimeError("wait called before commit")
            for req in self._reqs:
                req.wait()
            self._reqs = None
            self._ops = []
        else:
            for req in self._recv_reqs:
                req.wait()
            self._send_reqs.clear()
            self._recv_reqs.clear()

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v
    

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


def zigzag_ring_flash_attn_forward(
    process_group,
    ranks,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    lse = None
    next_k, next_v = None, None

    def forward(q, k, v, causal):
        params = get_default_args(_flash_attn_forward).copy()
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        if "deterministic" in params:
            params["deterministic"] = deterministic
        outputs = _flash_attn_forward(**params)
        if len(outputs) == 8:
            block_out, _, _, _, _, block_lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
        return block_out, block_lse

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        # TODO: Maybe find a better way to make sure launch order
        if step == 0:
            _ = torch.zeros((1,),device=torch.cuda.current_device())#we use this to guarantee commiunication is launched before computation
            block_out, block_lse = forward(q, k, v, causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            _ = torch.zeros((1,),device=torch.cuda.current_device())#we use this to guarantee commiunication is launched before computation
            block_out, block_lse = forward(q, k0, v0, causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            _ = torch.zeros((1,),device=torch.cuda.current_device())#we use this to guarantee commiunication is launched before computation
            block_out, block_lse = forward(q1, k, v, causal=False)
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def zigzag_ring_flash_attn_backward(
    process_group,
    ranks,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    #d_kv_comm = RingComm(process_group)

    # dkv_comm_ranks = ranks
    # d_kv_comm_group = dist.new_group(dkv_comm_ranks)
    # d_kv_comm = RingComm(d_kv_comm_group)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None
    #TODO:for other nccl version,we may can use different nccl stream to overlap communication and computation
    # kv_comm_stream = torch.cuda.Stream(device=q.device)
    # d_kv_comm_stream = torch.cuda.Stream(device=q.device)

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    original_dtype = q.dtype

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        params = get_default_args(_flash_attn_backward).copy()
        params.update(
            {
                "dout": dout,
                "q": q,
                "k": k,
                "v": v,
                "out": out,
                "softmax_lse": softmax_lse,
                "dq": dq_buffer[:, :seqlen_q],
                "dk": dk_buffer[:, :seqlen_kv],
                "dv": dv_buffer[:, :seqlen_kv],
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
            }
        )
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        _flash_attn_backward(**params)

    for step in range(kv_comm.world_size):
        if step == 0:
            next_k, next_v = kv_comm.send_recv_kv(k, v)
        else:
            if step + 1 != kv_comm.world_size:
                k_dk = torch.stack([k, dk], dim=0)
                v_dv = torch.stack([v, dv], dim=0)
                next_k_dk, next_v_dv = kv_comm.send_recv_kv(k_dk, v_dv)
            else:
                next_dk, next_dv = kv_comm.send_recv_kv(dk, dv)
        
        if step == 0:
            backward(dout, q, k, v, out, softmax_lse, causal=True)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                dq += dq_buffer
            else:
                backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                # always use the first half in dq_buffer.
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            #d_kv_comm.wait()
            kv_comm.wait()
            if step + 1 != kv_comm.world_size:
                next_k, next_v = next_k_dk[0].to(original_dtype), next_v_dv[0].to(original_dtype)
                next_dk, next_dv = next_k_dk[1], next_v_dv[1]
                k, v = next_k, next_v
                dk_comm_buffer, dv_comm_buffer = dk, dv
                dk, dv = next_dk, next_dv
            else:
                dk, dv = next_dk, next_dv
            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step == 0:
            kv_comm.wait()
            k, v = next_k, next_v
    next_dk, next_dv = kv_comm.send_recv_kv(dk, dv, dk_comm_buffer, dv_comm_buffer)
    kv_comm.wait()
    dk, dv = next_dk, next_dv

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        ranks,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = zigzag_ring_flash_attn_forward(
            group,
            ranks,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.ranks = ranks
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = zigzag_ring_flash_attn_backward(
            ctx.group,
            ctx.ranks,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None




def zigzag_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    ranks=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        ranks,
    )


class ZigzagRingFlashAttention(torch.nn.Module):
    def __init__(self, attention_dropout, cp_group, cp_ranks, softmax_scale=None, causal=True):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.cp_process_group = cp_group 
        self.cp_ranks = cp_ranks
        self.causal = causal

    def forward(self, q, k, v):
        assert q.dim() == 4, "q should be [B, S, H, D]"
        softmax_scale = self.softmax_scale
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        
        with torch.profiler.record_function("ZigZag_Ring_Flash_Attention_Forward"):
            context = zigzag_ring_flash_attn_func(
                q, k, v,
                dropout_p=self.attention_dropout,
                softmax_scale=softmax_scale,
                causal=self.causal,
                deterministic=_galvatron_flash_deterministic(),
                group=self.cp_process_group,
                ranks=self.cp_ranks,
            )
        return context