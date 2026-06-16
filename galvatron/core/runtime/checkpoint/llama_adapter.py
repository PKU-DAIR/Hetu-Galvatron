import json
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from galvatron.core.runtime.tensor_parallel.utils import VocabUtility
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import MixedPrecision
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

from galvatron.core.runtime.parallel_state import get_args
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
from .utils import copy_to_weight, copy_to_bias

from ..models.modules import (
    GalvatronEmbedding,
    GalvatronDecoderLayer,
    GalvatronFinalNorm,
    GalvatronCausalLMHead,
)

embedding_name = "model_embed_tokens.pt"
layer_name = "model_layers_%d.pt"
ln_f_name = "model_norm.pt"
cls_name = "lm_head.pt"


def load_distributed_checkpoint(load:str, tp_groups: dist.ProcessGroup, name: str, submodule: nn.Module, module: nn.Module):
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)
    args = get_args()
    load = os.path.join(load, f"iter_{args.load_iteration}")
    if name.endswith("embed_tokens"):
        file_path = os.path.join(load, embedding_name[:-3], f"{rank}.pt")
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
    elif name.endswith("norm") and 'input_layernorm' not in name and 'post_attention_layernorm' not in name:
        file_path = os.path.join(load, ln_f_name[:-3], f"{rank}.pt")
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
    elif name.endswith("lm_head"):
        file_path = os.path.join(load, cls_name[:-3], f"{rank}.pt")
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
    else:
        file_path = os.path.join(load, (layer_name % module.idx)[:-3], f"{rank}.pt")
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
    weight = checkpoint[f"{name}.weight"].to(device="cuda", dtype=torch.float32)
    copy_to_weight(submodule, weight)


def load_hf_checkpoint(load:str, tp_groups:dist.ProcessGroup, name:str, submodule:nn.Module, module:nn.Module):
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)
    if name.endswith("embed_tokens"):
        file_path = os.path.join(load, embedding_name)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
        args = get_args()
        vocab_size = checkpoint["embed_tokens.weight"].shape[0]
        padding_size = args.padded_vocab_size - vocab_size
        padded_weight = F.pad(
            checkpoint["embed_tokens.weight"].to(device="cuda", dtype=torch.float32),
            (0, 0, padding_size, 0),
            mode="constant",
            value=0,
        )
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.padded_vocab_size, rank, world_size
        )
        copy_to_weight(submodule, padded_weight[vocab_start_index:vocab_end_index])
    elif name == "norm":
        # Final RMSNorm only (must not use endswith("norm"): that matches input_layernorm / post_attention_layernorm).
        file_path = os.path.join(load, ln_f_name)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
        weight = checkpoint["weight"].to(device="cuda", dtype=torch.float32)
        copy_to_weight(submodule, weight)
    elif name == "lm_head":
        file_path = os.path.join(load, cls_name)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
        args = get_args()
        vocab_size = checkpoint["weight"].shape[0]
        padding_size = args.padded_vocab_size - vocab_size
        padded_weight = F.pad(
            checkpoint["weight"].to(device="cuda", dtype=torch.float32),
            (0, 0, padding_size, 0),
            mode="constant",
            value=0,
        )
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.padded_vocab_size, rank, world_size
        )
        copy_to_weight(submodule, padded_weight[vocab_start_index:vocab_end_index])
    else:
        if not hasattr(module, "idx"):
            raise ValueError(
                f"llama_adapter: unhandled submodule {name!r} under {type(module).__name__} "
                f"(expected embed_tokens, norm, lm_head, or decoder block with idx)"
            )
        file_path = os.path.join(load, layer_name % module.idx)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")

        if "input_layernorm" in name:
            w = checkpoint["input_layernorm.weight"].to(device="cuda", dtype=torch.float32)
            copy_to_weight(submodule, w)
        elif "linear_qkv" in name:
            args = get_args()
            nh = args.num_attention_heads
            ng = args.num_query_groups if args.group_query_attention else args.num_attention_heads
            dim = args.kv_channels
            assert nh % ng == 0
            weight = torch.cat(
                [
                    checkpoint["self_attn.q_proj.weight"].reshape((ng, dim * nh // ng, -1)),
                    checkpoint["self_attn.k_proj.weight"].reshape((ng, dim, -1)),
                    checkpoint["self_attn.v_proj.weight"].reshape((ng, dim, -1)),
                ],
                dim=1,
            ).reshape((-1, args.hidden_size))
            weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                weight.shape[0], rank, world_size
            )
            copy_to_weight(submodule, weight[weight_start_index:weight_end_index].contiguous())
            if getattr(submodule, "bias", None) is not None:
                if "self_attn.q_proj.bias" not in checkpoint:
                    raise ValueError("llama_adapter: linear_qkv has bias but checkpoint has no QKV bias")
                bias = torch.cat(
                    [
                        checkpoint["self_attn.q_proj.bias"].reshape((ng, dim * nh // ng)),
                        checkpoint["self_attn.k_proj.bias"].reshape((ng, dim)),
                        checkpoint["self_attn.v_proj.bias"].reshape((ng, dim)),
                    ],
                    dim=1,
                ).reshape(-1)
                copy_to_bias(
                    submodule,
                    bias[weight_start_index:weight_end_index]
                    .to(device="cuda", dtype=torch.float32)
                    .contiguous()
                )
        elif "linear_proj" in name:
            weight = checkpoint["self_attn.o_proj.weight"].to(device="cuda", dtype=torch.float32)
            weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                weight.shape[1], rank, world_size
            )
            copy_to_weight(submodule, weight[:, weight_start_index:weight_end_index].contiguous())
            if getattr(submodule, "bias", None) is not None and "self_attn.o_proj.bias" in checkpoint:
                b = checkpoint["self_attn.o_proj.bias"].to(device="cuda", dtype=torch.float32)
                copy_to_bias(submodule, b)
        elif "post_attention_layernorm" in name:
            w = checkpoint["post_attention_layernorm.weight"].to(device="cuda", dtype=torch.float32)
            copy_to_weight(submodule, w)
        elif "linear_fc1" in name:
            weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                checkpoint["mlp.gate_proj.weight"].shape[0], rank, world_size
            )
            weight = torch.cat(
                [
                    checkpoint["mlp.gate_proj.weight"][weight_start_index:weight_end_index].contiguous(),
                    checkpoint["mlp.up_proj.weight"][weight_start_index:weight_end_index].contiguous(),
                ],
                dim=0,
            )
            copy_to_weight(submodule, weight.contiguous())
            if getattr(submodule, "bias", None) is not None:
                raise NotImplementedError("llama_adapter: fc1 bias not supported for this layout")
        elif "linear_fc2" in name:
            weight = checkpoint["mlp.down_proj.weight"].to(device="cuda", dtype=torch.float32)
            weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                weight.shape[1], rank, world_size
            )
            copy_to_weight(submodule, weight[:, weight_start_index:weight_end_index].contiguous())
            if getattr(submodule, "bias", None) is not None and "mlp.down_proj.bias" in checkpoint:
                b = checkpoint["mlp.down_proj.bias"].to(device="cuda", dtype=torch.float32)
                copy_to_bias(submodule, b)
        else:
            raise ValueError(f"llama_adapter: unhandled submodule name {name!r} in layer {module.idx}")


@torch.no_grad()
def load_llama_module(load, tp_groups, name, submodule, module, distributed_checkpoint, ep_groups=None):
    if distributed_checkpoint:
        load_distributed_checkpoint(load, tp_groups, name, submodule, module)
    else:
        load_hf_checkpoint(load, tp_groups, name, submodule, module)


@torch.no_grad()
def _module_full_state_dict(m, is_fsdp2):
    """Full (unsharded) state dict of a single wrapped module, on every rank, CPU-offloaded.

    - FSDP2: gather DTensor shards via get_model_state_dict (collective; all ranks must call).
    - FSDP1: relies on the enclosing FSDP.state_dict_type(FULL_STATE_DICT) context.
    """
    if is_fsdp2:
        return get_model_state_dict(m, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return m.state_dict()


def save_llama_module(save_path, model, optimizer, opt_param_scheduler, iter_num, args:GalvatronRuntimeArgs):
    """Save model parameters by layer"""
    rank = torch.distributed.get_rank()
    is_fsdp2 = args.parallel.fsdp_version == "fsdp2"

    if rank == 0:
        print("Begin to save ckpt")
        os.makedirs(save_path, exist_ok=True)
        assert hasattr(model, "hybrid_parallel_configs")
        json.dump(model.hybrid_parallel_configs, open(os.path.join(save_path, "hybrid_parallel_configs.json"), "w"))

        os.makedirs(os.path.join(save_path, "iter_%d" % iter_num), exist_ok=True)
        opt_param_scheduler_state_dict = opt_param_scheduler.state_dict()
        json.dump(
            opt_param_scheduler_state_dict,
            open(os.path.join(save_path, "iter_%d" % iter_num, f"opt_param_scheduler.json"), "w"),
        )

    assert args.parallel.default_dp_type != "ddp", "Save / Load distributed checkpoint is not supported for DDP"
    # FSDP2 (fully_shard) has no state_dict_type context; full tensors are gathered
    # per-module in _module_full_state_dict instead.
    ctx = nullcontext() if is_fsdp2 else FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
    )
    with ctx:

        save_path = os.path.join(save_path, "iter_%d" % iter_num)
        idx = 0
        for block in model.model.model_cur_stage:
            for m in block.modules():
                if isinstance(m, FSDPModule) if is_fsdp2 else isinstance(m, FSDP):
                    # FSDP2 keeps the original module class (no _fsdp_wrapped_module).
                    wrapped_module = m if is_fsdp2 else m._fsdp_wrapped_module
                    if isinstance(wrapped_module, CheckpointWrapper):
                        wrapped_module = wrapped_module._checkpoint_wrapped_module
                    dp_rank = torch.distributed.get_rank(model.sdp_groups_whole[idx].group)
                    tp_rank = torch.distributed.get_rank(model.tp_groups_whole[idx].group)
                    state_dict = _module_full_state_dict(m, is_fsdp2)
                    if dp_rank == 0:
                        if isinstance(wrapped_module, GalvatronEmbedding):
                            os.makedirs(os.path.join(save_path, f"{embedding_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{embedding_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, GalvatronFinalNorm):
                            os.makedirs(os.path.join(save_path, f"{ln_f_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{ln_f_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, GalvatronCausalLMHead):
                            os.makedirs(os.path.join(save_path, f"{cls_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{cls_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, GalvatronDecoderLayer):
                            os.makedirs(
                                os.path.join(save_path, f"{(layer_name%wrapped_module.idx)[:-3]}"), exist_ok=True
                            )
                            torch.save(
                                state_dict,
                                os.path.join(save_path, f"{(layer_name%wrapped_module.idx)[:-3]}/{tp_rank}.pt"),
                            )
            idx += 1

    # Save optimizer
    optimizer_state_dict = optimizer.state_dict()
    os.makedirs(os.path.join(save_path, f"optimizer"), exist_ok=True)
    torch.save(optimizer_state_dict, os.path.join(save_path, f"optimizer/{rank}.pt"))

    torch.distributed.barrier()
    if rank == 0:
        print("Finish saving ckpt")