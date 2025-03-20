import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from megatron.core.tensor_parallel.utils import VocabUtility
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import MixedPrecision

from galvatron.core.arguments import get_args

from .LlamaModel_sequential import LlamaCls_, LlamaEmbeddings_, LlamaPreNorm_
from .LlamaModel_tensor_parallel import LlamaLayer_tp

embedding_name = "model_embed_tokens.pt"
layer_name = "model_layers_%d.pt"
ln_f_name = "model_norm.pt"
cls_name = "lm_head.pt"


def load_distributed_checkpoint(load, tp_groups, name, submodule, module):
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)
    args = get_args()
    load = os.path.join(load, f"iter_{args.load_iteration}")
    if name.endswith("embed_tokens"):
        file_path = os.path.join(load, embedding_name[:-3], f"{rank}.pt")
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
    elif name.endswith("norm"):
        file_path = os.path.join(load, ln_f_name[:-3], f"{rank}.pt")
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
    elif name.endswith("lm_head"):
        file_path = os.path.join(load, cls_name[:-3], f"{rank}.pt")
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
    else:
        file_path = os.path.join(load, (layer_name % module.idx)[:-3], f"{rank}.pt")
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
    weight = checkpoint[f"{name}.weight"].to(device="cuda", dtype=torch.float32)
    submodule.weight.copy_(weight)


def load_hf_checkpoint(load, tp_groups, name, submodule, module):
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
        submodule.weight.copy_(padded_weight[vocab_start_index:vocab_end_index])
    elif name.endswith("norm"):
        file_path = os.path.join(load, ln_f_name)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
        weight = checkpoint["weight"].to(device="cuda", dtype=torch.float32)
        submodule.weight.copy_(weight)
    elif name.endswith("lm_head"):
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
        submodule.weight.copy_(padded_weight[vocab_start_index:vocab_end_index].contiguous())
    else:
        file_path = os.path.join(load, layer_name % module.idx)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
        category = getattr(module, "category", None)
        if torch.cuda.current_device() == 0:
            print(file_path, module, category)
        if name.startswith("attention") or category == "attention":
            if name.endswith("LayerNorm"):
                weight = checkpoint["input_layernorm.weight"].to(device="cuda", dtype=torch.float32)
                submodule.weight.copy_(weight)
            elif name.endswith("query_key_value"):
                args = get_args()
                # q: num_heads * head_dim, hidden_size
                # k,v: num_key_value_heads * head_dim, hidden_size
                # while Megatron stores c_attn.weight as ((nheads 3 headdim), hidden_dim)
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
                submodule.weight.copy_(weight[weight_start_index:weight_end_index].contiguous())
            elif name.endswith("dense"):
                # o: hidden_size, num_heads * head_dim
                weight = checkpoint["self_attn.o_proj.weight"].to(device="cuda", dtype=torch.float32)
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[1], rank, world_size
                )
                submodule.weight.copy_(weight[:, weight_start_index:weight_end_index].contiguous())
        elif name.startswith("mlp") or category == "mlp":
            if name.endswith("LayerNorm"):
                weight = checkpoint["post_attention_layernorm.weight"].to(device="cuda", dtype=torch.float32)
                submodule.weight.copy_(weight)
            elif name.endswith("dense_h_to_4h"):
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

                submodule.weight.copy_(weight.contiguous())
            elif name.endswith("dense_4h_to_h"):
                weight = checkpoint["mlp.down_proj.weight"].to(device="cuda", dtype=torch.float32)
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[1], rank, world_size
                )
                submodule.weight.copy_(weight[:, weight_start_index:weight_end_index].contiguous())


@torch.no_grad()
def load_llama_module(load, tp_groups, name, submodule, module, distributed_checkpoint):
    if distributed_checkpoint:
        load_distributed_checkpoint(load, tp_groups, name, submodule, module)
    else:
        load_hf_checkpoint(load, tp_groups, name, submodule, module)


@torch.no_grad()
def save_llama_module(save_path, model, optimizer, opt_param_scheduler, iter_num, args):
    """Save model parameters by layer"""
    rank = torch.distributed.get_rank()

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

    assert args.default_dp_type != "ddp", "Save / Load distributed checkpoint is not supported for DDP"
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):

        save_path = os.path.join(save_path, "iter_%d" % iter_num)
        idx = 0
        for block in model.model.model_cur_stage:
            for m in block.modules():
                if isinstance(m, FSDP):
                    wrapped_module = m._fsdp_wrapped_module
                    if isinstance(wrapped_module, CheckpointWrapper):
                        wrapped_module = wrapped_module._checkpoint_wrapped_module
                    dp_rank = torch.distributed.get_rank(model.sdp_groups_whole[idx].group)
                    tp_rank = torch.distributed.get_rank(model.tp_groups_whole[idx].group)
                    state_dict = m.state_dict()
                    if dp_rank == 0:
                        if isinstance(wrapped_module, LlamaEmbeddings_):
                            os.makedirs(os.path.join(save_path, f"{embedding_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{embedding_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, LlamaPreNorm_):
                            os.makedirs(os.path.join(save_path, f"{ln_f_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{ln_f_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, LlamaCls_):
                            os.makedirs(os.path.join(save_path, f"{cls_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{cls_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, LlamaLayer_tp):
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
