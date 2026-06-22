import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from galvatron.core.runtime.tensor_parallel.utils import VocabUtility

from galvatron.core.runtime.parallel_state import get_args
from galvatron.core.runtime.checkpoint.utils import copy_to_weight, copy_to_bias

embedding_name = "transformer_embedding.pt"
layer_name = "transformer_h_%d.pt"
ln_f_name = "transformer_ln_f.pt"
cls_name = "transformer_embedding.pt"


@torch.no_grad()
def load_hf_checkpoint(load, tp_groups, name, submodule, module):
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)

    if name.endswith("embed_tokens"):
        file_path = os.path.join(load, embedding_name)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
        args = get_args()
        vocab_size = checkpoint["wte.weight"].shape[0]
        padding_size = args.padded_vocab_size - vocab_size
        padded_weight = F.pad(
            checkpoint["wte.weight"].to(device="cuda", dtype=torch.float32),
            (0, 0, padding_size, 0),
            mode="constant",
            value=0,
        )
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.padded_vocab_size, rank, world_size
        )
        copy_to_weight(submodule,padded_weight[vocab_start_index:vocab_end_index])

    elif name.endswith("embed_positions"):
        file_path = os.path.join(load, embedding_name)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
        weight = checkpoint["wpe.weight"].to(device="cuda", dtype=torch.float32)
        num_rows = submodule.weight.shape[0]
        # GalvatronEmbedding keeps full [seq_len, H] per rank; vocab-TP group can be
        # world_size > 1 while positions are not sharded across that group.
        if num_rows == weight.shape[0]:
            copy_to_weight(submodule,weight)
        else:
            seq_start_index, seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                weight.shape[0], rank, world_size
            )
            copy_to_weight(submodule,weight[seq_start_index:seq_end_index])

    elif name == "norm":
        file_path = os.path.join(load, ln_f_name)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
        weight = checkpoint["weight"].to(device="cuda", dtype=torch.float32)
        bias = checkpoint["bias"].to(device="cuda", dtype=torch.float32)
        copy_to_weight(submodule,weight)
        copy_to_bias(submodule,bias)

    elif name == "lm_head":
        # _LMHeadLinear clones lm_head_proj weights at init; load same slice as lm_head_proj.
        file_path = os.path.join(load, cls_name)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")
        args = get_args()
        vocab_size = checkpoint["wte.weight"].shape[0]
        padding_size = args.padded_vocab_size - vocab_size
        padded_weight = F.pad(
            checkpoint["wte.weight"].to(device="cuda", dtype=torch.float32),
            (0, 0, padding_size, 0),
            mode="constant",
            value=0,
        )
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.padded_vocab_size, rank, world_size
        )
        copy_to_weight(submodule,padded_weight[vocab_start_index:vocab_end_index].contiguous())

    else:
        if not hasattr(module, "idx"):
            raise ValueError(
                f"gpt_adapter: unhandled submodule {name!r} under {type(module).__name__} "
                f"(no layer idx for per-block checkpoint)"
            )
        file_path = os.path.join(load, layer_name % module.idx)
        checkpoint = torch.load(file_path, mmap=True, map_location="cpu")

        if "input_layernorm" in name:
            weight = checkpoint["ln_1.weight"].to(device="cuda", dtype=torch.float32)
            bias = checkpoint["ln_1.bias"].to(device="cuda", dtype=torch.float32)
            copy_to_weight(submodule,weight)
            copy_to_bias(submodule,bias)

        elif "linear_qkv" in name:
            args = get_args()
            weight = checkpoint["attn.c_attn.weight"].to(device="cuda", dtype=torch.float32)
            bias = checkpoint["attn.c_attn.bias"].to(device="cuda", dtype=torch.float32)
            headdim = args.hidden_size // args.num_attention_heads
            weight = rearrange(
                weight.t(),
                "(three nheads headdim) ... -> (nheads three headdim) ...",
                three=3,
                headdim=headdim,
            )
            bias = rearrange(
                bias,
                "(three nheads headdim) ... -> (nheads three headdim) ...",
                three=3,
                headdim=headdim,
            )
            weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                bias.shape[0], rank, world_size
            )
            copy_to_weight(submodule,weight[weight_start_index:weight_end_index].contiguous())
            copy_to_bias(submodule,bias[weight_start_index:weight_end_index].contiguous())

        elif "linear_proj" in name:
            weight = checkpoint["attn.c_proj.weight"].to(device="cuda", dtype=torch.float32)
            bias = checkpoint["attn.c_proj.bias"].to(device="cuda", dtype=torch.float32)
            weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                weight.shape[0], rank, world_size
            )
            copy_to_weight(submodule,weight[weight_start_index:weight_end_index].t().contiguous())
            copy_to_bias(submodule,bias.contiguous())

        elif "post_attention_layernorm" in name:
            weight = checkpoint["ln_2.weight"].to(device="cuda", dtype=torch.float32)
            bias = checkpoint["ln_2.bias"].to(device="cuda", dtype=torch.float32)
            copy_to_weight(submodule,weight)
            copy_to_bias(submodule,bias)

        elif "linear_fc1" in name:
            weight = checkpoint["mlp.c_fc.weight"].to(device="cuda", dtype=torch.float32)
            bias = checkpoint["mlp.c_fc.bias"].to(device="cuda", dtype=torch.float32)
            weight = weight.t()
            weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                weight.shape[0], rank, world_size
            )
            copy_to_weight(submodule,weight[weight_start_index:weight_end_index].contiguous())
            copy_to_bias(submodule,bias[weight_start_index:weight_end_index].contiguous())

        elif "linear_fc2" in name:
            weight = checkpoint["mlp.c_proj.weight"].to(device="cuda", dtype=torch.float32)
            bias = checkpoint["mlp.c_proj.bias"].to(device="cuda", dtype=torch.float32)
            weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                weight.shape[0], rank, world_size
            )
            copy_to_weight(submodule,weight[weight_start_index:weight_end_index].t().contiguous())
            copy_to_bias(submodule,bias.contiguous())


@torch.no_grad()
def load_gpt_module(load, tp_groups, name, submodule, module, distributed_checkpoint, ep_groups=None):
    if distributed_checkpoint:
        raise NotImplementedError("Distributed checkpoint is not supported for GPT")
    else:
        load_hf_checkpoint(load, tp_groups, name, submodule, module)
