"""High-level model construction API.

Provides functions to build hybrid-parallel models from a declarative
architecture list, as well as convenience helpers for profiling.

Key entry points:
    - ``build_model(args)``:  one-call model builder (resolve → arch → HP model)
    - ``build_sequential_from_arch(...)``:  lower-level PipeSequential builder
    - ``build_causal_lm_arch(args)``:  generate arch list for decoder-only LMs
    - ``get_hybrid_parallel_configs(args)``:  auto-derive HP configs
    - ``get_runtime_profiler(args, path)``:  create a RuntimeProfiler
"""

from typing import List, Optional

from galvatron.core.runtime.pipeline import PipeSequential

from .modules import (
    GalvatronEmbedding,
    GalvatronDecoderLayer,
    GalvatronAttention,
    GalvatronMLP,
    GalvatronFinalNorm,
    GalvatronCausalLMHead,
    GalvatronMoEAttention,
    GalvatronMoEMLP,
    GalvatronMoERouter,
)
from .arch import (
    MODULE_REGISTRY,
    _LAYER_MODULE_TYPES,
    ArchModelInfo,
)
from ..args_schema import GalvatronRuntimeArgs
from .arch import BlockNames

from galvatron.core.runtime.checkpoint.llama_adapter import load_llama_module
from galvatron.core.runtime.checkpoint.gpt_adapter import load_gpt_module
from galvatron.core.runtime.checkpoint.moe_adapter import load_moe_module


def build_sequential_from_arch(
    arch_list: List[str],
    args:GalvatronRuntimeArgs,
    tp_groups: List,
    sp_groups: List,
    cp_groups: List,
    ep_groups: Optional[List] = None,
    tp_of_ep_groups: Optional[List] = None,
    tp_and_ep_groups: Optional[List] = None,
) -> PipeSequential:
    """Build a ``PipeSequential`` model directly from an architecture list.

    Each element in *arch_list* is mapped to a TP-aware module via
    ``MODULE_REGISTRY``.  Layer-type modules (``decoder``, ``moe_decoder``)
    receive an incrementing ``layer_idx``; other modules do not.

    Args:
        arch_list: e.g. ``["embedding", "decoder", ..., "prenorm", "lm_head"]``
        args: Galvatron args (with ``args.model``, ``args.train``, ``args.parallel``)
        tp_groups: per-position TP comm groups
        sp_groups: per-position SP comm groups
        cp_groups: per-position CP comm groups

    Returns:
        A ``PipeSequential`` ready for pipeline-parallel wrapping.
    """
    seq = PipeSequential()
    layer_idx = 0

    for i, module_type in enumerate(arch_list):
        if module_type not in MODULE_REGISTRY:
            raise ValueError(
                f"Unknown module type '{module_type}'. "
                f"Available: {list(MODULE_REGISTRY.keys())}"
            )
        cls = MODULE_REGISTRY[module_type]

        if module_type in _LAYER_MODULE_TYPES:
            cls_kwargs = {
                "args": args,
                "layer_idx": layer_idx,
                "tp_group": tp_groups[i],
                "sp_group": sp_groups[i],
                "cp_group": cp_groups[i],
            }
            if module_type == "moe_decoder":
                cls_kwargs["ep_group"] = ep_groups[i]
                cls_kwargs["tp_of_ep_group"] = tp_of_ep_groups[i]
                cls_kwargs["tp_and_ep_group"] = tp_and_ep_groups[i]
            module = cls(**cls_kwargs)
            layer_idx += 1
        elif module_type in ("embedding", "lm_head"):
            module = cls(
                args,
                tp_group=tp_groups[i],
                sp_group=sp_groups[i],
                cp_group=cp_groups[i],
            )
        elif module_type in ("prenorm"):
            module = cls(
                args,
            )
        else:
            assert False, "Unknown module type: " + module_type

        seq.add_module(f"{module_type}_{i}", module)
    return seq


def build_causal_lm_arch(args:GalvatronRuntimeArgs) -> List[str]:
    """Build architecture list for a standard decoder-only causal LM."""

    if args.model.model_type in ["gpt", "llama", "qwen"]:
        num_layers = args.model.num_layers
        return ["embedding"] + ["decoder"] * num_layers + ["prenorm", "lm_head"]
    elif args.model.model_type in ["mistral", "mixtral"]:
        num_layers = args.model.num_layers
        return ["embedding"] + ["moe_decoder"] * num_layers + ["prenorm", "lm_head"]
    else:
        assert False, "Unknown model type: " + args.model.model_type


def get_block_names(args:GalvatronRuntimeArgs):
    """Derive FSDP/checkpoint wrapping class lists from model type."""
    if args.model.model_type in ["gpt", "llama", "qwen"]:
        # When profiling attention/MLP units separately, wrap the
        # attention and MLP blocks directly; otherwise wrap the whole
        # decoder layer as a unit.
        if args.profile.profile_unit in ("attention", "mlp"):
            return BlockNames(
                wrap_block_name=[GalvatronAttention, GalvatronMLP],
                wrap_checkpoint_block_name=[GalvatronAttention, GalvatronMLP],
                wrap_other_block_name=[GalvatronEmbedding, GalvatronFinalNorm, GalvatronCausalLMHead],
                all_block_name=[GalvatronEmbedding, GalvatronAttention, GalvatronMLP, GalvatronFinalNorm, GalvatronCausalLMHead],
            )
        else:
            return BlockNames(
                wrap_block_name=[GalvatronDecoderLayer],
                wrap_checkpoint_block_name=[GalvatronDecoderLayer],
                wrap_other_block_name=[GalvatronEmbedding, GalvatronFinalNorm, GalvatronCausalLMHead],
                all_block_name=[GalvatronEmbedding, GalvatronDecoderLayer, GalvatronFinalNorm, GalvatronCausalLMHead],
            )
    elif args.model.model_type in ["mistral", "mixtral"]:
        if args.profile.profile_unit in ("attention", "mlp"):
            assert False, "Currently, MoE model does not support profile_unit in ('attention', 'mlp')"
        else:
            return BlockNames(
                wrap_block_name=[GalvatronMoEAttention, GalvatronMoEMLP],
                wrap_checkpoint_block_name=[GalvatronMoEAttention, GalvatronMoEMLP],
                wrap_other_block_name=[GalvatronEmbedding, GalvatronFinalNorm, GalvatronCausalLMHead],
                all_block_name=[GalvatronEmbedding, GalvatronMoEAttention, GalvatronMoEMLP, GalvatronMoERouter, GalvatronFinalNorm, GalvatronCausalLMHead],
            )
    else:
        raise ValueError(f"Unknown model type: {args.model.model_type}")


def build_model(args:GalvatronRuntimeArgs):
    """One-call model builder: arch_list → hybrid-parallel model.

    Call ``resolve_model_config(args)`` before this to populate
    ``args.model.*`` from YAML / HF sources, or set them directly.
    """
    from galvatron.core.runtime.hybrid_parallel_model import construct_hybrid_parallel_model_api
    from galvatron.core.runtime.hybrid_parallel_config import get_hybrid_parallel_configs_api

    arch_list = build_causal_lm_arch(args)
    hybrid_parallel_config = get_hybrid_parallel_configs_api(args)
    model_info = ArchModelInfo(arch_list, args)
    block_names = get_block_names(args)
    if args.model.model_type in ("mistral", "mixtral"):
        load_module_func = load_moe_module
    elif args.model.model_size.startswith("gpt"):
        load_module_func = load_gpt_module
    else:
        load_module_func = load_llama_module

    model = construct_hybrid_parallel_model_api(
        arch_list=arch_list,
        args=args,
        hybrid_parallel_configs=hybrid_parallel_config,
        model_info=model_info,
        layernorm_name=["input_layernorm" ,"post_attention_layernorm", "norm"],
        tied_wte_attr_names=None if args.model.untie_embeddings_and_output_weights else ["embed_tokens", "lm_head"],
        block_names=block_names,
        load_module_func=load_module_func,
    )
    if args.model.is_moe_model:
        import torch
        from galvatron.core.runtime.moe.moe_utils import MoEAuxLossAutoScaler

        loss_scale = torch.ones(1, device=torch.cuda.current_device())
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / args.train.chunks)
    return model


def get_runtime_profiler(args, path, start_iter=10, end_iter=20):
    """Create a ``RuntimeProfiler`` with model info derived from args."""
    from galvatron.core.profiler import RuntimeProfiler
    from galvatron.utils.hf_config_adapter import model_layer_configs, model_name

    profiler = RuntimeProfiler(args)
    profiler.set_profiler_dist(
        path, model_layer_configs(args), model_name(args),
        start_iter=start_iter, end_iter=end_iter,
    )
    return profiler
