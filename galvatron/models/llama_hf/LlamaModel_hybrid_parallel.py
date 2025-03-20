from transformers import LlamaForCausalLM

from galvatron.core import (
    RuntimeProfiler,
    construct_hybrid_parallel_model_api,
    get_hybrid_parallel_configs_api,
    init_empty_weights,
)
from galvatron.models.llama_hf.LlamaModel_checkpoint import load_llama_module
from galvatron.models.llama_hf.LlamaModel_sequential import (
    LlamaCls_,
    LlamaEmbeddings_,
    LlamaModelInfo,
    LlamaPreNorm_,
    construct_sequential_model,
)
from galvatron.models.llama_hf.LlamaModel_tensor_parallel import LlamaLayer_tp, LlamaAttention_tp, LlamaMLP_tp, construct_tensor_parallel_model
from galvatron.models.llama_hf.meta_configs import config_from_meta, model_layer_configs, model_name, set_model_config

# from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm


def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, LlamaModelInfo)
    return hybrid_parallel_configs


def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    wrap_block_name = [LlamaLayer_tp]
    wrap_checkpoint_block_name = [LlamaLayer_tp]
    fine_wrap_block_name = [LlamaAttention_tp, LlamaMLP_tp]
    wrap_other_block_name = [LlamaEmbeddings_, LlamaPreNorm_, LlamaCls_]
    all_block_name = [LlamaEmbeddings_, LlamaLayer_tp, LlamaPreNorm_, LlamaCls_, LlamaAttention_tp, LlamaMLP_tp]
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        LlamaModelInfo,
        construct_sequential_model,
        construct_tensor_parallel_model,
        wrap_block_name=wrap_block_name,
        wrap_checkpoint_block_name=wrap_checkpoint_block_name,
        wrap_other_block_name=wrap_other_block_name,
        fine_wrap_block_name=fine_wrap_block_name,
        # tied_wte_attr_names=['embed_tokens', 'lm_head'],
        layernorm_name=["LayerNorm", "norm"],
        all_block_name=all_block_name,
        load_module_func=load_llama_module,
    )
    return hp_model


def get_llama_config(args, overwrite_args=True):
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args)
    if hasattr(args, "local_rank") and args.local_rank == 0:
        print(config)
    return config


def llama_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")

    if args.initialize_on_meta:
        with init_empty_weights():
            llama_model = LlamaForCausalLM(config)
    else:
        llama_model = LlamaForCausalLM(config)

    model = construct_hybrid_parallel_model(
        model=llama_model, model_config=config, training_args=args, hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model


def get_runtime_profiler(args, path, config, start_iter=10, end_iter=20):
    profiler = RuntimeProfiler(args)
    profiler.set_profiler_dist(
        path, model_layer_configs(config), model_name(config), start_iter=start_iter, end_iter=end_iter
    )
    return profiler
