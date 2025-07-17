from transformers import MixtralForCausalLM

from galvatron.core import (
    RuntimeProfiler,
    construct_hybrid_parallel_model_api,
    get_hybrid_parallel_configs_api,
    init_empty_weights,
)
from galvatron.models.moe.MoEModel_checkpoint import load_moe_module
from galvatron.models.moe.MoEModel_sequential import (
    MoECls_,
    MoEEmbeddings_,
    MoEModelInfo,
    MoEPreNorm_,
    construct_sequential_model,
)
from galvatron.models.moe.MoEModel_tensor_parallel import MoELayer_tp, construct_tensor_parallel_model, MoEAttention_tp, MoEMLP_tp, MoERouter
from galvatron.models.moe.meta_configs import config_from_meta, model_layer_configs, model_name, set_model_config

def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, MoEModelInfo)
    return hybrid_parallel_configs


def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    wrap_block_name = [MoEAttention_tp, MoEMLP_tp]
    wrap_checkpoint_block_name = [MoEAttention_tp, MoEMLP_tp]
    wrap_other_block_name = [MoEEmbeddings_, MoEPreNorm_, MoECls_]
    all_block_name = [MoEEmbeddings_, MoEAttention_tp, MoEMLP_tp, MoERouter, MoEPreNorm_, MoECls_]
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        MoEModelInfo,
        construct_sequential_model,
        construct_tensor_parallel_model,
        wrap_block_name=wrap_block_name,
        wrap_checkpoint_block_name=wrap_checkpoint_block_name,
        wrap_other_block_name=wrap_other_block_name,
        # tied_wte_attr_names=['embed_tokens', 'lm_head'],
        layernorm_name=["LayerNorm", "norm", "MLPLayerNorm"],
        all_block_name=all_block_name,
        load_module_func=load_moe_module,
        meta_init_buffer = False,
    )
    return hp_model


def get_moe_config(args, overwrite_args=True):
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args)
    if hasattr(args, "local_rank") and args.local_rank == 0:
        print(config)
    return config


def moe_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")

    if args.initialize_on_meta:
        with init_empty_weights():
            moe_model = MixtralForCausalLM(config)
    else:
        moe_model = MixtralForCausalLM(config)

    model = construct_hybrid_parallel_model(
        model=moe_model, model_config=config, training_args=args, hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model


def get_runtime_profiler(args, path, config, start_iter=10, end_iter=20):
    profiler = RuntimeProfiler(args)
    profiler.set_profiler_dist(
        path, model_layer_configs(config), model_name(config), start_iter=start_iter, end_iter=end_iter
    )
    return profiler
