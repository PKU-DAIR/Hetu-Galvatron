from torch.nn import LayerNorm
from transformers import ViTForImageClassification
from galvatron.core import construct_hybrid_parallel_model_api, get_hybrid_parallel_configs_api, RuntimeProfiler
from galvatron.models.vit_hf.ViTModel_sequential import (
    ViTModelInfo, 
    construct_sequential_model,
    ViTPatchEmbeddings_,
    ViTPositionEmbeddings_,
    ViTEmbeddings_,
    ViTLayers_,
    ViTCls_,
    ViTPreNorm_
)
from galvatron.models.vit_hf.ViTModel_tensor_parallel import (
    construct_tensor_parallel_model,
    ViTLayer_tp
)
from galvatron.models.vit_hf.ViTModel_checkpoint import load_vit_module
from galvatron.core import get_args
from galvatron.core.runtime.initialize import init_empty_weights
from galvatron.models.vit_hf.meta_configs import config_from_meta, set_model_config, model_layer_configs, model_name

def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(
        model_config, 
        training_args, 
        ViTModelInfo
    )
    
    return hybrid_parallel_configs

def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    args = get_args()
    wrap_block_name = [ViTLayer_tp]
    wrap_other_block_name = [
        ViTEmbeddings_,
        ViTPatchEmbeddings_,
        ViTPositionEmbeddings_,
        ViTPreNorm_,
        ViTCls_
    ]
    wrap_checkpoint_block_name = [ViTLayer_tp]
    all_block_name = [
        ViTEmbeddings_,
        ViTPatchEmbeddings_,
        ViTPositionEmbeddings_,
        ViTLayer_tp,
        ViTPreNorm_,
        ViTCls_
    ]
    
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        ViTModelInfo,
        construct_sequential_model,
        construct_tensor_parallel_model,
        wrap_block_name=wrap_block_name,
        wrap_checkpoint_block_name=wrap_checkpoint_block_name,
        wrap_other_block_name=wrap_other_block_name,
        tied_wte_attr_names=None,
        layernorm_name=['LayerNorm'],
        all_block_name=all_block_name,
        load_module_func=load_vit_module
    )
    return hp_model

def get_vit_config(args, overwrite_args=True):
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args)
    if hasattr(args, 'local_rank') and args.local_rank == 0:
        print(config)
    return config

def vit_model_hp(config, args):
    # 确保序列并行被禁用
    args.sequence_parallel = False
    args.use_ulysses = False
    
    hybrid_parallel_configs = get_hybrid_parallel_configs(
        model_config=config,
        training_args=args
    )
    
    if args.local_rank == 0:
        print("Creating Model...")
    
    # 定义ViTPooler类
    import torch.nn as nn
    
    class ViTPooler(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()
        
        def forward(self, hidden_states):
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token.
            first_token_tensor = hidden_states[:, 0]
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)
            return pooled_output
    
    # 创建模型
    if args.initialize_on_meta:
        with init_empty_weights():
            vit_model = ViTForImageClassification(config)
            if not hasattr(vit_model.vit, 'pooler') or vit_model.vit.pooler is None:
                vit_model.vit.pooler = ViTPooler(config)
    else:
        vit_model = ViTForImageClassification(config)
        if not hasattr(vit_model.vit, 'pooler') or vit_model.vit.pooler is None:
            vit_model.vit.pooler = ViTPooler(config)
    
    model = construct_hybrid_parallel_model(
        model=vit_model,
        model_config=config,
        training_args=args,
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model

def get_runtime_profiler(args, path, config, start_iter=10, end_iter=20):
    profiler = RuntimeProfiler(args)
    profiler.set_profiler_dist(
        path, 
        model_layer_configs(config), 
        model_name(config), 
        start_iter=start_iter, 
        end_iter=end_iter
    )
    return profiler