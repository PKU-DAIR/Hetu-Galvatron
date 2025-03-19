from torch.nn import LayerNorm
from transformers import BertForMaskedLM
from galvatron.core import construct_hybrid_parallel_model_api, get_hybrid_parallel_configs_api
from galvatron.models.bert_hf.BertModel_sequential import BertModelInfo, construct_sequential_model, BertEmbeddings_,BertMLMCls_
from galvatron.models.bert_hf.BertModel_tensor_parallel import construct_tensor_parallel_model, BertLayer_tp
from galvatron.models.bert_hf.BertModel_checkpoint import load_bert_module
from galvatron.core import get_args
from galvatron.core.initialize import init_empty_weights
from galvatron.models.bert_hf.meta_configs import config_from_meta, set_model_config

def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(
        model_config, 
        training_args, 
        BertModelInfo
    )
    return hybrid_parallel_configs

def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    args = get_args()
    wrap_block_name = [BertLayer_tp]  
    wrap_other_block_name = [
        BertEmbeddings_,
        BertMLMCls_
    ]
    wrap_checkpoint_block_name = [BertLayer_tp]
    all_block_name = [
        BertEmbeddings_,
        BertLayer_tp,
        BertMLMCls_
    ]
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        BertModelInfo,
        construct_sequential_model,      
        construct_tensor_parallel_model, 
        wrap_block_name=wrap_block_name,
        wrap_checkpoint_block_name=wrap_checkpoint_block_name,
        wrap_other_block_name=wrap_other_block_name,
        tied_wte_attr_names=None,   
        layernorm_name=['LayerNorm'],    
        all_block_name=all_block_name,
        load_module_func=load_bert_module
    )
    return hp_model

def get_bert_config(args, overwrite_args=True):
    config = config_from_meta(args.model_size) 
    config = set_model_config(config, args, overwrite_args)
    if hasattr(args, 'local_rank') and args.local_rank == 0:
        print(config)
    return config
    
def bert_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(
        model_config=config, 
        training_args=args
    )
    
    if args.local_rank == 0:
        print("Creating Model...")

    if args.initialize_on_meta:
        with init_empty_weights():
            bert_model = BertForMaskedLM(config)
    else:
        bert_model = BertForMaskedLM(config)
        
    model = construct_hybrid_parallel_model(
        model=bert_model,
        model_config=config,
        training_args=args,
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model