from torch.nn import LayerNorm
from transformers import BertForPreTraining
from galvatron.core import construct_hybrid_parallel_model_api, get_hybrid_parallel_configs_api
from galvatron.models.bert_hf.BertModel_sequential import BertModelInfo, construct_sequential_model
from galvatron.models.bert_hf.BertModel_sequential import BertVocabEmbedding_, BertPositionEmbedding_, BertTokenTypeEmbedding_
from galvatron.models.bert_hf.BertModel_sequential import BertCls_
from galvatron.models.bert_hf.BertModel_tensor_parallel import construct_tensor_parallel_model, BertLayer_tp
from galvatron.models.bert_hf.BertModel_checkpoint import load_bert_module
from galvatron.core import get_args

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
        BertVocabEmbedding_,
        BertPositionEmbedding_,
        BertTokenTypeEmbedding_,
        BertCls_
    ]
    wrap_checkpoint_block_name = [BertLayer_tp]
    
    all_block_name = [
        BertVocabEmbedding_,
        BertPositionEmbedding_,
        BertTokenTypeEmbedding_,
        BertLayer_tp,
        BertCls_
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
        tied_wte_attr_names=['word_embeddings', 'decoder'] if not args.untie_embeddings_and_output_weights else None,
        layernorm_name=['LayerNorm'],
        all_block_name=all_block_name,
        load_module_func=load_bert_module,
    )
    return hp_model

def bert_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(
        model_config=config, 
        training_args=args
    )
    
    if args.local_rank == 0:
        print("Creating Model...")

    bert_model = BertForPreTraining(config)
    
    model = construct_hybrid_parallel_model(
        model=bert_model,
        model_config=config,
        training_args=args,
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model