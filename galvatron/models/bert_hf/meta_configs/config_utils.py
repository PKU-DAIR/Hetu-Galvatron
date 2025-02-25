import os, json
from transformers import BertConfig
from galvatron.utils import dict_join_dirname

# ============= HuggingFace Model Config Paths =============
path_dict = {
    'bert-base': 'bert-base-uncased.json',
    'bert-large': 'bert-large-uncased.json',
    'bert-huge-32': 'bert-huge-uncased-32.json',
    'bert-huge-48': 'bert-huge-uncased-48.json',
}

def config_from_meta(model_type) -> BertConfig:
    if isinstance(model_type, str):
        global path_dict
        path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
        with open(path_dict[model_type]) as f:
            params = json.load(f)
    else:
        assert isinstance(model_type, dict), "model_type must be a string or a dictionary"
        params = model_type
    return BertConfig(**params)

# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    config.use_cache = False
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.vocab_size = args.vocab_size
        config.hidden_size = args.hidden_size
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_attention_heads
        config.max_position_embeddings = args.seq_length
        config.hidden_dropout_prob = args.hidden_dropout
        config.attention_probs_dropout_prob = args.attention_dropout
    # Overwrite layer number only    
    else:
        if args.set_layernum_manually:
            config.num_hidden_layers = args.num_hidden_layers
        if args.set_seqlen_manually:
            config.max_position_embeddings = args.seq_length
    
    # ======= Model Config --> Arguments ======
    overwrite_model_args(config, args)
    # This step is necessary that maintains the consistency of model config and arguments.
    if overwrite_args: # Overwrite necessary Megatron-LM arguments with the model config
        overwrite_megatron_args(config, args)
    return config

def overwrite_model_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_hidden_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.vocab_size = config.vocab_size
    args.seq_length = config.max_position_embeddings
    args.hidden_dropout = config.hidden_dropout_prob
    args.attention_dropout = config.attention_probs_dropout_prob

def overwrite_megatron_args(config, args):
    args.num_layers = config.num_hidden_layers
    args.hidden_size = config.hidden_size
    args.ffn_hidden_size = config.intermediate_size
    args.seq_length = config.max_position_embeddings
    args.num_attention_heads = config.num_attention_heads
    args.kv_channels = args.hidden_size // args.num_attention_heads
    
    args.hidden_dropout = config.hidden_dropout_prob
    args.attention_dropout = config.attention_probs_dropout_prob
    
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = (
            config.vocab_size + args.make_vocab_size_divisible_by - 1
        ) // args.make_vocab_size_divisible_by * args.make_vocab_size_divisible_by
    
    args.max_position_embeddings = config.max_position_embeddings
    args.tokenizer_type = 'BertWordPieceLowerCase'  
    args.layernorm_epsilon = config.layer_norm_eps  

# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    if hasattr(args, "profile_mode"):
        if args.profile_mode != "sequence":
            return 'hidden%d_head%d_seqlen%d' % (
                config.hidden_size,
                config.num_attention_heads,
                config.max_position_embeddings
            )
    return 'hidden%d_head%d' % (
        config.hidden_size,
        config.num_attention_heads
    )

def model_layer_configs(config):
    return [
        {
            'hidden_size': config.hidden_size,
            'seq_len': config.max_position_embeddings,
            'layer_num': config.num_hidden_layers,
        }
    ]