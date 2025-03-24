import os, json
from transformers import ViTConfig
from galvatron.utils import dict_join_dirname

# ============= HuggingFace Model Config Paths =============
path_dict = {
    'vit-base': 'vit-base-patch16-224.json',
    'vit-large': 'vit-large-patch16-224.json',
    'vit-huge': 'vit-huge-patch16-224.json',
    'vit-xhuge': 'vit-xhuge-patch16-224.json',
}

def config_from_meta(model_type) -> ViTConfig:
    if isinstance(model_type, str):
        global path_dict
        path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
        with open(path_dict[model_type]) as f:
            params = json.load(f)
    else:
        assert isinstance(model_type, dict), "model_type must be a string or a dictionary"
        params = model_type
    return ViTConfig(**params)

# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    config.use_cache = False
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.hidden_size = args.hidden_size
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_attention_heads
        config.image_size = args.image_size
        config.patch_size = args.patch_size
        config.num_channels = args.num_channels
        config.num_labels = args.num_labels
        config.intermediate_size = args.intermediate_size
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        config.layer_norm_eps = args.layer_norm_eps
    # Overwrite layer number only    
    else:
        if args.set_layernum_manually:
            config.num_hidden_layers = args.num_hidden_layers
        if args.set_seqlen_manually:
            config.image_size = args.image_size
            config.patch_size = args.patch_size
            config.num_channels = args.num_channels
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
    args.image_size = config.image_size
    args.patch_size = config.patch_size
    args.num_channels = config.num_channels
    args.num_labels = config.num_labels
    args.intermediate_size = config.intermediate_size
    args.hidden_dropout_prob = config.hidden_dropout_prob
    args.attention_probs_dropout_prob = config.attention_probs_dropout_prob
    args.layer_norm_eps = config.layer_norm_eps

def overwrite_megatron_args(config, args):
    args.num_layers = config.num_hidden_layers
    args.hidden_size = config.hidden_size
    args.ffn_hidden_size = config.intermediate_size
    args.seq_length = (config.image_size // config.patch_size) ** 2 + 1  # +1 for CLS token
    args.num_attention_heads = config.num_attention_heads
    args.kv_channels = args.hidden_size // args.num_attention_heads
    args.norm_epsilon = config.layer_norm_eps
    args.hidden_dropout = config.hidden_dropout_prob
    args.attention_dropout = config.attention_probs_dropout_prob

# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    if hasattr(args, "profile_mode"):
        if args.profile_mode != "sequence":
            return 'hidden%d_head%d_imgsize%d_patch%d' % (
                config.hidden_size,
                config.num_attention_heads,
                config.image_size,
                config.patch_size
            )
    return 'hidden%d_head%d' % (
        config.hidden_size,
        config.num_attention_heads
    )

def model_layer_configs(config):
    seq_len = (config.image_size // config.patch_size) ** 2 + 1  # +1 for CLS token
    return [
        {
            'hidden_size': config.hidden_size,
            'seq_len': seq_len,
            'layer_num': config.num_hidden_layers,
        }
    ]