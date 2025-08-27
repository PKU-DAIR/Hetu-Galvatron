import json
import os

from transformers import MixtralConfig

from galvatron.utils import dict_join_dirname

# ============= Meta AI Model Config Paths =============
path_dict = {
    # "mixtral-8x7b": "mixtral-8x7b.json",
    # "mixtral-8x7b-e64k8": "mixtral-8x7b-e64k8.json",
    # "qwen-30BA3B": "qwen-30BA3B.json",
    # "dbrx": "dbrx.json",
    # "mixtral-8x7b-e16k4": "mixtral-8x7b-e16k4.json",
    # "mixtral-8x7b-e32k8": "mixtral-8x7b-e32k8.json",
    "qwen-8x7b-e8k2": "qwen-8x7b-e8k2.json",
    "qwen-8x7b-e16k4": "qwen-8x7b-e16k4.json",
    "mixtral-8x7b-e8k2": "mixtral-8x7b-e8k2.json",
    "mixtral-8x7b-e16k4": "mixtral-8x7b-e16k4.json",
    "mixtral-8x22b-e8k2": "mixtral-8x22b-e8k2.json",
    "mixtral-8x22b-e16k4": "mixtral-8x22b-e16k4.json",
}


def config_from_meta(model_type) -> MixtralConfig:
    if isinstance(model_type, str):
        global path_dict
        path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
        with open(path_dict[model_type]) as f:
            params = json.load(f)
    else:
        assert isinstance(model_type, dict), "model_type must be a string or a dictionary"
        params = model_type

    config = MixtralConfig(
        hidden_size=params["hidden_size"],
        intermediate_size=params["intermediate_size"],
        head_dim=params["head_dim"] if "head_dim" in params else None,
        max_position_embeddings=params["max_position_embeddings"],
        num_attention_heads=params["num_attention_heads"],
        num_experts_per_tok=params["num_experts_per_tok"],
        num_hidden_layers=params["num_hidden_layers"],
        num_key_value_heads=params["num_key_value_heads"],
        num_local_experts=params["num_local_experts"],
        vocab_size=params["vocab_size"],
        rms_norm_eps=params["rms_norm_eps"],
        rope_theta=params["rope_theta"],
        router_aux_loss_coef=params["router_aux_loss_coef"],
    )
    return config


# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    config.use_cache = False
    config.model_name = args.model_size
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.hidden_size = args.hidden_size
        config.intermediate_size = args.intermediate_size
        config.head_dim = args.head_dim
        config.max_position_embeddings = args.seq_length
        config.num_attention_heads = args.num_attention_heads
        config.num_experts_per_tok = args.num_experts_per_tok
        config.num_hidden_layers = args.num_hidden_layers
        config.num_key_value_heads = args.num_key_value_heads
        config.num_local_experts = args.num_experts
        config.vocab_size = args.vocab_size
        config.rms_norm_eps = args.norm_epsilon
        config.rope_theta = args.rope_theta
        config.router_aux_loss_coef = args.router_aux_loss_coef
    # Overwrite layer number only
    else:
        if args.set_layernum_manually:
            config.num_hidden_layers = args.num_hidden_layers
        if args.set_seqlen_manually:
            config.max_position_embeddings = args.seq_length
        if args.set_experts_manually:
            config.num_local_experts = args.num_experts
            config.num_experts_per_topk = args.num_experts_per_topk
        if args.set_aux_loss_manually:
            config.router_aux_loss_coef = args.router_aux_loss_coef

    # ======= Model Config --> Arguments ======
    overwrite_model_args(config, args)
    # This step is necessary that maintains the consistency of model config and arguments.
    if overwrite_args:  # Overwrite necessary Megatron-LM arguments with the model config
        overwrite_megatron_args(config, args)
    return config


def overwrite_model_args(config, args):
    args.is_moe_model = True
    args.hidden_size = config.hidden_size
    args.intermediate_size = config.intermediate_size
    args.head_dim = config.head_dim
    args.seq_length = config.max_position_embeddings
    args.num_attention_heads = config.num_attention_heads
    args.num_experts_per_tok = config.num_experts_per_tok
    args.num_hidden_layers = config.num_hidden_layers
    args.num_key_value_heads = config.num_key_value_heads
    args.num_local_experts = config.num_local_experts
    args.vocab_size = config.vocab_size
    args.rms_norm_eps = config.rms_norm_eps
    args.rope_theta = config.rope_theta
    args.router_aux_loss_coef = config.router_aux_loss_coef

    # TODO: Add grad symc for layernorm, but not need to do when tp = 1
    if args.model_size.startswith("qwen-30"):
        args.qk_layernorm = True
    else:
        args.qk_layernorm = False

def overwrite_megatron_args(config, args):
    args.num_layers = config.num_hidden_layers
    args.hidden_size = config.hidden_size
    args.ffn_hidden_size = config.intermediate_size
    args.seq_length = config.max_position_embeddings
    args.vocab_size = config.vocab_size
    args.num_attention_heads = config.num_attention_heads
    if config.head_dim is not None:
        args.kv_channels = config.head_dim
    else:
        args.kv_channels = args.hidden_size // args.num_attention_heads
    # print(config.head_dim, args.kv_channels)
    args.norm_epsilon = config.rms_norm_eps
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.add_bias_linear = False
    if args.model_size.startswith("qwen-8x"):
        args.add_qkv_bias = True
    else:
        args.add_qkv_bias = False
    args.swiglu = True
    args.position_embedding_type = "rope"
    args.apply_rope_fusion = True
    args.rotary_base = config.rope_theta
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = config.vocab_size
        # args.padded_vocab_size = (config.vocab_size + args.make_vocab_size_divisible_by - 1 // args.make_vocab_size_divisible_by * args.make_vocab_size_divisible_by)
    if config.num_key_value_heads != config.num_attention_heads:
        args.group_query_attention = True
        args.num_query_groups = config.num_key_value_heads
    
    args.num_experts = config.num_local_experts
    args.moe_ffn_hidden_size = config.intermediate_size
    if args.router_aux_loss_coef is None or args.router_aux_loss_coef == 0:
        args.moe_router_load_balancing_type = "none"
    else:
        args.moe_router_load_balancing_type = "aux_loss"
        args.moe_aux_loss_coeff = args.router_aux_loss_coef

    args.moe_router_topk = config.num_experts_per_tok
    args.moe_token_dispatcher_type = "alltoall" # "flex" for deepep, use with moe_enable_deepep
    # TODO: args need to consider
    # moe_grouped_gemm, moe_use_legacy_grouped_gemm
    # moe_router_dtype
    # moe_permute_fusion (effiecient permute)

# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    if hasattr(args, "profile_mode"):
        if args.profile_mode != "sequence":
            return "%s_seqlen%d" % (config.model_name, config.max_position_embeddings)
    return "%s" % (config.model_name)


def model_layer_configs(config):
    return [
        {
            "hidden_size": config.hidden_size,
            "seq_len": config.max_position_embeddings,
            "layer_num": config.num_hidden_layers,
        }
    ]
