import torch
from flash_attn.ops.rms_norm import RMSNorm as LlamaRMSNorm

# from transformers.models.llama.modeling_llama import LlamaRMSNorm
# from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.tensor_parallel import ColumnParallelLinear, VocabParallelEmbedding
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from galvatron.core import get_args
from galvatron.core.runtime.tensor_parallel import AttnMaskType, AttnType, ParallelAttention, ParallelMLP


class LlamaAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        super().__init__()
        args = get_args()
        self.idx = layer_number
        self.category = "attention"
        self.sequence_parallel = args.sequence_parallel
        self.use_ulysses = sp_group.size > 1
        megatron_config = core_transformer_config_from_args(args)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.attention = ParallelAttention(
            megatron_config,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=AttnMaskType.causal,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
            use_ulysses=self.use_ulysses,
        )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_pos_emb = RotaryEmbedding(
            self.head_dim, args.rotary_percent, seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )

    def forward(self, hidden_states, attention_mask):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        if self.sequence_parallel:
            if self.use_ulysses:
                rotary_pos_emb = self.rotary_pos_emb(
                    hidden_states.shape[0], offset=hidden_states.shape[0] * torch.distributed.get_rank(self.sp_group)
                )
            else:
                rotary_pos_emb = self.rotary_pos_emb(
                    hidden_states.shape[0] * torch.distributed.get_world_size(self.tp_group)
                )
        else:
            rotary_pos_emb = self.rotary_pos_emb(hidden_states.shape[0])
        hidden_states, bias = self.attention(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class LlamaMLP_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None):
        super().__init__()
        megatron_config = core_transformer_config_from_args(get_args())
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(megatron_config, tp_group=self.tp_group)
        self.idx = layer_number
        self.category = "mlp"
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.mlp(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class LlamaLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        super().__init__()
        self.attention = LlamaAttention_tp(config, layer_number, tp_group, sp_group)
        self.mlp = LlamaMLP_tp(config, layer_number, tp_group)
        self.idx = layer_number

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
        )
        layer_output = self.mlp(attention_output)
        # outputs = (layer_output
        return layer_output


def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc):
    layers_tp = nn.ModuleList(
        [
            LlamaLayer_tp(config, i, tp_group=tp_groups_enc[i + 1], sp_group=sp_groups_enc[i + 1])
            for i in range(config.num_hidden_layers)
        ]
    )
    setattr(model.model, "layers", layers_tp)
    args = get_args()
    megatron_config = core_transformer_config_from_args(get_args())
    setattr(
        model.model,
        "embed_tokens",
        VocabParallelEmbedding(
            args.padded_vocab_size,
            megatron_config.hidden_size,
            config=megatron_config,
            init_method=megatron_config.init_method,
            tp_group=tp_groups_enc[0].group,
            sp_group=sp_groups_enc[0].group,
        ),
    )
    setattr(
        model,
        "lm_head",
        ColumnParallelLinear(
            megatron_config.hidden_size,
            args.padded_vocab_size,
            config=megatron_config,
            init_method=megatron_config.init_method,
            bias=False,
            tp_group=tp_groups_enc[-1].group,
            sp_group=sp_groups_enc[-1].group,
        ),
    )

    return model
