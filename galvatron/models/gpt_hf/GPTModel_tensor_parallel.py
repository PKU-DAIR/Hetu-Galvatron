import torch
from megatron.core.tensor_parallel import ColumnParallelLinear, VocabParallelEmbedding
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn
from torch.nn import LayerNorm

from galvatron.core import get_args
from galvatron.core.runtime.tensor_parallel import AttnMaskType, AttnType, ParallelAttention, ParallelMLP

# from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm


class GPTAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        super().__init__()
        args = get_args()
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
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.hidden_dropout = megatron_config.attention_dropout

    def forward(self, hidden_states, attention_mask):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.attention(hidden_states, attention_mask)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.hidden_dropout)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class GPTMLP_tp(nn.Module):
    def __init__(self, config, tp_group=None):
        super().__init__()
        megatron_config = core_transformer_config_from_args(get_args())
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(megatron_config, tp_group=self.tp_group)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.hidden_dropout = megatron_config.hidden_dropout

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.mlp(hidden_states)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.hidden_dropout)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class GPTLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        super().__init__()
        self.attention = GPTAttention_tp(config, layer_number, tp_group, sp_group)
        self.mlp = GPTMLP_tp(config, tp_group)
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
        return layer_output


def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc):
    layers_tp = nn.ModuleList(
        [
            GPTLayer_tp(config, i, tp_group=tp_groups_enc[i + 1], sp_group=sp_groups_enc[i + 1])
            for i in range(config.num_hidden_layers)
        ]
    )
    setattr(model.transformer, "h", layers_tp)
    args = get_args()
    megatron_config = core_transformer_config_from_args(get_args())
    setattr(
        model.transformer,
        "wte",
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
        model.transformer,
        "wpe",
        VocabParallelEmbedding(
            args.seq_length,
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
