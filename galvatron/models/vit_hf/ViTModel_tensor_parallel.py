import torch
from torch import nn
from torch.nn import LayerNorm
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from megatron.training.arguments import core_transformer_config_from_args

from galvatron.core import get_args
from galvatron.core.runtime.tensor_parallel import AttnMaskType, AttnType, ParallelAttention, ParallelMLP


class ViTAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        super().__init__()
        args = get_args()
        megatron_config = core_transformer_config_from_args(args)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.use_ulysses = args.use_ulysses
        
        self.hidden_dropout_prob = config.hidden_dropout_prob

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.attention = ParallelAttention(
            megatron_config,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=AttnMaskType.padding,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
            use_ulysses=self.use_ulysses    
        )

    def forward(self, hidden_states, attention_mask=None):
        seq_len, bsz, hs = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.zeros((bsz, 1, seq_len, seq_len), dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.repeat(bsz, 1, 1, 1).to(hidden_states.device)

        residual = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.attention(hidden_states, attention_mask)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.hidden_dropout_prob)
        hidden_states = hidden_states + residual
        return hidden_states

class ViTMLP_tp(nn.Module):
    def __init__(self, config, tp_group=None):
        super().__init__()
        args = get_args()
        megatron_config = core_transformer_config_from_args(args)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ParallelMLP(megatron_config, tp_group=self.tp_group)
        self.hidden_dropout_prob = config.hidden_dropout_prob

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.mlp(hidden_states)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.hidden_dropout_prob)
        hidden_states = hidden_states + residual
        return hidden_states

class ViTLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        super().__init__()
        self.attention = ViTAttention_tp(config, layer_number, tp_group, sp_group)
        self.mlp = ViTMLP_tp(config, tp_group)
        self.idx = layer_number

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        layer_output = self.mlp(attention_output)
        return layer_output


def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc):
    args = get_args()
    megatron_config = core_transformer_config_from_args(args)
    
    setattr(
        model.vit.embeddings,
        "patch_embeddings",
        ColumnParallelLinear(
            config.patch_size * config.patch_size * config.num_channels,
            config.hidden_size,
            config=megatron_config,
            gather_output=True,
            bias=True,
            init_method=megatron_config.init_method,
            tp_group=tp_groups_enc[0].group,
            sp_group=sp_groups_enc[0].group,
        ),
    )
    
    model.vit.embeddings.tp_group = tp_groups_enc[0].group
    model.vit.embeddings.sp_group = sp_groups_enc[0].group
    
    layers_tp = nn.ModuleList([
        ViTLayer_tp(
            config,
            i,
            tp_group=tp_groups_enc[i + 1],
            sp_group=sp_groups_enc[i + 1]
        ) for i in range(config.num_hidden_layers)
    ])
    setattr(model.vit.encoder, 'layer', layers_tp)
    
    model.classifier = ColumnParallelLinear(
        config.hidden_size,
        config.num_labels,
        bias=True,
        config=megatron_config,
        init_method=megatron_config.init_method,
        tp_group=tp_groups_enc[-1].group,
        sp_group=sp_groups_enc[-1].group
    )
    
    return model
