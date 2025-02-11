import torch
from torch import nn
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel import VocabParallelEmbedding, ColumnParallelLinear
from megatron.training.arguments import core_transformer_config_from_args
from galvatron.core import get_args
from galvatron.core.tensor_parallel import ParallelMLP, ParallelAttention
from galvatron.core.tensor_parallel import AttnMaskType, AttnType
from torch.nn import LayerNorm


class BertAttention_tp(nn.Module):
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
            attn_mask_type=AttnMaskType.padding,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
            use_ulysses=self.use_ulysses
        )
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.hidden_dropout = megatron_config.attention_dropout

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states, bias = self.attention(hidden_states, attention_mask)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.hidden_dropout)
        hidden_states = hidden_states + residual
        # Post-LayerNorm
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertMLP_tp(nn.Module):
    def __init__(self, config, tp_group=None):
        super().__init__()
        megatron_config = core_transformer_config_from_args(get_args())
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(megatron_config, tp_group=self.tp_group)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.hidden_dropout = megatron_config.hidden_dropout
        
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states, bias = self.mlp(hidden_states)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.hidden_dropout)
        hidden_states = hidden_states + residual
        # Post-LayerNorm
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        super().__init__()
        self.attention = BertAttention_tp(config, layer_number, tp_group, sp_group)
        self.mlp = BertMLP_tp(config, tp_group)
        self.idx = layer_number

    def forward(self, hidden_states, attention_mask=None):
        
        attention_output = self.attention(hidden_states, attention_mask)
        
        layer_output = self.mlp(attention_output)

        return layer_output

def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc):
    layers_tp = nn.ModuleList([
        BertLayer_tp(
            config, 
            i, 
            tp_group=tp_groups_enc[i + 1], 
            sp_group=sp_groups_enc[i + 1]
        ) for i in range(config.num_hidden_layers)
    ])
    
    setattr(model.bert.encoder, 'layer', layers_tp)
    
    args = get_args()
    megatron_config = core_transformer_config_from_args(get_args())

    setattr(model.bert.embeddings, 'word_embeddings', VocabParallelEmbedding(
        args.padded_vocab_size,
        megatron_config.hidden_size,
        config=megatron_config,
        init_method=megatron_config.init_method,
        tp_group=tp_groups_enc[0].group,
        sp_group=sp_groups_enc[0].group
    ))
    

    setattr(model.bert.embeddings, 'position_embeddings', VocabParallelEmbedding(
        config.max_position_embeddings,
        megatron_config.hidden_size,
        config=megatron_config,
        init_method=megatron_config.init_method,
        tp_group=tp_groups_enc[0].group,
        sp_group=sp_groups_enc[0].group
    ))

    setattr(model.cls.predictions, 'decoder', ColumnParallelLinear(
        config.hidden_size,
        args.padded_vocab_size,
        bias=True,  
        config=megatron_config,
        init_method=megatron_config.init_method,
        tp_group=tp_groups_enc[-1].group,
        sp_group=sp_groups_enc[-1].group
    ))

    tensor_parallel.set_tensor_model_parallel_attributes(
        model.cls.predictions.bias,
        True, 0, 1
    )
    #optional: token type embeddings
    # setattr(model.bert.embeddings, 'token_type_embeddings', VocabParallelEmbedding(
    #     config.type_vocab_size,
    #     megatron_config.hidden_size,
    #     config=megatron_config,
    #     init_method=megatron_config.init_method,
    #     tp_group=tp_groups_enc[0].group,
    #     sp_group=sp_groups_enc[0].group
    # ))
    
    return model
