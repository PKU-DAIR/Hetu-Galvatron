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
    def __init__(self, config, layer_number, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.sp_size = sp_group.size
        self.cp_size = cp_group.size
        self.use_ulysses = self.sp_size > 1   
        self.use_zigzag_cp = self.cp_size > 1
        megatron_config = core_transformer_config_from_args(args)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.cp_group = cp_group.group if cp_group is not None else None
        self.cp_ranks = cp_group.ranks if cp_group is not None else None
        self.attention = ParallelAttention(
            megatron_config,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=AttnMaskType.causal,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
            cp_group=self.cp_group,
            cp_ranks=self.cp_ranks,
            use_ulysses=self.use_ulysses,
            use_zigzag_cp=self.use_zigzag_cp,
        )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_number
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_pos_emb = RotaryEmbedding(
            self.head_dim, args.rotary_percent, seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor, 
            cp_group=self.cp_group, sp_group=self.sp_group
        )

    def forward(self, hidden_states, attention_mask):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        if self.sequence_parallel:
            if self.use_ulysses:
                if self.use_zigzag_cp:
                    #max_seq_len = hidden_states.shape[0] * self.cp_size * self.sp_size
                    #no offset for zigzag cp, because the offset is already included in the Megatron RotaryEmbedding
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] * self.cp_size * self.sp_size)
                else:
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] , offset=hidden_states.shape[0] * torch.distributed.get_rank(self.sp_group))
            else:
                if self.use_zigzag_cp:
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] * torch.distributed.get_world_size(self.tp_group) * self.cp_size)
                else:
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] * torch.distributed.get_world_size(self.tp_group)
                    )
        else:
            if self.use_zigzag_cp:
                rotary_pos_emb = self.rotary_pos_emb(hidden_states.shape[0] * self.cp_size)
            else:
                rotary_pos_emb = self.rotary_pos_emb(hidden_states.shape[0])
        hidden_states, bias = self.attention(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class LlamaMLP_tp(nn.Module):
    def __init__(self, config, tp_group=None):
        super().__init__()
        megatron_config = core_transformer_config_from_args(get_args())
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(megatron_config, tp_group=self.tp_group)
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.mlp(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class LlamaLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        self.attention = LlamaAttention_tp(config, layer_number, tp_group, sp_group, cp_group)
        self.mlp = LlamaMLP_tp(config, tp_group)
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


def construct_tensor_parallel_model(model, config, tp_groups_whole, sp_groups_whole, cp_groups_whole):
    layers_tp = nn.ModuleList(
        [
            LlamaLayer_tp(config, i, tp_group=tp_groups_whole[i + 1], sp_group=sp_groups_whole[i + 1], cp_group=cp_groups_whole[i + 1])
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
            tp_group=tp_groups_whole[0].group,
            sp_group=sp_groups_whole[0].group,
            cp_group=cp_groups_whole[0].group
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
            tp_group=tp_groups_whole[-1].group,
            sp_group=sp_groups_whole[-1].group,
            cp_group=cp_groups_whole[-1].group,
        ),
    )

    return model
