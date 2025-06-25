import torch
from flash_attn.ops.rms_norm import RMSNorm
from megatron.core import mpu
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from megatron.core.transformer.enums import AttnMaskType, AttnType
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from galvatron.core import get_args
from galvatron.core.runtime.tensor_parallel.mlp import MLPSubmodules
from galvatron.core.runtime.tensor_parallel.attention import SelfAttention, SelfAttentionSubmodules
from galvatron.core.runtime.tensor_parallel.attention_impl import DotProductAttention, FlashSelfOrCrossAttention, DistributedAttention
from galvatron.core.runtime.moe.mlp import GroupedMLP, SequentialMLP
from galvatron.core.runtime.moe.router import TopKRouter
from galvatron.core.runtime.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)


class MoEAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        super().__init__()
        args = get_args()
        self.idx = layer_number
        self.sequence_parallel = args.sequence_parallel
        self.use_ulysses = sp_group.size > 1
        megatron_config = core_transformer_config_from_args(args)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.attention = SelfAttention(
            megatron_config,
            SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                flash_attention=FlashSelfOrCrossAttention,
                dist_attention=DistributedAttention,
                linear_proj=RowParallelLinear,
            ),
            layer_number,
            attn_mask_type=AttnMaskType.causal,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
        )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.LayerNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_pos_emb = RotaryEmbedding(
            self.head_dim, args.rotary_percent, 
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            rotary_base=args.rotary_base
        )

        self.MLPLayerNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, rotary_embedding):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        if self.sequence_parallel:
            if self.use_ulysses:
                rotary_pos_emb = self.rotary_pos_emb(
                    hidden_states.shape[0], offset=hidden_states.shape[0] * torch.distributed.get_rank(self.sp_group)
                )
            else:
                if rotary_embedding is not None:
                    rotary_pos_emb = rotary_embedding
                else:
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] * torch.distributed.get_world_size(self.tp_group)
                    )
        else:
            if rotary_embedding is not None:
                rotary_pos_emb = rotary_embedding
            else:
                rotary_pos_emb = self.rotary_pos_emb(hidden_states.shape[0])
        hidden_states, bias = self.attention(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + input_tensor

        mlp_residual = hidden_states
        hidden_states = self.MLPLayerNorm(hidden_states)
        return hidden_states, mlp_residual

class MoERouter(nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        args = get_args()
        megatron_config = core_transformer_config_from_args(args)
        self.idx = layer_number
        self.router = TopKRouter(config=megatron_config)
        self.router.layer_number = layer_number

    def forward(self, hidden_states):
        if hasattr(self, "predict"):
            self.predict(hidden_states)
        probs, routing_map = self.router(hidden_states)
        return probs, routing_map
    
# TODO: Add shared expert support
class MoEMLP_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, ep_group=None, tp_of_ep_group=None, tp_and_ep_group=None):
        super().__init__()
        args = get_args()
        self.use_fsep = args.use_fsep
        self.is_moe_layer = self.use_fsep
        self.idx = layer_number
        megatron_config = core_transformer_config_from_args(args)
        self.config = megatron_config
        self.tp_group = tp_group.group if tp_group is not None else None
        self.ep_group = ep_group.group if ep_group is not None else None
        self.tp_of_ep_group = tp_of_ep_group.group if tp_of_ep_group is not None else None
        self.tp_and_ep_group = tp_and_ep_group.group if tp_and_ep_group is not None else None

        self.expert_parallel_size = mpu.get_expert_model_parallel_world_size(self.ep_group)
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_global_experts = self.config.num_moe_experts
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            mpu.get_expert_model_parallel_rank(self.ep_group) * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))

        if self.config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config, ep_group=self.ep_group, tp_of_ep_group=self.tp_of_ep_group, tp_and_ep_group=self.tp_and_ep_group
            )
        elif self.config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config, ep_group=self.ep_group, tp_of_ep_group=self.tp_of_ep_group, tp_and_ep_group=self.tp_and_ep_group,
                layer_number = self.idx
            )
        elif self.config.moe_token_dispatcher_type == "alltoall_seq":
            assert False, "alltoall_seq is deprecated"
        elif self.config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config, ep_group=self.ep_group, tp_of_ep_group=self.tp_of_ep_group, tp_and_ep_group=self.tp_and_ep_group
            )
        
        if args.moe_grouped_gemm:
            assert self.use_fsep is False, "Grouped gemm does not support fsep."
            self.experts = GroupedMLP(self.num_local_experts, self.config, self.tp_of_ep_group)
        else:
            # TODO: TE Tensor Parallel Adaptation
            if self.use_fsep:
                self.experts = SequentialMLP(
                    self.num_global_experts,
                    self.config,
                    MLPSubmodules(
                        linear_fc1=ColumnParallelLinear,
                        linear_fc2=RowParallelLinear,
                    ),
                    self.tp_of_ep_group,
                    self.tp_and_ep_group,
                )
                self.real_experts = SequentialMLP(
                    self.num_local_experts,
                    self.config,
                    MLPSubmodules(
                        linear_fc1=ColumnParallelLinear,
                        linear_fc2=RowParallelLinear,
                    ),
                    self.tp_of_ep_group,
                    self.tp_and_ep_group,
                )
                self.global_expert_indices = torch.tensor(range(self.num_global_experts),dtype=torch.int32,device=torch.cuda.current_device())
                self.global_expert_indices = self.global_expert_indices.repeat(self.expert_parallel_size * self.num_local_experts // self.num_global_experts).reshape(-1, self.expert_parallel_size).transpose(0,1).contiguous()
            else:
                self.experts = SequentialMLP(
                    self.num_local_experts,
                    self.config,
                    MLPSubmodules(
                        linear_fc1=ColumnParallelLinear,
                        linear_fc2=RowParallelLinear,
                    ),
                    self.tp_of_ep_group,
                    self.tp_and_ep_group,
                )

    def forward(self, hidden_states, mlp_residual, probs, routing_map):
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
        if self.use_fsep:
            expert_output, mlp_bias = self.real_experts(dispatched_input, tokens_per_expert)
        else:
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        hidden_states, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        hidden_states = hidden_states + mlp_residual
        return hidden_states


class MoELayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None, ep_group=None, tp_of_ep_group=None, tp_and_ep_group=None):
        super().__init__()
        self.attention = MoEAttention_tp(config, layer_number, tp_group, sp_group)
        self.router = MoERouter(layer_number)
        self.mlp = MoEMLP_tp(config, layer_number, tp_group, ep_group, tp_of_ep_group, tp_and_ep_group)
        self.idx = layer_number

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rotary_embedding=None,
    ):
        attention_output, mlp_residual = self.attention(
            hidden_states,
            attention_mask,
            rotary_embedding,
        )
        probs, routing_map = self.router(attention_output)
        layer_output = self.mlp(attention_output, mlp_residual, probs, routing_map)
        return layer_output


def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc, ep_groups_enc, tp_of_ep_groups_enc, tp_and_ep_groups_enc):
    layers_tp = nn.ModuleList(
        [
            MoELayer_tp(config, i, 
                tp_group=tp_groups_enc[i + 1], 
                sp_group=sp_groups_enc[i + 1], 
                ep_group=ep_groups_enc[i + 1], 
                tp_of_ep_group=tp_of_ep_groups_enc[i + 1], 
                tp_and_ep_group=tp_and_ep_groups_enc[i + 1])
            for i in range(config.num_hidden_layers)
        ]
    )
    for i in range(len(layers_tp)-1):
        layers_tp[i].router.predict = layers_tp[i+1].router.router.predict
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
            reduce_scatter_embeddings=args.sequence_parallel,
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
