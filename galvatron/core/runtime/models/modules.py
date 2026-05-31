import torch
import torch.nn as nn

from galvatron.core.runtime import parallel_state
from galvatron.core.runtime.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from galvatron.core.runtime.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
)
from galvatron.core.runtime.tensor_parallel.utils import VocabUtility, divide
from galvatron.core.runtime.utils.utils import cp_zigzag_positions, sp_narrow_positions

from galvatron.core.runtime.transformer.attention import SelfAttention, SelfAttentionSubmodules, AttnMaskType
from galvatron.core.runtime.transformer.attention_impl import (
    FlashSelfOrCrossAttention,
    DistributedAttention,
    ZigzagRingFlashAttention,
)
from galvatron.core.runtime.transformer.mlp import MLP, MLPSubmodules

from galvatron.core.runtime.transformer.fused_kernels import fused_vocab_parallel_cross_entropy
from galvatron.core.runtime.transformer.rotary_pos_embedding import RotaryEmbedding
from galvatron.core.runtime.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce
from galvatron.core.runtime.transformer.norm import GalvatronNorm
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
from galvatron.core.runtime.transformer.attention import PackedSeqParams


# =========================================================================
# Embedding
# =========================================================================

class GalvatronEmbedding(nn.Module):
    """Token embedding (+ optional learned position embedding).

    Supports vocab-parallel embedding and sequence-parallel scatter.
    """

    def __init__(self, args: GalvatronRuntimeArgs, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        m = args.model
        self.sequence_parallel = args.train.sequence_parallel
        self.vocab_sp = args.parallel.vocab_sp

        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.cp_group = cp_group.group if cp_group is not None else None

        self.embed_tokens = VocabParallelEmbedding(
            m.padded_vocab_size,
            m.hidden_size,
            config=m,
            reduce_scatter_embeddings=self.sequence_parallel,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
            cp_group=self.cp_group,
        )

        self.has_position_embedding = m.position_embedding_type == "learned_absolute"
        if self.has_position_embedding:
            seq_len = args.train.seq_length
            self.embed_positions = nn.Embedding(seq_len, m.hidden_size)

        self.drop = nn.Dropout(m.hidden_dropout) if m.hidden_dropout > 0 else nn.Identity()

        self.sp_size = parallel_state.get_parallel_world_size(self.sp_group) if self.sp_group is not None else 1
        self.sp_rank = parallel_state.get_parallel_rank(self.sp_group) if self.sp_group is not None else 0
        self.cp_size = parallel_state.get_parallel_world_size(self.cp_group) if self.cp_group is not None else 1
        self.cp_rank = parallel_state.get_parallel_rank(self.cp_group) if self.cp_group is not None else 0

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None, cu_seqlens=None):
        if self.vocab_sp:
            input_ids_lens = input_ids.shape[1] # input_ids.shape is (b, seq_len)
            partition_size = divide(input_ids_lens, self.sp_size)
            start_index, end_index = self.sp_rank * partition_size, (self.sp_rank + 1) * partition_size
            input_ids = input_ids[:, start_index:end_index].contiguous()

        hidden_states = self.embed_tokens(input_ids)

        if self.has_position_embedding:
            if position_ids is None:
                # Layout differs only in (1) which dim is seq vs batch and (2) the final
                # reshape; the position-id computation itself is the same.
                sbh = self.embed_tokens.reduce_scatter_embeddings
                if sbh:  # SBH
                    s, b = hidden_states.shape[0], hidden_states.shape[1]
                else:    # BSH
                    b, s = hidden_states.shape[0], hidden_states.shape[1]

                # CP zigzag (load-balanced) is applied per-sample in pack mode, or on the
                # whole sequence otherwise; SP then narrows the resulting stream by sp_rank.
                if cu_seqlens is None:
                    original_seq_len = s * self.cp_size * self.sp_size
                    positions = sp_narrow_positions(
                        cp_zigzag_positions(original_seq_len, self.cp_size, self.cp_rank, hidden_states.device),
                        self.sp_size, self.sp_rank,
                    )
                else:
                    assert b == 1, f'Sequence parallel with cu_seqlens only supports batch size of 1, but got batch size {b}.'
                    # Pack mode: each sample restarts positions at 0; CP zigzag is per-sample,
                    # SP narrows the concatenated stream. cu_seqlens is the ORIGINAL
                    # (pre-CP/SP) cumulative sequence lengths.
                    seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                    positions = sp_narrow_positions(
                        torch.cat([
                            cp_zigzag_positions(L, self.cp_size, self.cp_rank, hidden_states.device)
                            for L in seq_lens
                        ]),
                        self.sp_size, self.sp_rank,
                    )

                position_ids = positions.unsqueeze(1).expand(s, b) if sbh else positions.unsqueeze(0).expand(b, s)

            hidden_states = hidden_states + self.embed_positions(position_ids)

        hidden_states = self.drop(hidden_states)
        return hidden_states


# =========================================================================
# Attention layer
# =========================================================================

class GalvatronAttention(nn.Module):
    """Pre-norm self-attention with residual connection."""

    def __init__(self, args: GalvatronRuntimeArgs, layer_idx, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        m = args.model
        self.sequence_parallel = args.train.sequence_parallel
        self.sp_size = sp_group.size if sp_group is not None else 1
        self.cp_size = cp_group.size if cp_group is not None else 1
        self.tp_size = tp_group.size if tp_group is not None else 1
        self.use_ulysses = self.sp_size > 1
        self.use_zigzag_cp = self.cp_size > 1

        self.layer_idx = layer_idx
        self.cp_group = cp_group.group if cp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.tp_group = tp_group.group if tp_group is not None else None
        self.cp_ranks = cp_group.ranks if cp_group is not None else None

        if m.qk_layernorm:
            q_ln = nn.LayerNorm
            k_ln = nn.LayerNorm
        else:
            q_ln = None
            k_ln = None

        self.attention = SelfAttention(
            m,
            SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                flash_attention=FlashSelfOrCrossAttention,
                dist_attention=DistributedAttention,
                zigzag_ring_flash_attn=ZigzagRingFlashAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=q_ln,
                k_layernorm=k_ln,
            ),
            layer_idx,
            attn_mask_type=AttnMaskType.causal,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
            cp_group=self.cp_group,
            cp_ranks=self.cp_ranks,
        )

        self.input_layernorm = GalvatronNorm(m, m.hidden_size, eps=m.norm_epsilon)

        self.head_dim = m.kv_channels or (m.hidden_size // m.num_attention_heads)
        self.use_rope = m.position_embedding_type in ("rope", "mrope")

    def forward(self, hidden_states, position_ids=None, attention_mask=None, rotary_embedding=None, cu_seqlens=None):
        if self.use_rope:
            assert rotary_embedding is not None, "rotary_embedding must be provided for attention when using RoPE"

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        packed_seq_params: PackedSeqParams = None
        if cu_seqlens is not None:
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            )
        
        hidden_states, attn_bias = self.attention(hidden_states, attention_mask, rotary_pos_emb=rotary_embedding, packed_seq_params=packed_seq_params)
        if attn_bias is not None:
            hidden_states = hidden_states + attn_bias
        return hidden_states + residual


# =========================================================================
# MLP layer
# =========================================================================

class GalvatronMLP(nn.Module):
    """Pre-norm feed-forward with residual connection."""

    def __init__(self, args: GalvatronRuntimeArgs, layer_idx, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        m = args.model
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.cp_group = cp_group.group if cp_group is not None else None

        self.mlp = MLP(
            m,
            MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
            tp_group=self.tp_group,
        )

        self.post_attention_layernorm = GalvatronNorm(m, m.hidden_size, eps=m.norm_epsilon)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_bias = self.mlp(hidden_states)
        if mlp_bias is not None:
            hidden_states = hidden_states + mlp_bias
        return hidden_states + residual


# =========================================================================
# Decoder layer (attention + mlp combined)
# =========================================================================

class GalvatronDecoderLayer(nn.Module):
    """Pre-norm decoder block = ``GalvatronAttention`` + ``GalvatronMLP``."""

    def __init__(self, args: GalvatronRuntimeArgs, layer_idx, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        self.idx = layer_idx
        self.attn = GalvatronAttention(args, layer_idx, tp_group, sp_group, cp_group)
        self.ffn = GalvatronMLP(args, layer_idx, tp_group, sp_group, cp_group)

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None, cu_seqlens=None):
        hidden_states = self.attn(hidden_states, position_ids, attention_mask, rotary_embedding, cu_seqlens)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


# =========================================================================
# Final norm
# =========================================================================

class GalvatronFinalNorm(nn.Module):
    """Final normalization layer before the LM head."""

    def __init__(self, args: GalvatronRuntimeArgs):
        super().__init__()
        m = args.model
        self.norm = GalvatronNorm(m, m.hidden_size, eps=m.norm_epsilon)

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None, cu_seqlens=None):
        return self.norm(hidden_states)


# =========================================================================
# LM head
# =========================================================================

class _LMHeadLinear(nn.Module):
    """TP-aware linear projection (for LM head)."""

    def __init__(self, config, sequence_parallel, tp_group):
        super().__init__()
        world_size = parallel_state.get_parallel_world_size(tp_group)
        self.weight = nn.Parameter(
            torch.empty(
                divide(config.padded_vocab_size, world_size),
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        world_size = parallel_state.get_parallel_world_size(tp_group)
        if self.sequence_parallel and world_size <= 1:
            self.sequence_parallel = False

    def forward(self, hidden_states):
        return linear_with_grad_accumulation_and_async_allreduce(
            input=hidden_states,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
        )


class GalvatronCausalLMHead(nn.Module):
    """TP-aware causal language model head with vocab-parallel cross-entropy."""

    def __init__(self, args: GalvatronRuntimeArgs, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        m = args.model
        self.sequence_parallel = args.train.sequence_parallel
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.cp_group = cp_group.group if cp_group is not None else None
        self.parallel_loss = True
        self.half_entropy = not args.parallel.entropy_in_fp32
        self.vocab_sp = args.parallel.vocab_sp

        self.lm_head = _LMHeadLinear(m, self.sequence_parallel, self.tp_group)

        self.sp_size = parallel_state.get_parallel_world_size(self.sp_group) if self.sp_group is not None else 1
        self.sp_rank = parallel_state.get_parallel_rank(self.sp_group) if self.sp_group is not None else 0

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None, cu_seqlens=None):
        if self.vocab_sp:
            labels_lens = labels.shape[1] # labels.shape is (b, seq_len)
            partition_size = divide(labels_lens, self.sp_size)
            start_index, end_index = self.sp_rank * partition_size, (self.sp_rank + 1) * partition_size
            labels = labels[:, start_index:end_index].contiguous()

        if not self.sequence_parallel:
            hidden_states = copy_to_tensor_model_parallel_region(hidden_states, self.tp_group)

        logits_parallel = self.lm_head(hidden_states)
        labels = labels.transpose(0, 1).contiguous()

        if not self.parallel_loss:
            output = gather_from_tensor_model_parallel_region(logits_parallel, self.tp_group)
            logits = output if self.half_entropy else output.float()
            shift_logits = logits.contiguous().view(-1, logits.size(-1))
            shift_labels = labels.contiguous().view(-1).to(shift_logits.device)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels)
        else:
            loss = fused_vocab_parallel_cross_entropy(
                logits_parallel, labels, self.half_entropy, tp_group=self.tp_group,
            )
            if self.vocab_sp:
                loss = loss.transpose(0, 1).contiguous() # (seq_len, b) -> (b, seq_len) for consistency with the case without vocab parallelism, but this is really just for better logging and debugging, it doesn't affect the actual loss value since it's just a transpose
                loss = gather_from_tensor_model_parallel_region(loss, self.sp_group)

        if self.vocab_sp == False:
            loss = loss.transpose(0, 1).contiguous()
        return loss


from .moe_modules import (
    GalvatronMoEAttention,
    GalvatronMoERouter,
    GalvatronMoEMLP,
    GalvatronMoEDecoderLayer,
)
