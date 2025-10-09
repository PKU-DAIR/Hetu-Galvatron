import galvatron.core.tensor_parallel.transformer as transformer
import math
import torch
import torch.nn.functional as F
import megatron
from megatron.training import get_args
from megatron.core import tensor_parallel
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.mappings_group import get_tensor_model_parallel_rank_group, get_tensor_model_parallel_world_size_group, reduce_from_tensor_model_parallel_region_group
from megatron.legacy.model.enums import AttnMaskType, LayerType, AttnType
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding, apply_rotary_pos_emb

from functools import wraps
import torch_npu
from mindspeed.model.transformer import (
    flash_self_attention_forward,
    flash_self_attention_init_wrapper,
    core_attention_forward,
    core_attention_init_wrapper,
    parallel_mlp_forward_wrapper,
    parallel_mlp_init_wrapper,
    
)
try:
    from einops import rearrange
except ImportError:
    rearrange = None

def parallel_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        config = args[0]
        tp_group = kwargs['tp_group']
        sp_group = kwargs['sp_group']
        query_projection_size = config.kv_channels * config.num_attention_heads
        _args = get_args()
        if _args.group_query_attention:
            kv_projection_size = _args.kv_channels * _args.num_query_groups
        else:
            kv_projection_size = _args.kv_channels * _args.num_attention_heads
        # qkv bias
        bias = _args.add_qkv_bias or _args.add_bias_linear
        if args[0].context_parallel_size > 1 and args[0].context_parallel_algo in ['ulysses_cp_algo', 'hybrid_cp_algo']:
            u_group = mpu.get_context_parallel_group()
            if args[0].context_parallel_algo == 'hybrid_cp_algo':
                u_group = get_context_parallel_group_for_hybrid_ulysses()
            if self.use_flash_attn:
                self.core_attention_flash = UlyssesContextAttention(self.core_attention_flash, u_group)
            else:
                self.core_attention = UlyssesContextAttention(self.core_attention, u_group)
        
        if _args.use_ulysses:
            self.dist_attn = transformer.DistributedAttention(self.core_attention_flash if self.use_flash_attn else self.core_attention, sp_group, 
                                                  gather_idx = 0,)
        
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=False,
            tp_group=tp_group)
        # dense bias
        bias = _args.add_dense_bias or _args.add_bias_linear
        skip_bias_add = _args.skip_bias_add
        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=skip_bias_add,
            tp_group=tp_group)
    return wrapper

def parallel_attention_forward(self, hidden_states, attention_mask,
            encoder_output=None, inference_params=None,
            rotary_pos_emb=None):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    is_first_step = False
    if inference_params:
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_len = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)
            inference_value_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)

            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory, inference_value_memory)
            is_first_step = True
        else:
            inference_key_memory, inference_value_memory = \
                inference_params.key_value_memory_dict[self.layer_number]

    # =====================
    # Query, Key, and Value
    # =====================
    if self.attention_type == AttnType.self_attn:

        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query_layer,
        key_layer,
        value_layer) = torch.split(
            mixed_x_layer,
            [
                (
                    self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head
            ],
            dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
    else:
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv_layer, _ = self.key_value(encoder_output)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key_layer,
        value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query_layer, _ = self.query(hidden_states)
        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_tensor_shape)

    # ==================================
    # Adjust key and value for inference
    # ==================================

    # duplicate the pos_emb for self attention
    if rotary_pos_emb is not None:
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb
        else:
            rotary_pos_emb = ((rotary_pos_emb,) * 2)

    if inference_params:
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key_layer.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key_layer.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end,
                                batch_start:batch_end, ...] = key_layer
        inference_value_memory[sequence_start:sequence_end,
                                batch_start:batch_end, ...] = value_layer
        key_layer = inference_key_memory[
            :sequence_end, batch_start:batch_end, ...]
        value_layer = inference_value_memory[
            :sequence_end, batch_start:batch_end, ...]


        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # need to cross check this condition during inference
            # if not set_inference_key_value_memory:
            if not is_first_step:
                # In inference, we compute one token at a time.
                # Select the correct positional embedding
                # (only the last token in the sequence)
                q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
            else:
                # In the first forward pass of inference,
                # we use the entire provided prefix.
                # q_pos_emb here has the rope embeddings of the entire
                # prefix + to-be-generated output so
                # we slice to just the prefix.
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

    # ==================================
    # core attention computation
    # ==================================

    # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
    if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
        key_layer = key_layer.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim=2
        )
        value_layer = value_layer.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim=2
        )

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb, self.config)
        key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb, self.config)
        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

    if self.use_ulysses:
        if self.use_flash_attn:
            batch_dim_idx = 1
            context_layer = self.dist_attn(query_layer, key_layer, value_layer, batch_dim_idx, attention_mask)
            context_layer = rearrange(context_layer, '... h d -> ... (h d)').contiguous()     
        else:
            batch_dim_idx = 1 # [S,B,H,D]
            context_layer = self.dist_attn(query_layer, key_layer, value_layer, batch_dim_idx, attention_mask)
            context_layer = rearrange(context_layer, '... h d -> ... (h d)').contiguous() 
    else: 
        if not self.use_flash_attn:
            if self.checkpoint_core_attention:
                context_layer = self._checkpointed_attention_forward(
                    query_layer, key_layer, value_layer, attention_mask)
            else:
                context_layer = self.core_attention(
                    query_layer, key_layer, value_layer, attention_mask)
        else:
            if not self.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context_layer = self.core_attention_flash(query_layer, key_layer, value_layer, attention_mask)
            else:
                context_layer = self.core_attention_flash(query_layer, key_layer, value_layer, attention_mask)

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.dense(context_layer)

    return output, bias

def vocab_parallel_cross_entropy_forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0, tp_group=None):
    # Maximum value along vocab dimension across all GPUs.
    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    if tp_group == None:
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )
    else:
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group
        )
    
    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

    # Get the partition's vocab indecies
    get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
    partition_vocab_size = vocab_parallel_logits.size()[-1]
    if tp_group == None:
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
    else:
        rank = get_tensor_model_parallel_rank_group(tp_group)
        world_size = get_tensor_model_parallel_world_size_group(tp_group)
    
    vocab_start_index, vocab_end_index = get_vocab_range(
        partition_vocab_size, rank, world_size)

    # Create a mask of valid vocab ids (1 means it needs to be masked).
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target *= ~target_mask

    # Get predicted-logits = logits[target].
    # For Simplicity, we convert logits to a 2-D tensor with size
    # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
    masked_target_1d = masked_target.view(-1)
    arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                             device=logits_2d.device)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
    predicted_logits_1d = predicted_logits_1d.clone().contiguous()
    predicted_logits = predicted_logits_1d.view_as(target)
    predicted_logits *= ~target_mask
    # All reduce is needed to get the chunks from other GPUs.
    if tp_group == None:
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )
    else:
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )

    # Sum of exponential of logits along vocab dimension across all GPUs.
    exp_logits = vocab_parallel_logits
    torch.exp(vocab_parallel_logits, out=exp_logits)
    sum_exp_logits = exp_logits.sum(dim=-1)
    if tp_group == None:
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )
    else:
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
    loss = torch.log(sum_exp_logits) - predicted_logits

    # Normalize and optionally smooth logits
    exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

    vocab_size = exp_logits.size(-1)
    if label_smoothing > 0:
        """
        We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
        = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
        = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
        = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
        = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
        = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
        """
        if label_smoothing >= 1.0:
            raise ValueError("label_smoothing value should in (0,1)")
        smoothing = label_smoothing * vocab_size / (vocab_size - 1)

        # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
        log_probs = torch.log(exp_logits)
        mean_log_probs = log_probs.mean(dim=-1)
        loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

    ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

    # Store softmax, target-mask and masked-target for backward pass.
    ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

    return loss

def vocab_parallel_embedding_forward(self, input_):
    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input *= ~input_mask
    else:
        masked_input = input_
        # Get the embeddings.

    # For higher accumulation accuracy for bf16 on NPU.
    output_parallel = F.embedding(masked_input, self.weight)

    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    # Reduce across all the model parallel GPUs.
    if self.tp_group is None:
        output = reduce_from_tensor_model_parallel_region(output_parallel)
    else:
        output = reduce_from_tensor_model_parallel_region_group(output_parallel, self.tp_group)
    return output

def exe_adaptation():
    megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward = vocab_parallel_cross_entropy_forward
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = vocab_parallel_embedding_forward
    
    transformer.ParallelMLP.__init__ = parallel_mlp_init_wrapper(transformer.ParallelMLP.__init__)
    transformer.ParallelMLP.forward = parallel_mlp_forward_wrapper(transformer.ParallelMLP.forward)
    transformer.FlashSelfAttention.forward = flash_self_attention_forward
    transformer.FlashSelfAttention.__init__ = flash_self_attention_init_wrapper(transformer.FlashSelfAttention.__init__)
    transformer.ParallelAttention.__init__ = parallel_attention_init_wrapper(transformer.ParallelAttention.__init__)
    transformer.ParallelAttention.forward = parallel_attention_forward
    transformer.CoreAttention.__init__ = core_attention_init_wrapper(transformer.CoreAttention.__init__)
    transformer.CoreAttention.forward = core_attention_forward

    # transformer.ParallelMLP.__init__ = parallel_mlp_init
    # transformer.FlashSelfAttention.forward = flash_self_attention_forward
    # transformer.apply_rotary_pos_emb = apply_fused_rotary_pos_emb
    # transformer.ParallelAttention.__init__ = ParallelAttention_wrapper(transformer.ParallelAttention.__init__)
    # transformer.CoreAttention.__init__ = core_attention_wrapper(transformer.CoreAttention.__init__)
    # transformer.CoreAttention.forward = core_attention_forward
    # transformer.ParallelTransformerLayer.forward = parallel_transformer_layer_forward_wrapper(transformer.ParallelTransformerLayer.forward)
    # transformer.ParallelTransformer._checkpointed_forward = parallel_transformer_checkpointed_forward_wrapper(transformer.ParallelTransformer._checkpointed_forward)
    
    # megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward = _VocabParallelCrossEntropyForward
    # megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = VocabParallelEmbeddingForward