import torch
import torch.nn as nn

# from transformers.models.llama.modeling_llama import LlamaRMSNorm
# from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm
from flash_attn.ops.rms_norm import RMSNorm as LlamaRMSNorm
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from galvatron.core import get_args
from galvatron.core.runtime import ModelInfo, mixed_precision_dtype
from galvatron.core.runtime.pipeline import PipeSequential
from galvatron.core.runtime.tensor_parallel import colummn_row_reset_parameters

class LlamaEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.model
        self.embed_tokens = model.embed_tokens
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        self.tp_group = self.embed_tokens.tp_group
        self.sp_group = self.embed_tokens.sp_group
        self.cp_group = self.embed_tokens.cp_group
        self.cp_size = torch.distributed.get_world_size(self.cp_group) if self.cp_group is not None else 1
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            seq_ulysses = int(args.seq_length / self.cp_size)
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                seq_ulysses,
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group),
            )

    def forward(self, tokens, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None):
        # tokens = input_ids[:, :-1].contiguous()
        # labels = input_ids[:, 1:].contiguous()
        if self.vocab_sp:
            tokens = tokens[:, self.seq_start_index : self.seq_end_index].contiguous()
        
        # [b, s] -> [s /cp / tp, b, h]
        hidden_states = self.embed_tokens(tokens)
        return hidden_states


class LlamaLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        model = model.model
        self.layer = model.layers[layer_idx]
        self.layer_idx = layer_idx

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None):
        # attention_mask = get_ltor_masks_and_position_ids(input_ids)
        hidden_states = self.layer(hidden_states, attention_mask=attention_mask, rotary_embedding=rotary_embedding)  # , position_ids = position_ids)
        return hidden_states


class LlamaPreNorm_(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None):
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaLoss_(nn.Module):
    def __init__(self, weight, sequence_parallel, tp_group):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        world_size = mpu.get_tensor_model_parallel_world_size(tp_group)
        if self.sequence_parallel and world_size <= 1:
            self.sequence_parallel = False
            # disable sp to avoid global buffer

    def forward(self, hidden_states):
        logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
            input=hidden_states,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
        )
        return logits_parallel


class LlamaCls_(nn.Module):
    def __init__(self, model, parallel_loss=True, half_entropy=True):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = model.lm_head.tp_group
        self.sp_group = model.lm_head.sp_group
        self.cp_group = model.lm_head.cp_group
        self.cp_size = torch.distributed.get_world_size(self.cp_group) if self.cp_group is not None else 1
        self.lm_head = LlamaLoss_(model.lm_head.weight, self.sequence_parallel, self.tp_group)
        self.clone_scatter_output_in_embedding = get_args().clone_scatter_output_in_embedding
        self.parallel_loss = parallel_loss
        self.half_entropy = half_entropy
        args = get_args()
        if args.entropy_in_fp32:
            self.half_entropy = False
        self.seq_length = args.seq_length
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                self.seq_length // self.cp_size,
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group),
            )

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None):
        if self.vocab_sp:
            labels = labels[:, self.seq_start_index : self.seq_end_index].contiguous()
        if not self.sequence_parallel:
            hidden_states = copy_to_tensor_model_parallel_region(hidden_states, self.tp_group)

        logits_parallel = self.lm_head(hidden_states)

        # [b s] -> [s b]
        labels = labels.transpose(0, 1).contiguous()

        # loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), input_ids)
        if not self.parallel_loss:
            output = gather_from_tensor_model_parallel_region(logits_parallel, self.tp_group)
            if not self.half_entropy:
                logits = output.float()
            else:
                logits = output
            loss = None
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous()  # logits[:-1, ..., :].contiguous()
            shift_labels = labels.contiguous()  # input_ids[1:, ...].contiguous()
            # Flatten the tokens
            from torch.nn import CrossEntropyLoss

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = fused_vocab_parallel_cross_entropy(logits_parallel, labels, self.half_entropy, tp_group=self.tp_group)
            # loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel, labels, self.half_entropy, tp_group=self.tp_group)
            if self.vocab_sp:
                loss = gather_from_tensor_model_parallel_region(loss, self.sp_group)
            # loss = loss.mean()
        loss = loss.transpose(0, 1).contiguous()
        return loss


def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module("embeddings", LlamaEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = LlamaLayers_(model, i)
        model_.add_module("layer_%d" % i, enc)
    model_.add_module("prenorm", LlamaPreNorm_(model, config))
    model_.add_module("cls", LlamaCls_(model))
    LlamaLoss_.reset_parameters = colummn_row_reset_parameters
    return model_


class LlamaModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(LlamaModelInfo, self).__init__()
        layernum_list = [config.num_hidden_layers]
        seq_len, hidden_size = config.max_position_embeddings, config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        if args.shape_order == "SBH":
            layer_shapes_list = [[[seq_len, -1, hidden_size]]]
        else:
            layer_shapes_list = [[[-1, seq_len, hidden_size]]]
        layer_dtypes_list = [[mixed_precision]]
        module_types = ["embed"] + ["gpt_dec"] * config.num_hidden_layers + ["norm", "cls"]
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)
