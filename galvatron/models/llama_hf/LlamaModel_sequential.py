import torch.nn as nn
import torch
from galvatron.core.pipeline import PipeSequential
from galvatron.core import mixed_precision_dtype, ModelInfo
from galvatron.core import get_args
from megatron.core import tensor_parallel
# from transformers.models.llama.modeling_llama import LlamaRMSNorm
from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm
# from flash_attn.ops.rms_norm import RMSNorm as LlamaRMSNorm
from galvatron.core.tensor_parallel import colummn_row_reset_parameters
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.mappings_group import get_tensor_model_parallel_world_size_group
from mindspeed.model.transformer import get_attention_mask

def get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""
    micro_batch_size, seq_length = data.size()
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
    
    # position_ids = torch.arange(seq_length, dtype=torch.long,
    #                             device=data.device)
    # position_ids = position_ids.unsqueeze(0).expand_as(data)
    attention_mask = (attention_mask < 0.5)

    return attention_mask# , position_ids


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
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    args.seq_length, torch.distributed.get_rank(self.sp_group), torch.distributed.get_world_size(self.sp_group)
                )
            
        
    def forward(self, tokens, position_ids=None, attention_mask=None, labels=None):
        # tokens = input_ids[:, :-1].contiguous()
        # labels = input_ids[:, 1:].contiguous()
        if self.vocab_sp:
            tokens = tokens[:, self.seq_start_index:self.seq_end_index].contiguous()
        hidden_states = self.embed_tokens(tokens)
        # [b, s, h] -> [s, b, h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        
        if self.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region_group(hidden_states, self.tp_group)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.clone_scatter_output_in_embedding:
                hidden_states = hidden_states.clone()

        return hidden_states

class LlamaLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        model = model.model
        self.layer = model.layers[layer_idx]

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None):
        if attention_mask == None:
            attention_mask = get_attention_mask()
        # attention_mask = get_ltor_masks_and_position_ids(input_ids)
        # if torch.cuda.current_device() == 0:
        #     print(hidden_states)
        hidden_states = self.layer(hidden_states, attention_mask = attention_mask) # , position_ids = position_ids)
        return hidden_states

class LlamaPreNorm_(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None):
        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaLoss_(nn.Module):
    def __init__(self, weight, sequence_parallel, tp_group):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        world_size = get_tensor_model_parallel_world_size_group(tp_group)
        if self.sequence_parallel and world_size <= 1:
            self.sequence_parallel = False
            # disable sp to avoid global buffer
            
    
    def forward(self, hidden_states):
        logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
            input=hidden_states,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=False,
            async_grad_allreduce=False,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group)
        return logits_parallel


class LlamaCls_(nn.Module):
    def __init__(self, model, parallel_loss = True, half_entropy = True):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = model.lm_head.tp_group
        self.sp_group = model.lm_head.sp_group
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
                    self.seq_length, torch.distributed.get_rank(self.sp_group), torch.distributed.get_world_size(self.sp_group)
                )
    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None):
        if self.vocab_sp:
            labels = labels[:, self.seq_start_index:self.seq_end_index].contiguous()
            
        if not self.sequence_parallel:
            hidden_states = tensor_parallel.copy_to_tensor_model_parallel_region_group(hidden_states, self.tp_group)
        
        logits_parallel = self.lm_head(hidden_states)
        
        # [b s] -> [s b]
        labels = labels.transpose(0,1).contiguous()
        
        # loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), input_ids)
        if not self.parallel_loss:
            output = tensor_parallel.gather_from_tensor_model_parallel_region_group(logits_parallel, self.tp_group)
            if not self.half_entropy:
                logits = output.float()
            else:
                logits = output
            loss = None
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous() # logits[:-1, ..., :].contiguous()
            shift_labels = labels.contiguous() # input_ids[1:, ...].contiguous()
            # Flatten the tokens
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            if not self.half_entropy:
                loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel.float(), labels, tp_group = self.tp_group)
            else:
                loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel, labels, tp_group = self.tp_group)
            if self.vocab_sp:
                loss = tensor_parallel.gather_from_tensor_model_parallel_region_group(loss, self.sp_group)
            # loss = loss.mean()
        loss = loss.transpose(0,1).contiguous()
        return loss

def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('embeddings', LlamaEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = LlamaLayers_(model, i)
        model_.add_module('layer_%d'%i, enc)
    model_.add_module('prenorm', LlamaPreNorm_(model, config))
    model_.add_module('cls', LlamaCls_(model))
    LlamaLoss_.reset_parameters = colummn_row_reset_parameters
    return model_

class LlamaModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(LlamaModelInfo, self).__init__()
        layernum_list = [config.num_hidden_layers]
        seq_len, hidden_size = config.max_position_embeddings, config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        if args.shape_order == "SBH":
            layer_shapes_list = [[[seq_len,-1,hidden_size]]]
        else:
            layer_shapes_list = [[[-1,seq_len,hidden_size]]]
        layer_dtypes_list = [[mixed_precision]]
        module_types = ['embed'] + ['gpt_dec']*config.num_hidden_layers + ['norm', 'cls']
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)