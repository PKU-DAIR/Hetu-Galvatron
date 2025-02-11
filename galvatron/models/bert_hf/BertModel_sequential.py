import torch.distributed
import torch.nn as nn
import torch
from galvatron.core.pipeline import PipeSequential
from galvatron.core import mixed_precision_dtype, ModelInfo
from galvatron.core import get_args
from megatron.core import tensor_parallel
from galvatron.core.tensor_parallel import colummn_row_reset_parameters
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.mappings_group import get_tensor_model_parallel_world_size_group

def bert_extended_attention_mask(attention_mask):
    attention_mask_b1s = attention_mask.unsqueeze(1)
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    extended_attention_mask = attention_mask_bss.unsqueeze(1)
    extended_attention_mask = (extended_attention_mask < 0.5)
    return extended_attention_mask

class BertWordEmbedding_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.word_embeddings = model.bert.embeddings.word_embeddings
    def forward(self, input_ids):
        return self.word_embeddings(input_ids)

class BertTokenTypeEmbedding_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.token_type_embeddings = model.bert.embeddings.token_type_embeddings
    def forward(self, token_type_ids):
        return self.token_type_embeddings(token_type_ids)

class BertPositionEmbedding_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.position_embeddings = model.bert.embeddings.position_embeddings
    def forward(self, position_ids):
        return self.position_embeddings(position_ids)
        
class BertEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.word_embeddings = BertWordEmbedding_(model)
        self.token_type_embeddings = BertTokenTypeEmbedding_(model)
        self.position_embeddings = BertPositionEmbedding_(model)
        
        args = get_args()
        self.hidden_dropout = torch.nn.Dropout(args.hidden_dropout)
        self.LayerNorm = model.bert.embeddings.LayerNorm
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        
        self.tp_group = self.word_embeddings.word_embeddings.tp_group
        self.sp_group = self.word_embeddings.word_embeddings.sp_group

        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                args.seq_length,
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group)
            )
    
    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, labels=None):
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if self.vocab_sp:
            input_ids = input_ids[:, self.seq_start_index:self.seq_end_index].contiguous()
            token_type_ids = token_type_ids[:, self.seq_start_index:self.seq_end_index].contiguous()
            position_ids = position_ids[:, self.seq_start_index:self.seq_end_index].contiguous()
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        
        # [b,s,h] -> [s,b,h]
        embeddings = embeddings.transpose(0, 1).contiguous()
   
        if self.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region_group(
                embeddings,
                self.tp_group
            )
            # Clone if needed
            if self.clone_scatter_output_in_embedding:
                embeddings = embeddings.clone()
            # Dropout with RNG tracker
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.hidden_dropout(embeddings)
        else:
            embeddings = self.hidden_dropout(embeddings)
            
        return embeddings

class BertLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        self.layer = model.bert.encoder.layer[layer_idx]

    def forward(self, hidden_states, token_type_ids=None, position_ids=None, attention_mask=None, labels=None):
        if attention_mask is not None:
            attention_mask = bert_extended_attention_mask(attention_mask)
        hidden_states = self.layer(
            hidden_states,
            attention_mask=attention_mask
        )
        return hidden_states

#For next sentence prediction
# class BertPooler_(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.pooler = model.bert.pooler
        
#     def forward(self, hidden_states, **kwargs):
#         pooled_output = self.pooler(hidden_states)
#         return hidden_states, pooled_output

class BertLoss_(nn.Module):
    def __init__(self, weight, bias, sequence_parallel, tp_group):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone())
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
            bias=self.bias,
            gradient_accumulation_fusion=False,
            async_grad_allreduce=False,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group
        )
        return logits_parallel

class BertMLMCls_(nn.Module):
    def __init__(self, model, parallel_loss=True, half_entropy=True):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = model.cls.predictions.decoder.tp_group
        self.sp_group = model.cls.predictions.decoder.sp_group
        self.transform = model.cls.predictions.transform
        self.lm_head = BertLoss_(
            model.cls.predictions.decoder.weight,
            model.cls.predictions.decoder.bias,
            self.sequence_parallel,
            self.tp_group
        )
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
                self.seq_length, 
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group)
            )

    def forward(self, hidden_states, token_type_ids=None, position_ids=None, attention_mask=None, labels=None):
        if self.vocab_sp:
            labels = labels[:, self.seq_start_index:self.seq_end_index].contiguous()

        if not self.sequence_parallel:
            hidden_states = tensor_parallel.copy_to_tensor_model_parallel_region_group(
                hidden_states, 
                self.tp_group
            )
        
        hidden_states = self.transform(hidden_states)
        logits_parallel = self.lm_head(hidden_states)
        
        if not self.parallel_loss:
            # gather logits
            output = tensor_parallel.gather_from_tensor_model_parallel_region_group(
                logits_parallel,
                self.tp_group
            )
            logits = output if self.half_entropy else output.float()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),  # [batch_size * seq_length, vocab_size]
                labels.view(-1)                     # [batch_size * seq_length]
            )
        else:
            if not self.vocab_sp:
        
                logits_parallel = logits_parallel.view(-1, logits_parallel.size(-1))  # [batch_size * seq_len, vocab_size/world_size]
                labels = labels.view(-1)  # [batch_size * seq_len]
                
                loss = tensor_parallel.vocab_parallel_cross_entropy(
                    logits_parallel if self.half_entropy else logits_parallel.float(),
                    labels,
                    tp_group=self.tp_group
                )
            else:
                # 序列并行处理
                logits_parallel = logits_parallel.view(-1, logits_parallel.size(-1))
                labels = labels.view(-1)
                
                loss = tensor_parallel.vocab_sequence_parallel_cross_entropy(
                    logits_parallel if self.half_entropy else logits_parallel.float(),
                    labels,
                    self.sp_group
                )
        return loss
        
def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('embeddings', BertEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = BertLayers_(model, i)
        model_.add_module(f'layer_{i}', enc)
    model_.add_module('cls', BertMLMCls_(model))
    BertLoss_.reset_parameters = colummn_row_reset_parameters
    return model_

class BertModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(BertModelInfo, self).__init__()
        layernum_list = [config.num_hidden_layers]
        seq_len = config.max_position_embeddings
        hidden_size = config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        if args.shape_order == "SBH":
            layer_shapes_list = [[[seq_len, -1, hidden_size]]]
        else:
            layer_shapes_list = [[[-1, seq_len, hidden_size]]]
        layer_dtypes_list = [[mixed_precision]]
        module_types = ['embed'] + ['bert_enc'] * config.num_hidden_layers + ['mlm_head']
        
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)