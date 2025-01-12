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

class BertVocabEmbedding_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.word_embeddings = model.bert.embeddings.word_embeddings
        
    def forward(self, tokens):
        return self.word_embeddings(tokens)

class BertPositionEmbedding_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.position_embeddings = model.bert.embeddings.position_embeddings
        
    def forward(self, position_ids):
        return self.position_embeddings(position_ids)

class BertTokenTypeEmbedding_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.token_type_embeddings = model.bert.embeddings.token_type_embeddings
        
    def forward(self, token_type_ids):
        return self.token_type_embeddings(token_type_ids)

class BertEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.bert
        self.word_embeddings = BertVocabEmbedding_(model)
        self.position_embeddings = BertPositionEmbedding_(model)
        self.token_type_embeddings = BertTokenTypeEmbedding_(model)
        args = get_args()
        self.LayerNorm = model.embeddings.LayerNorm
        self.dropout = nn.Dropout(args.hidden_dropout)
        self.sequence_parallel = args.sequence_parallel
        self.tp_group = getattr(self.word_embeddings.word_embeddings, "tp_group", None)
        self.sp_group = getattr(self.word_embeddings.word_embeddings, "sp_group", None)
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                args.seq_length, torch.distributed.get_rank(self.sp_group), torch.distributed.get_world_size(self.sp_group)
            )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        if self.vocab_sp:
            input_ids = input_ids[:, self.seq_start_index:self.seq_end_index].contiguous()
            
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        
        if self.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region_group(embeddings, self.tp_group)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.dropout(embeddings)
        else:
            embeddings = self.dropout(embeddings)
            
        return embeddings, attention_mask



class BertLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        model = model.bert
        self.layer = model.encoder.layer[layer_idx]
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = getattr(self.layer, "tp_group", None)
        self.sp_group = getattr(self.layer, "sp_group", None)

    def forward(self, hidden_states, attention_mask=None, labels=None):
        layer_kwargs = {'seqlen': hidden_states.shape[1]} if (
            self.sp_group is not None and self.sequence_parallel
        ) else {}
        
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_mask = extended_attention_mask
            
        hidden_states = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            **layer_kwargs
        )
        return hidden_states, attention_mask

class BertPreNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = None
        self.sp_group = None

    def forward(self, hidden_states, attention_mask=None, labels=None):
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states, attention_mask

class BertLoss_(nn.Module):
    def __init__(self, weight, sequence_parallel, tp_group):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        world_size = get_tensor_model_parallel_world_size_group(tp_group)
        if self.sequence_parallel and world_size <= 1:
            self.sequence_parallel = False
    
    def forward(self, hidden_states):
        logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
            input=hidden_states,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=False,
            async_grad_allreduce=False,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group
        )
        return logits_parallel

class BertCls_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = getattr(model.cls.predictions.decoder, "tp_group", None)
        self.sp_group = getattr(model.cls.predictions.decoder, "sp_group", None)
        self.predictions = model.cls.predictions  
        self.seq_relationship = model.cls.seq_relationship  
        
        self.mlm_loss = BertLoss_(
            model.cls.predictions.decoder.weight,
            self.sequence_parallel,
            self.tp_group
        )
        
        args = get_args()
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                args.seq_length, 
                torch.distributed.get_rank(self.sp_group), 
                torch.distributed.get_world_size(self.sp_group)
            )

    def forward(self, hidden_states, attention_mask=None, labels=None, next_sentence_label=None):
        if self.vocab_sp:
            labels = labels[:, self.seq_start_index:self.seq_end_index].contiguous()
            
        if not self.sequence_parallel:
            hidden_states = tensor_parallel.copy_to_tensor_model_parallel_region_group(hidden_states, self.tp_group)

        mlm_hidden = self.predictions.transform(hidden_states)
        prediction_scores = self.mlm_loss(mlm_hidden)
        
        seq_relationship_score = self.seq_relationship(hidden_states[:, 0])
        
        outputs = (prediction_scores, seq_relationship_score)
        
        if labels is not None:
            loss_fct = tensor_parallel.vocab_parallel_cross_entropy
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
                tp_group=self.tp_group
            )
            
            if next_sentence_label is not None:
                next_sentence_loss = nn.CrossEntropyLoss()(
                    seq_relationship_score.view(-1, 2),
                    next_sentence_label.view(-1)
                )
                total_loss = masked_lm_loss + next_sentence_loss
                return total_loss
            return masked_lm_loss
            
        return outputs

class BertPooler_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = getattr(model.bert.pooler, "tp_group", None)
        self.sp_group = getattr(model.bert.pooler, "sp_group", None)
        self.dense = model.bert.pooler.dense
        self.activation = torch.tanh

    def forward(self, hidden_states, attention_mask=None, labels=None):
        first_token_tensor = hidden_states[:, 0]
        
        if self.sequence_parallel:
            first_token_tensor = tensor_parallel.gather_from_sequence_parallel_region_group(
                first_token_tensor, self.tp_group
            )
            
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return hidden_states, pooled_output, attention_mask

def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('embeddings', BertEmbeddings_(model))
    
    for i in range(config.num_hidden_layers):
        model_.add_module(f'layer_{i}', BertLayers_(model, i))
    
    model_.add_module('prenorm', BertPreNorm_(model))
    model_.add_module('pooler', BertPooler_(model))
    model_.add_module('cls', BertCls_(model))
    
    BertLoss_.reset_parameters = colummn_row_reset_parameters
    return model_

class BertModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(BertModelInfo, self).__init__()
        
        layernum_list = [config.num_hidden_layers]
        
        seq_len = config.max_position_embeddings
        hidden_size = config.hidden_size
        if args.shape_order == "SBH":
            layer_shapes_list = [[[seq_len, -1, hidden_size]]]
        else:
            layer_shapes_list = [[[-1, seq_len, hidden_size]]]
            
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        layer_dtypes_list = [[mixed_precision]]

        module_types = ['embed'] + ['bert_enc'] * config.num_hidden_layers + ['norm', 'cls']
        
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)