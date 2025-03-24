import torch
import torch.distributed
import torch.nn as nn
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.mappings_group import get_tensor_model_parallel_world_size_group
from megatron.core.tensor_parallel.utils import VocabUtility

from galvatron.core import get_args
from galvatron.core.runtime import ModelInfo, mixed_precision_dtype
from galvatron.core.runtime.pipeline import PipeSequential
from galvatron.core.runtime.tensor_parallel import colummn_row_reset_parameters
import einops

class ViTPatchEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_embeddings = model.vit.embeddings.patch_embeddings
        
    def forward(self, pixel_values):
        embeddings, bias = self.patch_embeddings(pixel_values)
        if bias is not None:
            embeddings = embeddings + bias
        return embeddings

class ViTPositionEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.position_embeddings = model.vit.embeddings.position_embeddings
        
    def forward(self, embeddings):
        return embeddings + self.position_embeddings

class ViTEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.vit.config
        self.patch_size = self.config.patch_size
        
        self.patch_embeddings = ViTPatchEmbeddings_(model)
        self.position_embeddings = ViTPositionEmbeddings_(model)
        self.cls_token = model.vit.embeddings.cls_token
        
        args = get_args()
        self.hidden_dropout = torch.nn.Dropout(args.hidden_dropout)
        
        self.tp_group = getattr(model.vit.embeddings, 'tp_group', None)
        self.sp_group = getattr(model.vit.embeddings, 'sp_group', None)

    def forward(self, pixel_values, labels=None, position_ids=None, attention_mask=None):
        batch_size = pixel_values.shape[0]

        patches = einops.rearrange(
            pixel_values,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        
        embeddings = self.patch_embeddings(patches)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        embeddings = self.position_embeddings(embeddings)
         # [seq_len, batch, hidden]
        embeddings = embeddings.transpose(0, 1).contiguous()

        embeddings = self.hidden_dropout(embeddings)
        return embeddings

class ViTLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        self.layer = model.vit.encoder.layer[layer_idx]

    def forward(self, hidden_states, labels=None, position_ids=None, attention_mask=None):
        hidden_states = self.layer(hidden_states, attention_mask)
        return hidden_states

class ViTPreNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.LayerNorm = model.vit.layernorm
        
    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None):
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class ViTLoss_(nn.Module):
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
            tp_group=self.tp_group,
        )
        return logits_parallel

    def reset_parameters(self):
        # Implementation of reset_parameters method
        pass

class ViTCls_(nn.Module):
    def __init__(self, model, parallel_loss=True, half_entropy=True):
        super().__init__()
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        
        self.tp_group = model.classifier.tp_group
        self.sp_group = model.classifier.sp_group
        
        self.pooler = model.vit.pooler
        
        self.lm_head = ViTLoss_(
            model.classifier.weight, 
            self.sequence_parallel, 
            self.tp_group
        )
        
        self.parallel_loss = parallel_loss
        self.half_entropy = half_entropy
        self.vocab_sp = False 
        if args.entropy_in_fp32:
            self.half_entropy = False
    
    def forward(self, hidden_states, labels=None, position_ids=None, attention_mask=None):
        hidden_states_transformed = hidden_states.permute(1, 0, 2).contiguous()
    
        pooled_output = self.pooler(hidden_states_transformed)
        
        pooled_output = pooled_output.unsqueeze(0)
        
        logits_parallel = self.lm_head(pooled_output)
        
        if labels is None:
            return logits_parallel
        
        if labels.dim() == 1:  # [batch_size]
            labels = labels.unsqueeze(0).contiguous()
        elif labels.dim() > 1:
            labels = labels.transpose(0, 1).contiguous()
        
        if not self.parallel_loss:
            output = tensor_parallel.gather_from_tensor_model_parallel_region_group(
                logits_parallel, self.tp_group
            )
            logits = output.float() if not self.half_entropy else output
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = logits.view(-1, logits.size(-1))
            
            shift_labels = labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss_input = logits_parallel.float() if not self.half_entropy else logits_parallel
            loss = tensor_parallel.vocab_parallel_cross_entropy(
                loss_input, labels, tp_group=self.tp_group
            )
        
        if loss.dim() == 2:
            loss = loss.transpose(0, 1).contiguous()
        elif loss.dim() == 1:
            loss = loss.unsqueeze(1)
        
        return loss



def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('embeddings', ViTEmbeddings_(model))
    
    for i in range(config.num_hidden_layers):
        enc = ViTLayers_(model, i)
        model_.add_module(f'layer_{i}', enc)
    
    model_.add_module('prenorm', ViTPreNorm_(model))
    
    model_.add_module('cls', ViTCls_(model))
    
    ViTLoss_.reset_parameters = colummn_row_reset_parameters
    
    return model_

class ViTModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(ViTModelInfo, self).__init__()
        layernum_list = [config.num_hidden_layers]
        seq_len = (config.image_size // config.patch_size) ** 2 + 1  # +1 for CLS token
        hidden_size = config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        
        if args.shape_order == "SBH":
            layer_shapes_list = [[[seq_len, -1, hidden_size]]]
        else:
            layer_shapes_list = [[[-1, seq_len, hidden_size]]]
            
        layer_dtypes_list = [[mixed_precision]]
        
        module_types = ['embed'] + ['vit_enc'] * config.num_hidden_layers + ['prenorm', 'cls']
        
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)