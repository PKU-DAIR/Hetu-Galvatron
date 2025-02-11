import torch
from einops import rearrange
import os
from galvatron.core.arguments import get_args
import torch.nn.functional as F
from megatron.core.tensor_parallel.utils import VocabUtility
import torch.distributed as dist

# 与checkpoint转换文件保持一致的文件名格式
embedding_name = "bert_embeddings.pt"
layer_name = "bert_encoder_layer_%d.pt"
pooler_name = "bert_pooler.pt"
cls_name = "cls_predictions.pt"  # MLM头的文件名

@torch.no_grad()
def load_bert_module(load, tp_groups, name, submodule, module):
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)
    args = get_args()

    # 处理embeddings层
    if name.endswith("word_embeddings"):
        file_path = os.path.join(load, embedding_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        vocab_size = checkpoint["word_embeddings.weight"].shape[0]
        padding_size = args.padded_vocab_size - vocab_size
        padded_weight = F.pad(
            checkpoint["word_embeddings.weight"].to(device="cuda", dtype=torch.float32),
            (0, 0, padding_size, 0),
            mode='constant',
            value=0
        )
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.padded_vocab_size, rank, world_size
        )
        submodule.weight.copy_(padded_weight[vocab_start_index:vocab_end_index])
    
    elif name.endswith("position_embeddings"):
        file_path = os.path.join(load, embedding_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        weight = checkpoint["position_embeddings.weight"].to(device="cuda", dtype=torch.float32)
        submodule.weight.copy_(weight)
    
    elif name.endswith("token_type_embeddings"):
        file_path = os.path.join(load, embedding_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        weight = checkpoint["token_type_embeddings.weight"].to(device="cuda", dtype=torch.float32)
        submodule.weight.copy_(weight)
    
    # 处理LayerNorm
    elif name.endswith("LayerNorm"):
        if "attention" in name:
            file_path = os.path.join(load, layer_name % module.idx)
            checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
            weight = checkpoint["attention.output.LayerNorm.weight"].to(device="cuda", dtype=torch.float32)
            bias = checkpoint["attention.output.LayerNorm.bias"].to(device="cuda", dtype=torch.float32)
        elif "mlp" in name:
            file_path = os.path.join(load, layer_name % module.idx)
            checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
            weight = checkpoint["output.LayerNorm.weight"].to(device="cuda", dtype=torch.float32)
            bias = checkpoint["output.LayerNorm.bias"].to(device="cuda", dtype=torch.float32)
        else:  # embedding层的LayerNorm
            file_path = os.path.join(load, embedding_name)
            checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
            weight = checkpoint["LayerNorm.weight"].to(device="cuda", dtype=torch.float32)
            bias = checkpoint["LayerNorm.bias"].to(device="cuda", dtype=torch.float32)
        submodule.weight.copy_(weight)
        submodule.bias.copy_(bias)
    
    # 处理MLM头
    elif name.endswith("cls"):
        file_path = os.path.join(load, cls_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        
        if hasattr(submodule, "predictions"):
            # Transform层
            transform_weight = checkpoint["transform.dense.weight"].to(device="cuda", dtype=torch.float32)
            transform_bias = checkpoint["transform.dense.bias"].to(device="cuda", dtype=torch.float32)
            transform_ln_weight = checkpoint["transform.LayerNorm.weight"].to(device="cuda", dtype=torch.float32)
            transform_ln_bias = checkpoint["transform.LayerNorm.bias"].to(device="cuda", dtype=torch.float32)
            
            submodule.predictions.transform.dense.weight.copy_(transform_weight)
            submodule.predictions.transform.dense.bias.copy_(transform_bias)
            submodule.predictions.transform.LayerNorm.weight.copy_(transform_ln_weight)
            submodule.predictions.transform.LayerNorm.bias.copy_(transform_ln_bias)
            
            # Decoder层
            vocab_size = checkpoint["decoder.weight"].shape[0]
            padding_size = args.padded_vocab_size - vocab_size
            padded_weight = F.pad(
                checkpoint["decoder.weight"].to(device="cuda", dtype=torch.float32),
                (0, 0, padding_size, 0),
                mode='constant',
                value=0
            )
            vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                args.padded_vocab_size, rank, world_size
            )
            submodule.predictions.decoder.weight.copy_(padded_weight[vocab_start_index:vocab_end_index])
            
            if hasattr(submodule.predictions.decoder, "bias"):
                decoder_bias = checkpoint["decoder.bias"].to(device="cuda", dtype=torch.float32)
                submodule.predictions.decoder.bias.copy_(decoder_bias)
    
    # 处理Transformer层
    else:
        file_path = os.path.join(load, layer_name % module.idx)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        
        if name.startswith("attention"):
            if name.endswith("query_key_value"):
                # 合并QKV权重和偏置
                q_weight = checkpoint["attention.self.query.weight"].to(device="cuda", dtype=torch.float32)
                k_weight = checkpoint["attention.self.key.weight"].to(device="cuda", dtype=torch.float32)
                v_weight = checkpoint["attention.self.value.weight"].to(device="cuda", dtype=torch.float32)
                q_bias = checkpoint["attention.self.query.bias"].to(device="cuda", dtype=torch.float32)
                k_bias = checkpoint["attention.self.key.bias"].to(device="cuda", dtype=torch.float32)
                v_bias = checkpoint["attention.self.value.bias"].to(device="cuda", dtype=torch.float32)
                
                weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    bias.shape[0], rank, world_size
                )
                submodule.weight.copy_(weight[weight_start_index:weight_end_index])
                submodule.bias.copy_(bias[weight_start_index:weight_end_index])
            elif name.endswith("dense"):
                weight = checkpoint["attention.output.dense.weight"].to(device="cuda", dtype=torch.float32)
                bias = checkpoint["attention.output.dense.bias"].to(device="cuda", dtype=torch.float32)
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[0], rank, world_size
                )
                submodule.weight.copy_(weight[weight_start_index:weight_end_index].t())
                submodule.bias.copy_(bias)
                
        elif name.startswith("mlp"):
            if name.endswith("dense_h_to_4h"):
                weight = checkpoint["intermediate.dense.weight"].to(device="cuda", dtype=torch.float32)
                bias = checkpoint["intermediate.dense.bias"].to(device="cuda", dtype=torch.float32)
                weight = weight.t()
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[0], rank, world_size
                )
                submodule.weight.copy_(weight[weight_start_index:weight_end_index])
                submodule.bias.copy_(bias[weight_start_index:weight_end_index])
            elif name.endswith("dense_4h_to_h"):
                weight = checkpoint["output.dense.weight"].to(device="cuda", dtype=torch.float32)
                bias = checkpoint["output.dense.bias"].to(device="cuda", dtype=torch.float32)
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[0], rank, world_size
                )
                submodule.weight.copy_(weight[weight_start_index:weight_end_index].t())
                submodule.bias.copy_(bias)