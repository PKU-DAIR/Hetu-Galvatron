import torch
from einops import rearrange
import os
from galvatron.core.arguments import get_args
import torch.nn.functional as F
from megatron.core.tensor_parallel.utils import VocabUtility
import torch.distributed as dist
import json
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from .ViTModel_sequential import ViTEmbeddings_, ViTPreNorm_, ViTCls_, ViTPatchEmbeddings_, ViTPositionEmbeddings_
from .ViTModel_tensor_parallel import ViTLayer_tp

patch_embedding_name = "model_patch_embeddings.pt"
position_embedding_name = "model_position_embeddings.pt"
embedding_name = "model_embeddings.pt"
layer_name = "model_layers_%d.pt"
ln_f_name = "model_norm.pt"
pooler_name = "model_pooler.pt"
cls_name = "model_classifier.pt"

@torch.no_grad()
def save_vit_module(save_path, tp_groups, name, submodule, module):
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)
    
    if rank != 0:
        return
    
    os.makedirs(save_path, exist_ok=True)
    
    if "patch_embeddings.projection" in name or "position_embeddings" in name or "cls_token" in name:
        checkpoint = {}
        if not os.path.exists(os.path.join(save_path, embeddings_name)):
            checkpoint = {}
        else:
            checkpoint = torch.load(os.path.join(save_path, embeddings_name))
        
        if "patch_embeddings.projection" in name:
            checkpoint["patch_embeddings.projection.weight"] = submodule.weight.data.cpu()
            if hasattr(submodule, "bias") and submodule.bias is not None:
                checkpoint["patch_embeddings.projection.bias"] = submodule.bias.data.cpu()
        elif "position_embeddings" in name:
            checkpoint["position_embeddings.weight"] = submodule.weight.data.cpu()
        elif "cls_token" in name:
            checkpoint["cls_token"] = submodule.data.cpu()
        
        torch.save(checkpoint, os.path.join(save_path, embeddings_name))
    
    elif name == "LayerNorm" and hasattr(module, '__class__') and module.__class__.__name__ == "ViTPreNorm_":
        checkpoint = {}
        if os.path.exists(os.path.join(save_path, layernorm_name)):
            checkpoint = torch.load(os.path.join(save_path, layernorm_name))
        
        checkpoint["layernorm.weight"] = submodule.weight.data.cpu()
        checkpoint["layernorm.bias"] = submodule.bias.data.cpu()
        
        torch.save(checkpoint, os.path.join(save_path, layernorm_name))
    
    elif name.endswith("LayerNorm"):
        if "attention" in name:
            checkpoint = {}
            if os.path.exists(os.path.join(save_path, layer_name % module.idx)):
                checkpoint = torch.load(os.path.join(save_path, layer_name % module.idx))
            checkpoint["attention.layernorm.weight"] = submodule.weight.data.cpu()
            checkpoint["attention.layernorm.bias"] = submodule.bias.data.cpu()
            torch.save(checkpoint, os.path.join(save_path, layer_name % module.idx))
        elif "mlp" in name or "ffn" in name:
            checkpoint = {}
            if os.path.exists(os.path.join(save_path, layer_name % module.idx)):
                checkpoint = torch.load(os.path.join(save_path, layer_name % module.idx))
            checkpoint["ffn.layernorm.weight"] = submodule.weight.data.cpu()
            checkpoint["ffn.layernorm.bias"] = submodule.bias.data.cpu()
            torch.save(checkpoint, os.path.join(save_path, layer_name % module.idx))
        else:
            checkpoint = {}
            if os.path.exists(os.path.join(save_path, embeddings_name)):
                checkpoint = torch.load(os.path.join(save_path, embeddings_name))
            checkpoint["layernorm.weight"] = submodule.weight.data.cpu()
            checkpoint["layernorm.bias"] = submodule.bias.data.cpu()
            torch.save(checkpoint, os.path.join(save_path, embeddings_name))
    
    elif name.startswith("attention"):
        checkpoint = {}
        if os.path.exists(os.path.join(save_path, layer_name % module.idx)):
            checkpoint = torch.load(os.path.join(save_path, layer_name % module.idx))
        
        if name.endswith("query_key_value") or name.endswith("qkv"):
            gathered_weight = [torch.zeros_like(submodule.weight.data) for _ in range(world_size)]
            gathered_bias = [torch.zeros_like(submodule.bias.data) for _ in range(world_size)]
            
            dist.all_gather(gathered_weight, submodule.weight.data, group=tp_groups)
            dist.all_gather(gathered_bias, submodule.bias.data, group=tp_groups)
            
            if rank == 0:
                full_weight = torch.cat(gathered_weight, dim=0)
                full_bias = torch.cat(gathered_bias, dim=0)
                
                hidden_size = full_weight.shape[1]
                head_size = hidden_size // module.num_attention_heads
                num_heads = module.num_attention_heads
                
                q_weight = full_weight[:hidden_size]
                k_weight = full_weight[hidden_size:2*hidden_size]
                v_weight = full_weight[2*hidden_size:]
                
                q_bias = full_bias[:hidden_size]
                k_bias = full_bias[hidden_size:2*hidden_size]
                v_bias = full_bias[2*hidden_size:]
                
                checkpoint["attention.self.query.weight"] = q_weight.cpu()
                checkpoint["attention.self.key.weight"] = k_weight.cpu()
                checkpoint["attention.self.value.weight"] = v_weight.cpu()
                checkpoint["attention.self.query.bias"] = q_bias.cpu()
                checkpoint["attention.self.key.bias"] = k_bias.cpu()
                checkpoint["attention.self.value.bias"] = v_bias.cpu()
        
        elif name.endswith("dense") or name.endswith("output"):
            gathered_weight = [torch.zeros_like(submodule.weight.data) for _ in range(world_size)]
            
            dist.all_gather(gathered_weight, submodule.weight.data, group=tp_groups)
            
            if rank == 0:
                full_weight = torch.cat(gathered_weight, dim=0)
                checkpoint["attention.output.dense.weight"] = full_weight.t().cpu()
                checkpoint["attention.output.dense.bias"] = submodule.bias.data.cpu()
        
        torch.save(checkpoint, os.path.join(save_path, layer_name % module.idx))
    
    elif name.startswith("mlp") or name.startswith("ffn"):
        checkpoint = {}
        if os.path.exists(os.path.join(save_path, layer_name % module.idx)):
            checkpoint = torch.load(os.path.join(save_path, layer_name % module.idx))
        
        if name.endswith("dense_h_to_4h") or name.endswith("fc1"):
            gathered_weight = [torch.zeros_like(submodule.weight.data) for _ in range(world_size)]
            gathered_bias = [torch.zeros_like(submodule.bias.data) for _ in range(world_size)]
            
            dist.all_gather(gathered_weight, submodule.weight.data, group=tp_groups)
            dist.all_gather(gathered_bias, submodule.bias.data, group=tp_groups)
            
            if rank == 0:
                full_weight = torch.cat(gathered_weight, dim=0)
                full_bias = torch.cat(gathered_bias, dim=0)
                checkpoint["ffn.fc1.weight"] = full_weight.cpu()
                checkpoint["ffn.fc1.bias"] = full_bias.cpu()
        
        elif name.endswith("dense_4h_to_h") or name.endswith("fc2"):
            gathered_weight = [torch.zeros_like(submodule.weight.data) for _ in range(world_size)]
            
            dist.all_gather(gathered_weight, submodule.weight.data, group=tp_groups)
            
            if rank == 0:
                full_weight = torch.cat(gathered_weight, dim=0)
                checkpoint["ffn.fc2.weight"] = full_weight.t().cpu()
                checkpoint["ffn.fc2.bias"] = submodule.bias.data.cpu()
        
        torch.save(checkpoint, os.path.join(save_path, layer_name % module.idx))
    
    elif name.endswith("classifier") or name.endswith("head") or name == "lm_head.weight":
        checkpoint = {}
        if os.path.exists(os.path.join(save_path, classifier_name)):
            checkpoint = torch.load(os.path.join(save_path, classifier_name))
        
        if name == "lm_head.weight":
            gathered_weight = [torch.zeros_like(submodule.data) for _ in range(world_size)]
            dist.all_gather(gathered_weight, submodule.data, group=tp_groups)
            
            if rank == 0:
                full_weight = torch.cat(gathered_weight, dim=0)
                checkpoint["classifier.weight"] = full_weight.cpu()
        else:
            checkpoint["classifier.weight"] = submodule.weight.data.cpu()
            if hasattr(submodule, "bias") and submodule.bias is not None:
                checkpoint["classifier.bias"] = submodule.bias.data.cpu()
        
        torch.save(checkpoint, os.path.join(save_path, classifier_name))
    
    elif name.endswith("pooler.dense"):
        checkpoint = {}
        if os.path.exists(os.path.join(save_path, pooler_name)):
            checkpoint = torch.load(os.path.join(save_path, pooler_name))
        
        gathered_weight = [torch.zeros_like(submodule.weight.data) for _ in range(world_size)]
        
        dist.all_gather(gathered_weight, submodule.weight.data, group=tp_groups)
        
        if rank == 0:
            full_weight = torch.cat(gathered_weight, dim=1)
            checkpoint["pooler.dense.weight"] = full_weight.cpu()
            checkpoint["pooler.dense.bias"] = submodule.bias.data.cpu()
        
        torch.save(checkpoint, os.path.join(save_path, pooler_name))

    elif name.endswith("cls.pooler.dense"):
        checkpoint = {}
        if os.path.exists(os.path.join(save_path, pooler_name)):
            checkpoint = torch.load(os.path.join(save_path, pooler_name))
        
        gathered_weight = [torch.zeros_like(submodule.weight.data) for _ in range(world_size)]
        
        dist.all_gather(gathered_weight, submodule.weight.data, group=tp_groups)
        
        if rank == 0:
            full_weight = torch.cat(gathered_weight, dim=1)
            checkpoint["pooler.dense.weight"] = full_weight.cpu()
            checkpoint["pooler.dense.bias"] = submodule.bias.data.cpu()
        
        torch.save(checkpoint, os.path.join(save_path, pooler_name))

@torch.no_grad()
def load_vit_module(load, tp_groups, name, submodule, module, distributed_checkpoint):
    """Load ViT model module"""
    if distributed_checkpoint:
        load_distributed_checkpoint(load, tp_groups, name, submodule, module)
    else:
        load_hf_checkpoint(load, tp_groups, name, submodule, module)

@torch.no_grad()
def save_vit_module(save_path, model, optimizer, opt_param_scheduler, iter_num, args):
    """Save model parameters by layer"""
    rank = torch.distributed.get_rank()

    if rank == 0:
        print("Starting to save checkpoint")
        os.makedirs(save_path, exist_ok=True)
        assert hasattr(model, "hybrid_parallel_configs")
        json.dump(model.hybrid_parallel_configs, open(os.path.join(save_path, "hybrid_parallel_configs.json"), "w"))

        os.makedirs(os.path.join(save_path, "iter_%d" % iter_num), exist_ok=True)
        opt_param_scheduler_state_dict = opt_param_scheduler.state_dict()
        json.dump(
            opt_param_scheduler_state_dict,
            open(os.path.join(save_path, "iter_%d" % iter_num, f"opt_param_scheduler.json"), "w"),
        )

    assert args.default_dp_type != "ddp", "Saving/loading distributed checkpoint does not support DDP"
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        save_path = os.path.join(save_path, "iter_%d" % iter_num)
        idx = 0
        for block in model.model.model_cur_stage:
            for m in block.modules():
                if isinstance(m, FSDP):
                    wrapped_module = m._fsdp_wrapped_module
                    if isinstance(wrapped_module, CheckpointWrapper):
                        wrapped_module = wrapped_module._checkpoint_wrapped_module
                    dp_rank = torch.distributed.get_rank(model.sdp_groups_whole[idx].group)
                    tp_rank = torch.distributed.get_rank(model.tp_groups_whole[idx].group)
                    state_dict = m.state_dict()
                    if dp_rank == 0:
                        if isinstance(wrapped_module, ViTPatchEmbeddings_):
                            os.makedirs(os.path.join(save_path, f"{patch_embedding_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{patch_embedding_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, ViTPositionEmbeddings_):
                            os.makedirs(os.path.join(save_path, f"{position_embedding_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{position_embedding_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, ViTEmbeddings_):
                            os.makedirs(os.path.join(save_path, f"{embedding_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{embedding_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, ViTPreNorm_):
                            os.makedirs(os.path.join(save_path, f"{ln_f_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{ln_f_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, ViTCls_):
                            os.makedirs(os.path.join(save_path, f"{cls_name[:-3]}"), exist_ok=True)
                            torch.save(state_dict, os.path.join(save_path, f"{cls_name[:-3]}/{tp_rank}.pt"))
                            
                            if hasattr(wrapped_module, 'pooler') and wrapped_module.pooler is not None:
                                pooler_state = {}
                                for key, value in state_dict.items():
                                    if 'pooler' in key:
                                        pooler_state[key] = value
                                
                                if pooler_state:
                                    os.makedirs(os.path.join(save_path, f"{pooler_name[:-3]}"), exist_ok=True)
                                    torch.save(pooler_state, os.path.join(save_path, f"{pooler_name[:-3]}/{tp_rank}.pt"))
                        elif isinstance(wrapped_module, ViTLayer_tp):
                            os.makedirs(
                                os.path.join(save_path, f"{(layer_name%wrapped_module.idx)[:-3]}"), exist_ok=True
                            )
                            torch.save(
                                state_dict,
                                os.path.join(save_path, f"{(layer_name%wrapped_module.idx)[:-3]}/{tp_rank}.pt"),
                            )
            idx += 1

    
    optimizer_state_dict = optimizer.state_dict()
    os.makedirs(os.path.join(save_path, f"optimizer"), exist_ok=True)
    torch.save(optimizer_state_dict, os.path.join(save_path, f"optimizer/{rank}.pt"))

    torch.distributed.barrier()
    if rank == 0:
        print("Checkpoint saving completed")

@torch.no_grad()
def load_distributed_checkpoint(load_path, model, args):
    """Load model from distributed checkpoint"""
    rank = torch.distributed.get_rank()
    
    if rank == 0:
        print("Starting to load checkpoint")
    
    for block in model.model.model_cur_stage:
        for m in block.modules():
            if isinstance(m, FSDP):
                wrapped_module = m._fsdp_wrapped_module
                if isinstance(wrapped_module, CheckpointWrapper):
                    wrapped_module = wrapped_module._checkpoint_wrapped_module
                
                if isinstance(wrapped_module, ViTCls_) and hasattr(wrapped_module, 'pooler') and wrapped_module.pooler is not None:
                    try:
                        pooler_path = os.path.join(load_path, f"{pooler_name[:-3]}")
                        if os.path.exists(pooler_path):
                            tp_rank = torch.distributed.get_rank(model.tp_groups_whole[idx].group)
                            pooler_state = torch.load(os.path.join(pooler_path, f"{tp_rank}.pt"))
                            
                        
                            pooler_dict = {}
                            for key, value in pooler_state.items():
                                if 'pooler' in key:
                                    pooler_dict[key] = value
                            
                            
                            if pooler_dict:
                                wrapped_module.pooler.load_state_dict(pooler_dict)
                    except Exception as e:
                        if rank == 0:
                            print(f"Warning: Failed to load pooler state: {e}")
    
    if rank == 0:
        print("Checkpoint loading completed")