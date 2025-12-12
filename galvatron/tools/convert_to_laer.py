import os
import torch
import torch.nn.functional as F
import argparse
import gc
import galvatron

embedding_name = "model_embed_tokens.pt"
layer_name = "model_layers_%d.pt"
ln_f_name = "model_norm.pt"
cls_name = "lm_head.pt"

def vocab_range_from_global_vocab_size(num_experts, ep_rank, ep_size):
    numerator = num_experts // ep_size
    index_f = ep_rank * numerator
    index_l = index_f + numerator
    return index_f, index_l

def convert_checkpoint_format(input_dir, output_dir, tp_size, ep_size, num_layers, hidden_size=4096, vocab_size=32000, num_experts=8, num_attention_heads=32, num_key_value_heads=8, head_dim=128, use_cache=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    def to_gpu(tensor):
        """将tensor移到GPU，如果GPU可用"""
        if device.type == "cuda":
            return tensor.to(device)
        return tensor
    
    def to_cpu(tensor):
        """将tensor移回CPU"""
        if device.type == "cuda":
            return tensor.cpu()
        return tensor
    
    def clear_gpu_cache():
        """清理GPU缓存"""
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    # 预加载所有checkpoint到内存，避免重复IO
    if use_cache:
        print("Pre-loading all checkpoints...")
        checkpoint_cache = {}
        for ep_rank in range(ep_size):
            tp_rank = ep_rank % tp_size
            input_rank_dir = os.path.join(input_dir, f"mp_rank_0{tp_rank}_00{ep_rank}")
            checkpoint_path = os.path.join(input_rank_dir, "model_optim_rng.pt")
            if os.path.exists(checkpoint_path):
                # 优化加载参数
                checkpoint = torch.load(
                    checkpoint_path, 
                    map_location="cpu", 
                    mmap=True,
                )
                checkpoint_cache[(tp_rank, ep_rank)] = checkpoint.get("model", checkpoint)
    else:
        checkpoint_cache = {}
    
    def get_model_state(tp_rank, ep_rank):
        """获取模型状态，支持缓存和直接加载两种模式"""
        if use_cache:
            return checkpoint_cache.get((tp_rank, ep_rank))
        else:
            input_rank_dir = os.path.join(input_dir, f"mp_rank_0{tp_rank}_00{ep_rank}")
            checkpoint_path = os.path.join(input_rank_dir, "model_optim_rng.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, 
                    map_location="cpu", 
                    mmap=True,
                )
                return checkpoint.get("model", checkpoint)
            return None
    
    print("Processing embedding...")
    all_embedding_slices = []
    for tp_rank in range(tp_size):
        ep_rank = tp_rank
        model_state = get_model_state(tp_rank, ep_rank)
        if model_state and "embedding.word_embeddings.weight" in model_state:
            embedding_weight = to_gpu(model_state["embedding.word_embeddings.weight"])
            all_embedding_slices.append(embedding_weight)
    
    # GPU上进行拼接
    if all_embedding_slices:
        embedding_data = {"embed_tokens.weight": to_cpu(torch.cat(all_embedding_slices, dim=0))}
        torch.save(embedding_data, os.path.join(output_dir, embedding_name))
        del embedding_data, all_embedding_slices
        clear_gpu_cache()
    print("Embedding saved.")
    
    print("Processing final layernorm...")
    ln_f_weight = None
    for tp_rank in range(tp_size):
        ep_rank = tp_rank
        model_state = get_model_state(tp_rank, ep_rank)
        if model_state and "decoder.final_layernorm.weight" in model_state:
            ln_f_weight = to_gpu(model_state["decoder.final_layernorm.weight"])
            break
    
    if ln_f_weight is not None:
        ln_f_data = {"weight": to_cpu(ln_f_weight)}
        torch.save(ln_f_data, os.path.join(output_dir, ln_f_name))
        del ln_f_data, ln_f_weight
        clear_gpu_cache()
    print("Final layernorm saved.")
    
    print("Processing output layer...")
    all_cls_slices = []
    for tp_rank in range(tp_size):
        ep_rank = tp_rank
        model_state = get_model_state(tp_rank, ep_rank)
        if model_state and "output_layer.weight" in model_state:
            output_weight = to_gpu(model_state["output_layer.weight"])
            all_cls_slices.append(output_weight)
    
    if all_cls_slices:
        cls_data = {"weight": to_cpu(torch.cat(all_cls_slices, dim=0))}
        torch.save(cls_data, os.path.join(output_dir, cls_name))
        del cls_data, all_cls_slices
        clear_gpu_cache()
    print("Output layer saved.")
    
    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}...")
        
        all_qkv_slices = []
        all_qkv_bias_slices = []
        all_proj_slices = []
        ln_qkv_weight = None
        ln_mlp_weight = None
        ln_q_weight = None
        ln_k_weight = None
        
        for tp_rank in range(tp_size):
            # print(f"tp_rank: {tp_rank}")
            ep_rank = tp_rank
            model_state = get_model_state(tp_rank, ep_rank)
            if model_state:
                if f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight" in model_state:
                    qkv_weight = to_gpu(model_state[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"])
                    all_qkv_slices.append(qkv_weight)
                    if f"decoder.layers.{layer_idx}.self_attention.linear_qkv.bias" in model_state:
                        qkv_bias = to_gpu(model_state[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.bias"])
                        all_qkv_bias_slices.append(qkv_bias)
                
                if f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight" in model_state:
                    proj_weight = to_gpu(model_state[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"])
                    all_proj_slices.append(proj_weight)
                
                if f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight" in model_state:
                    ln_qkv_weight = to_gpu(model_state[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight"])

                if f"decoder.layers.{layer_idx}.self_attention.q_layernorm.weight" in model_state:
                    ln_q_weight = to_gpu(model_state[f"decoder.layers.{layer_idx}.self_attention.q_layernorm.weight"])
                
                if f"decoder.layers.{layer_idx}.self_attention.k_layernorm.weight" in model_state:
                    ln_k_weight = to_gpu(model_state[f"decoder.layers.{layer_idx}.self_attention.k_layernorm.weight"])
                
                if f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight" in model_state:
                    ln_mlp_weight = to_gpu(model_state[f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight"])
        

        router_weight = None
        w1 = []
        w2 = []
        w3 = []

        for ep_rank in range(ep_size):
            # print(f"ep_rank: {ep_rank}")
            tp_rank = ep_rank % tp_size
            model_state = get_model_state(tp_rank, ep_rank)
            if model_state:
                if f"decoder.layers.{layer_idx}.mlp.router.weight" in model_state:
                    router_weight = to_gpu(model_state[f"decoder.layers.{layer_idx}.mlp.router.weight"])

                expert_start_index, expert_end_index = vocab_range_from_global_vocab_size(
                    num_experts, ep_rank, ep_size
                )
                
                for expert_idx in range(expert_start_index, expert_end_index):
                    local_expert_idx = expert_idx - expert_start_index
                    
                    if f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{local_expert_idx}.linear_fc1.weight" in model_state:
                        fc1_weight = to_gpu(model_state[f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{local_expert_idx}.linear_fc1.weight"])
                        middle = fc1_weight.shape[0]
                        w1.append(fc1_weight[:middle//2])
                        w3.append(fc1_weight[middle//2:])
                    
                    if f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{local_expert_idx}.linear_fc2.weight" in model_state:
                        fc2_weight = to_gpu(model_state[f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{local_expert_idx}.linear_fc2.weight"])
                        w2.append(fc2_weight)
        
        layer_data = {}
        
        q_weight = []
        k_weight = []
        v_weight = []
        q_bias = []
        k_bias = []
        v_bias = []
        for slices in all_qkv_slices:
            slices = slices.reshape( num_key_value_heads // tp_size, -1 , hidden_size)
            tp1 = head_dim * num_attention_heads // num_key_value_heads
            tp2 = tp1 + head_dim
            q_weight.append(slices[:, :tp1])
            k_weight.append(slices[:, tp1:tp2])
            v_weight.append(slices[:, tp2:])
        
        if len(all_qkv_bias_slices) > 0:
            for slices in all_qkv_bias_slices:
                slices = slices.reshape( num_key_value_heads // tp_size, -1 )
                tp1 = head_dim * num_attention_heads // num_key_value_heads
                tp2 = tp1 + head_dim
                q_bias.append(slices[:, :tp1])
                k_bias.append(slices[:, tp1:tp2])
                v_bias.append(slices[:, tp2:])

        # GPU上进行拼接操作
        layer_data["self_attn.q_proj.weight"] = to_cpu(torch.cat(q_weight, dim=0).reshape(-1, hidden_size))
        layer_data["self_attn.k_proj.weight"] = to_cpu(torch.cat(k_weight, dim=0).reshape(-1, hidden_size))
        layer_data["self_attn.v_proj.weight"] = to_cpu(torch.cat(v_weight, dim=0).reshape(-1, hidden_size))

        if len(all_qkv_bias_slices) > 0:
            layer_data["self_attn.q_proj.bias"] = to_cpu(torch.cat(q_bias, dim=0).reshape(-1))
            layer_data["self_attn.k_proj.bias"] = to_cpu(torch.cat(k_bias, dim=0).reshape(-1))
            layer_data["self_attn.v_proj.bias"] = to_cpu(torch.cat(v_bias, dim=0).reshape(-1))

        layer_data["self_attn.o_proj.weight"] = to_cpu(torch.cat(all_proj_slices, dim=1))
        
        layer_data["input_layernorm.weight"] = to_cpu(ln_qkv_weight) if ln_qkv_weight is not None else None
        layer_data["self_attn.q_layernorm.weight"] = to_cpu(ln_q_weight) if ln_q_weight is not None else None
        layer_data["self_attn.k_layernorm.weight"] = to_cpu(ln_k_weight) if ln_k_weight is not None else None
        
        layer_data["post_attention_layernorm.weight"] = to_cpu(ln_mlp_weight) if ln_mlp_weight is not None else None
        
        layer_data["block_sparse_moe.gate.weight"] = to_cpu(router_weight) if router_weight is not None else None
        
        # 添加experts
        for key, w in enumerate(zip(w1,w2,w3)):
            w1, w2, w3 = w
            expert_num = key
            layer_data[f"block_sparse_moe.experts.{expert_num}.w1.weight"] = to_cpu(w1)
            layer_data[f"block_sparse_moe.experts.{expert_num}.w3.weight"] = to_cpu(w3)
            layer_data[f"block_sparse_moe.experts.{expert_num}.w2.weight"] = to_cpu(w2)
        
        torch.save(layer_data, os.path.join(output_dir, layer_name % layer_idx))
        del layer_data, all_qkv_slices, all_proj_slices, w1, w2, w3
        clear_gpu_cache()
        print(f"Layer {layer_idx} saved.")

    # 清理缓存
    del checkpoint_cache
    print(f"Checkpoint conversion completed. Output saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint format for MoE model")
    parser.add_argument("--input_dir", type=str, required=True, help="Input checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output checkpoint directory")
    parser.add_argument("--tp_size", type=int, required=True, help="Tensor parallel size")
    parser.add_argument("--ep_size", type=int, required=True, help="Expert parallel size")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of layers")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden size")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--num_attention_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=8, help="Number of key-value heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    
    args = parser.parse_args()
    
    convert_checkpoint_format(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        num_experts=args.num_experts,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim
    )
if __name__ == "__main__":
    main() 