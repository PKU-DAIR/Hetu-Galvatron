import torch
import os
import argparse
from collections import defaultdict
from transformers import LlamaForCausalLM, BertForMaskedLM
from galvatron.models.llama_hf.meta_configs.config_utils import config_from_meta
import torch.nn.functional as F
from megatron.core.tensor_parallel.utils import VocabUtility
import torch.distributed as dist

def convert_checkpoints_llama(input_checkpoint_path, output_dir, load_iteration, model_config):
    """Convert Galvatron checkpoint to HuggingFace format"""
    config = config_from_meta(model_config)
    llama_model = LlamaForCausalLM(config)

    iter_dir = os.path.join(input_checkpoint_path, f"iter_{load_iteration}")
    
    embed_dir = os.path.join(iter_dir, "model_embed_tokens")
    assert os.path.exists(embed_dir), f"Embedding directory {embed_dir} does not exist"
    weights = []
    for rank_file in sorted(os.listdir(embed_dir)):
        checkpoint = torch.load(os.path.join(embed_dir, rank_file), map_location='cpu')
        weights.append(checkpoint["embed_tokens.weight"])
    weights = torch.cat(weights, dim=0)
    if weights.shape[0] > config.vocab_size:
        weights = weights[:config.vocab_size].contiguous()
    llama_model.model.embed_tokens.weight.data.copy_(weights)

    for layer_idx in range(config.num_hidden_layers):
        layer_dir = os.path.join(iter_dir, f"model_layers_{layer_idx}")
        assert os.path.exists(layer_dir), f"Layer directory {layer_dir} does not exist"
        q_weights = []
        k_weights = []
        v_weights = []
        o_weights = []
        gate_weights = []
        up_weights = []
        down_weights = []
        
        tp_size = len(os.listdir(layer_dir))
        for rank_file in sorted(os.listdir(layer_dir)):
            checkpoint = torch.load(os.path.join(layer_dir, rank_file), map_location='cpu')

            qkv_weight = checkpoint["attention.attention.query_key_value.weight"]
            head_dim = config.hidden_size // config.num_attention_heads
            nh = config.num_attention_heads // tp_size
            ng = config.num_key_value_heads // tp_size
            dim = head_dim
            qkv_weight = qkv_weight.reshape((ng, -1, config.hidden_size))
            
            q = qkv_weight[:, :dim*nh//ng, :].reshape(-1, config.hidden_size)
            k = qkv_weight[:, dim*nh//ng:dim*(nh//ng+1), :].reshape(-1, config.hidden_size)
            v = qkv_weight[:, dim*(nh//ng+1):, :].reshape(-1, config.hidden_size)
            
            q_weights.append(q)
            k_weights.append(k)
            v_weights.append(v)

            o_weights.append(checkpoint["attention.attention.dense.weight"])

            mlp_weight = checkpoint["mlp.mlp.dense_h_to_4h.weight"]
            gate_size = mlp_weight.shape[0] // 2
            gate_weights.append(mlp_weight[:gate_size])
            up_weights.append(mlp_weight[gate_size:])
            down_weights.append(checkpoint["mlp.mlp.dense_4h_to_h.weight"])

            llama_model.model.layers[layer_idx].input_layernorm.weight.data.copy_(
                checkpoint["attention.LayerNorm.weight"]
            )
            llama_model.model.layers[layer_idx].post_attention_layernorm.weight.data.copy_(
                checkpoint["mlp.LayerNorm.weight"]
            )
        
        q_weights = [q.contiguous() for q in q_weights]
        k_weights = [k.contiguous() for k in k_weights]
        v_weights = [v.contiguous() for v in v_weights]
        o_weights = [o.contiguous() for o in o_weights]
        gate_weights = [g.contiguous() for g in gate_weights]
        up_weights = [u.contiguous() for u in up_weights]
        down_weights = [d.contiguous() for d in down_weights]

        layer = llama_model.model.layers[layer_idx]
        layer.self_attn.q_proj.weight.data.copy_(torch.cat(q_weights, dim=0).contiguous())
        layer.self_attn.k_proj.weight.data.copy_(torch.cat(k_weights, dim=0).contiguous())
        layer.self_attn.v_proj.weight.data.copy_(torch.cat(v_weights, dim=0).contiguous())
        layer.self_attn.o_proj.weight.data.copy_(torch.cat(o_weights, dim=1).contiguous())
        layer.mlp.gate_proj.weight.data.copy_(torch.cat(gate_weights, dim=0).contiguous())
        layer.mlp.up_proj.weight.data.copy_(torch.cat(up_weights, dim=0).contiguous())
        layer.mlp.down_proj.weight.data.copy_(torch.cat(down_weights, dim=1).contiguous())
            
    norm_dir = os.path.join(iter_dir, "model_norm")
    assert os.path.exists(norm_dir), f"Norm directory {norm_dir} does not exist"
    checkpoint = torch.load(os.path.join(norm_dir, "0.pt"), map_location='cpu')
    llama_model.model.norm.weight.data.copy_(checkpoint["norm.weight"])

    lm_head_dir = os.path.join(iter_dir, "lm_head")
    assert os.path.exists(lm_head_dir), f"LM head directory {lm_head_dir} does not exist"
    weights = []
    for rank_file in sorted(os.listdir(lm_head_dir)):
        checkpoint = torch.load(os.path.join(lm_head_dir, rank_file), map_location='cpu')
        weights.append(checkpoint["lm_head.weight"])
    weights = torch.cat(weights, dim=0)
    if weights.shape[0] > config.vocab_size:
        weights = weights[:config.vocab_size].contiguous()
    llama_model.lm_head.weight.data.copy_(weights)

    os.makedirs(output_dir, exist_ok=True)
    llama_model.save_pretrained(output_dir)
    print(f"Successfully converted checkpoint to HuggingFace format at {output_dir}")

def convert_checkpoints_bert_mlm(input_checkpoint_path, output_dir, load_iteration, model_config):
    config = config_from_meta(model_config)
    model = BertForMaskedLM(config)
    
    iter_dir = os.path.join(input_checkpoint_path, f"iter_{load_iteration}")

    embed_dir = os.path.join(iter_dir, "model_embed_tokens")
    assert os.path.exists(embed_dir), f"Embedding directory {embed_dir} does not exist"
    
    weights = []
    for rank_file in sorted(os.listdir(embed_dir)):
        checkpoint = torch.load(os.path.join(embed_dir, rank_file), map_location='cpu')
        weights.append(checkpoint["word_embeddings.weight"])
    weights = torch.cat(weights, dim=0)
    if weights.shape[0] > config.vocab_size:
        weights = weights[:config.vocab_size].contiguous()
    model.bert.embeddings.word_embeddings.weight.data.copy_(weights)
    
    pos_embed_file = os.path.join(embed_dir, "0.pt")  
    checkpoint = torch.load(pos_embed_file, map_location='cpu')
    model.bert.embeddings.position_embeddings.weight.data.copy_(
        checkpoint["position_embeddings.weight"]
    )
    
    model.bert.embeddings.token_type_embeddings.weight.data.copy_(
        checkpoint["token_type_embeddings.weight"]
    )
    
    model.bert.embeddings.LayerNorm.weight.data.copy_(
        checkpoint["LayerNorm.weight"]
    )
    model.bert.embeddings.LayerNorm.bias.data.copy_(
        checkpoint["LayerNorm.bias"]
    )
    
    for layer_idx in range(config.num_hidden_layers):
        layer_dir = os.path.join(iter_dir, f"model_layers_{layer_idx}")
        assert os.path.exists(layer_dir), f"Layer directory {layer_dir} does not exist"
        
        q_weights, k_weights, v_weights = [], [], []
        q_bias, k_bias, v_bias = [], [], []
        o_weights, o_bias = [], []
        intermediate_weights, intermediate_bias = [], []
        output_weights, output_bias = [], []
        
        tp_size = len(os.listdir(layer_dir))
        for rank_file in sorted(os.listdir(layer_dir)):
            checkpoint = torch.load(os.path.join(layer_dir, rank_file), map_location='cpu')
            
            qkv_weight = checkpoint["attention.self.query_key_value.weight"]
            qkv_bias = checkpoint["attention.self.query_key_value.bias"]
            
            hidden_size = config.hidden_size
            attention_head_size = hidden_size // config.num_attention_heads
            nh = config.num_attention_heads // tp_size
            
            q = qkv_weight[:hidden_size]
            k = qkv_weight[hidden_size:2*hidden_size]
            v = qkv_weight[2*hidden_size:]
            
            q_b = qkv_bias[:hidden_size]
            k_b = qkv_bias[hidden_size:2*hidden_size]
            v_b = qkv_bias[2*hidden_size:]
            
            q_weights.append(q)
            k_weights.append(k)
            v_weights.append(v)
            q_bias.append(q_b)
            k_bias.append(k_b)
            v_bias.append(v_b)
            
            o_weights.append(checkpoint["attention.output.dense.weight"])
            o_bias.append(checkpoint["attention.output.dense.bias"])
            
            intermediate_weights.append(checkpoint["intermediate.dense.weight"])
            intermediate_bias.append(checkpoint["intermediate.dense.bias"])
            
            output_weights.append(checkpoint["output.dense.weight"])
            output_bias.append(checkpoint["output.dense.bias"])
            
            model.bert.encoder.layer[layer_idx].attention.output.LayerNorm.weight.data.copy_(
                checkpoint["attention.output.LayerNorm.weight"]
            )
            model.bert.encoder.layer[layer_idx].attention.output.LayerNorm.bias.data.copy_(
                checkpoint["attention.output.LayerNorm.bias"]
            )
            model.bert.encoder.layer[layer_idx].output.LayerNorm.weight.data.copy_(
                checkpoint["output.LayerNorm.weight"]
            )
            model.bert.encoder.layer[layer_idx].output.LayerNorm.bias.data.copy_(
                checkpoint["output.LayerNorm.bias"]
            )
        
        layer = model.bert.encoder.layer[layer_idx]
        layer.attention.self.query.weight.data.copy_(torch.cat(q_weights, dim=0))
        layer.attention.self.key.weight.data.copy_(torch.cat(k_weights, dim=0))
        layer.attention.self.value.weight.data.copy_(torch.cat(v_weights, dim=0))
        layer.attention.self.query.bias.data.copy_(torch.cat(q_bias, dim=0))
        layer.attention.self.key.bias.data.copy_(torch.cat(k_bias, dim=0))
        layer.attention.self.value.bias.data.copy_(torch.cat(v_bias, dim=0))
        
        layer.attention.output.dense.weight.data.copy_(torch.cat(o_weights, dim=1))
        layer.attention.output.dense.bias.data.copy_(o_bias[0])  
        
        layer.intermediate.dense.weight.data.copy_(torch.cat(intermediate_weights, dim=0))
        layer.intermediate.dense.bias.data.copy_(torch.cat(intermediate_bias, dim=0))
        
        layer.output.dense.weight.data.copy_(torch.cat(output_weights, dim=1))
        layer.output.dense.bias.data.copy_(output_bias[0])
    
    mlm_dir = os.path.join(iter_dir, "cls_predictions")
    assert os.path.exists(mlm_dir), f"MLM directory {mlm_dir} does not exist"
    
    for rank_file in sorted(os.listdir(mlm_dir)):
        checkpoint = torch.load(os.path.join(mlm_dir, rank_file), map_location='cpu')
        
        model.cls.predictions.transform.dense.weight.data.copy_(
            checkpoint["transform.dense.weight"]
        )
        model.cls.predictions.transform.dense.bias.data.copy_(
            checkpoint["transform.dense.bias"]
        )
        model.cls.predictions.transform.LayerNorm.weight.data.copy_(
            checkpoint["transform.LayerNorm.weight"]
        )
        model.cls.predictions.transform.LayerNorm.bias.data.copy_(
            checkpoint["transform.LayerNorm.bias"]
        )
        
        if not config.tie_word_embeddings:
            model.cls.predictions.decoder.weight.data.copy_(
                checkpoint["decoder.weight"]
            )
            if hasattr(model.cls.predictions.decoder, "bias"):
                model.cls.predictions.decoder.bias.data.copy_(
                    checkpoint["decoder.bias"]
                )
    
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Successfully converted checkpoint to HuggingFace format at {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert Galvatron checkpoints to HuggingFace format.")
    parser.add_argument("--load_iteration", type=int, required=True, help="Iteration to load.")
    parser.add_argument("--input_checkpoint", type=str, required=True, help="Path to the input Galvatron checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the HuggingFace checkpoint.")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config file.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type.")
    args = parser.parse_args()

    if args.model_type == 'gpt':
        # convert_checkpoints_gpt(args.input_checkpoint, args.output_dir)
        # TODO: implement
        pass
    elif args.model_type == 'llama':
        convert_checkpoints_llama(args.input_checkpoint, args.output_dir, args.load_iteration, args.model_config)
    elif args.model_type == 'bert_mlm':
        convert_checkpoints_bert_mlm(args.input_checkpoint, args.output_dir, args.load_iteration, args.model_config)
if __name__ == "__main__":
    main()
