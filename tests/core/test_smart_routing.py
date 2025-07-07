import torch
import sys
import os
from unittest.mock import patch
import random

# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from galvatron.core.runtime.moe.smart_routing import MoEAlltoAllSmartTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


@patch('megatron.core.parallel_state.get_expert_tensor_parallel_world_size', return_value=1)
@patch('megatron.core.parallel_state.get_expert_model_parallel_world_size', return_value=8)
@patch('megatron.core.parallel_state.get_expert_model_parallel_rank', return_value=0)
@patch('megatron.core.parallel_state.get_expert_tensor_parallel_rank', return_value=0)
def test_get_smart_routing_map(mock_tp_rank, mock_ep_rank, mock_ep_world_size, mock_tp_world_size):
    """
    测试get_smart_routing_map函数
    """
    print("Testing get_smart_routing_map")
    
    # 准备测试数据 - 基于用户提供的输入
    num_global_tokens_per_expert = torch.tensor([[[504, 632, 608, 549, 424, 557, 313, 509],
                                                   [429, 584, 603, 525, 476, 644, 301, 534],
                                                   [434, 665, 609, 538, 422, 586, 326, 516],
                                                   [450, 593, 620, 512, 410, 624, 313, 574],
                                                   [467, 576, 631, 553, 426, 560, 311, 572],
                                                   [435, 585, 604, 561, 448, 573, 335, 555],
                                                   [505, 681, 571, 519, 436, 531, 328, 525],
                                                   [462, 565, 591, 546, 427, 628, 328, 549]]], 
                                                device='cuda' if torch.cuda.is_available() else 'cpu')
    
    global_expert_indices = torch.tensor([[0, 1],
                                         [2, 3],
                                         [4, 5],
                                         [6, 7],
                                         [0, 1],
                                         [2, 3],
                                         [4, 5],
                                         [6, 7]], 
                                        device='cuda' if torch.cuda.is_available() else 'cpu', 
                                        dtype=torch.int32)
    
    print(f"Input shapes:")
    print(f"  num_global_tokens_per_expert: {num_global_tokens_per_expert.shape}")
    print(f"  global_expert_indices: {global_expert_indices.shape}")
    print()
    
    # 创建配置和dispatcher
    config = TransformerConfig(
        num_moe_experts=8,
        moe_router_topk=2,
        hidden_size=1024,
        num_attention_heads=8,
        num_layers=24,
    )
    
    # 创建dispatcher实例
    local_expert_indices = [0, 1]  # 假设当前rank的本地专家
    num_local_experts = 2
    
    dispatcher = MoEAlltoAllSmartTokenDispatcher(
        num_local_experts=num_local_experts,
        local_expert_indices=local_expert_indices,
        global_expert_indices=global_expert_indices,
        config=config
    )
    
    # 调用get_smart_routing_map
    print("Calling get_smart_routing_map...")
    result = dispatcher.get_smart_routing_map(num_global_tokens_per_expert, global_expert_indices)
    result = result.to(device='cpu')
    
    print(f"Output shape: {result.shape}")
    print(f"Expected shape: [tp_size={num_global_tokens_per_expert.shape[0]}, "
          f"ep_size={num_global_tokens_per_expert.shape[1]}, "
          f"num_local_experts * ep_size={num_local_experts * num_global_tokens_per_expert.shape[1]}]")
    print()
    
    # 验证输出维度
    tp_size, ep_size, num_global_experts = num_global_tokens_per_expert.shape
    _, num_local_experts_from_indices = global_expert_indices.shape
    expected_output_shape = (tp_size, ep_size, ep_size * num_local_experts_from_indices)
    
    assert result.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {result.shape}"
    print("Output shape is correct")
    
    # 验证token守恒 - 总的输入token数应该等于总的输出token数
    total_input_tokens = num_global_tokens_per_expert.sum()
    total_output_tokens = result.sum()
    
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    
    assert abs(total_input_tokens - total_output_tokens) < 1e-6, "Token conservation failed"
    print("Token conservation verified")
    
    # 分析路由策略效果
    print("\nRouting Analysis:")
    
    # 分析节点内 vs 节点间通信
    gpus_per_node = 8
    intra_node_tokens = 0
    inter_node_tokens = 0
    
    for tp_idx in range(tp_size):
        for src_rank in range(ep_size):
            src_node = src_rank // gpus_per_node
            for target_idx in range(ep_size * num_local_experts_from_indices):
                target_rank = target_idx // num_local_experts_from_indices
                target_node = target_rank // gpus_per_node
                tokens = result[tp_idx, src_rank, target_idx]
                
                if src_node == target_node:
                    intra_node_tokens += tokens
                else:
                    inter_node_tokens += tokens
    
    print(f"Intra-node tokens: {intra_node_tokens}")
    print(f"Inter-node tokens: {inter_node_tokens}")
    print(f"Intra-node ratio: {intra_node_tokens / (intra_node_tokens + inter_node_tokens) * 100:.2f}%")
    
    # 检查专家负载均衡
    print("\nLoad Balancing Analysis:")
    expert_loads = torch.zeros(num_global_experts)
    
    for tp_idx in range(tp_size):
        for target_rank in range(ep_size):
            for local_expert_idx in range(num_local_experts_from_indices):
                global_expert_id = global_expert_indices[target_rank, local_expert_idx]
                target_idx = target_rank * num_local_experts_from_indices + local_expert_idx
                expert_load = result[tp_idx, :, target_idx].sum()
                expert_loads[global_expert_id] += expert_load
    
    print(f"Expert loads: {expert_loads}")
    print(f"Load std: {expert_loads.std():.2f}")
    print(f"Load mean: {expert_loads.mean():.2f}")
    
    # 验证每个专家都有对应的副本
    expert_locations = {}
    for rank in range(ep_size):
        for local_idx in range(num_local_experts_from_indices):
            expert_id = global_expert_indices[rank, local_idx].item()
            if expert_id not in expert_locations:
                expert_locations[expert_id] = []
            expert_locations[expert_id].append((rank, local_idx))
    
    print("\nExpert Replication:")
    for expert_id, locations in expert_locations.items():
        print(f"Expert {expert_id}: replicated at {len(locations)} locations: {locations}")
    
    print("\nAll tests passed!")
    return result

@patch('megatron.core.parallel_state.get_expert_tensor_parallel_world_size', return_value=1)
@patch('megatron.core.parallel_state.get_expert_model_parallel_world_size', return_value=8)
@patch('megatron.core.parallel_state.get_expert_model_parallel_rank', return_value=0)
@patch('megatron.core.parallel_state.get_expert_tensor_parallel_rank', return_value=0)
def test_get_new_routing_map(result, mock_tp_rank, mock_ep_rank, mock_ep_world_size, mock_tp_world_size):
    """
    测试get_new_routing_map函数 - 针对rank0的数据
    """
    print("\nTesting get_new_routing_map")
    
    # 使用rank0的实际数据: [504, 632, 608, 549, 424, 557, 313, 509]
    rank0_data = [504, 632, 608, 549, 424, 557, 313, 509]
    new_num_global_tokens_per_expert = result

    # 创建配置和dispatcher
    config = TransformerConfig(
        num_moe_experts=8,
        moe_router_topk=2,
        hidden_size=1024,
        num_attention_heads=8,
        num_layers=24,
    )
    
    global_expert_indices = torch.tensor([[0, 1],
                                         [2, 3],
                                         [4, 5],
                                         [6, 7],
                                         [0, 1],
                                         [2, 3],
                                         [4, 5],
                                         [6, 7]], 
                                        device='cuda' if torch.cuda.is_available() else 'cpu', 
                                        dtype=torch.int32)
    
    dispatcher = MoEAlltoAllSmartTokenDispatcher(
        num_local_experts=2,
        local_expert_indices=[0, 1],
        global_expert_indices=global_expert_indices,
        config=config
    )
    
    routing_map = torch.zeros((sum(rank0_data), 8), dtype=torch.bool)
    probs = torch.zeros((sum(rank0_data), 8))

    total = []
    for i in range(len(rank0_data)):
        total.extend([i] * rank0_data[i])
    
    random.shuffle(total)
    for i in range(len(total)):
        routing_map[i, total[i]] = True
        probs[i, total[i]] = 0.8 + total[i] * 0.02


    # 调用get_new_routing_map
    print("\nCalling get_new_routing_map...")
    new_routing_map, new_probs = dispatcher.get_new_routing_map(
        new_num_global_tokens_per_expert, global_expert_indices, routing_map, probs
    )
    
    print(f"Results:")
    print(f"  new_routing_map shape: {new_routing_map.shape}")
    print(f"  new_probs shape: {new_probs.shape}")
    
    # 统计结果
    original_token_count = routing_map.sum().item()
    new_token_count = new_routing_map.sum().item()
    new_location_loads = new_routing_map.sum(dim=0)
    
    print(f"  Original total tokens: {original_token_count}")
    print(f"  New total tokens: {new_token_count}")
    print(f"  New tokens per location: {new_location_loads}")
    
    # 验证基本约束
    assert new_token_count == original_token_count, "Token count should not increase"
    
    print("✓ All basic checks passed!")
    print("✓ get_new_routing_map test completed!")
    
    return new_routing_map, new_probs

if __name__ == "__main__":
    # 设置随机种子保证结果可重现
    torch.manual_seed(42)
    random.seed(42)
    
    # 运行测试
    result = test_get_smart_routing_map()
    
    print(result)
    # 测试get_new_routing_map函数
    new_routing_map, new_probs = test_get_new_routing_map(result)