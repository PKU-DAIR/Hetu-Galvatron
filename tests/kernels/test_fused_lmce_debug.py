#!/usr/bin/env python
"""
Liger Fused Linear Cross Entropy Distributed Test

Compare three implementations with different parallelism strategies:
1. megatron_baseline: Vocab TP (linear + vocab_parallel_cross_entropy)
2. triton_baseline: Vocab TP (linear + triton_fused_vocab_parallel_cross_entropy)
3. liger_fsdp: FSDP Fully Shard (weight sharded, local computation)

Test metrics: memory usage, accuracy, and speed.

Run: torchrun --nproc_per_node=4 test_fused_lmce_debug.py
     torchrun --nproc_per_node=8 test_fused_lmce_debug.py
"""

import torch
import torch.distributed as dist
import sys
sys.path.insert(0, '/home/pkuhetu/wangyj/Galvatron-251022/Liger-Kernel/src')
import galvatron

from megatron.core.parallel_state import initialize_model_parallel, destroy_model_parallel
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.fusions.triton_fused_cross_entropy import triton_fused_vocab_parallel_cross_entropy
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

# Import allgather/reduce_scatter functions
try:
    from megatron.core.utils import is_torch_min_version
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
except:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base


def megatron_vocab_tp(hidden_states, weight, target, sequence_parallel, tp_group, ignore_index=-100):
    """
    Megatron Vocab TP: Linear + Vocab Parallel Cross Entropy
    
    Args:
        hidden_states: [S, B, H] or [S/TP, B, H] if sequence_parallel=True
        weight: [V/TP, H] - vocab partitioned across TP
        target: [S, B] - always full sequence
        sequence_parallel: bool
        tp_group: tensor parallel process group
    
    Returns:
        loss: [S, B] loss tensor
    """
    # Stage 1: LM Head with vocab TP
    logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
        input=hidden_states,
        weight=weight,
        bias=None,
        gradient_accumulation_fusion=False,
        allreduce_dgrad=not sequence_parallel,
        sequence_parallel=sequence_parallel,
        tp_group=tp_group,
    )
    
    # Stage 2: Vocab Parallel Cross Entropy
    loss = vocab_parallel_cross_entropy(
        logits_parallel, 
        target,
    )
    
    return loss


def triton_vocab_tp(hidden_states, weight, target, sequence_parallel, tp_group, ignore_index=-100):
    """
    Triton Vocab TP: Linear + Triton Fused Vocab Parallel Cross Entropy
    
    Args:
        hidden_states: [S, B, H] or [S/TP, B, H] if sequence_parallel=True
        weight: [V/TP, H] - vocab partitioned across TP
        target: [S, B] - always full sequence
        sequence_parallel: bool
        tp_group: tensor parallel process group
    
    Returns:
        loss: [S, B] loss tensor
    """
    # Stage 1: LM Head with vocab TP
    logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
        input=hidden_states,
        weight=weight,
        bias=None,
        gradient_accumulation_fusion=False,
        allreduce_dgrad=not sequence_parallel,
        sequence_parallel=sequence_parallel,
        tp_group=tp_group,
    )
    
    # Stage 2: Triton Fused Vocab Parallel Cross Entropy
    loss = triton_fused_vocab_parallel_cross_entropy(
        logits_parallel, 
        target,
        tp_group=tp_group,
    )
    
    return loss


def liger_fsdp_fully_shard(hidden_states_local, weight_local, target, sequence_parallel, tp_group, ignore_index=-100):
    """
    Liger with FSDP Fully Shard (optimized version)
    
    优化思路：
    - 不需要AllGather hidden_states，直接用local部分计算
    - 只需要AllGather weight
    - 每个rank计算local loss，最后AllGather
    
    关键：使用custom autograd function来正确处理梯度
    
    Args:
        hidden_states_local: [S/TP, B, H] or [S, B, H]
        weight_local: [V/TP, H] - vocab sharded
        target: [S, B] - full target
        sequence_parallel: bool
        tp_group: process group
    
    Returns:
        loss: [S, B] - full loss (after allgather)
    """
    world_size = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    
    # Get shapes
    if sequence_parallel:
        seq_len_local, batch_size, hidden_size = hidden_states_local.shape
        seq_len_global = seq_len_local * world_size
        
        # Split target into local part
        target_local = target[rank * seq_len_local : (rank + 1) * seq_len_local, :]  # [S/TP, B]
    else:
        seq_len_global, batch_size, hidden_size = hidden_states_local.shape
        seq_len_local = seq_len_global
        target_local = target  # [S, B]
    
    vocab_size_local, _ = weight_local.shape
    vocab_size_global = vocab_size_local * world_size
    
    # Custom autograd function to handle weight allgather/reduce-scatter
    class WeightAllGatherReduceScatter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight_local_input, group, sequence_parallel):
            ctx.group = group
            ctx.sequence_parallel = sequence_parallel
            ctx.vocab_size_local = weight_local_input.size(0)
            ctx.hidden_size = weight_local_input.size(1)
            
            vocab_size_global = ctx.vocab_size_local * dist.get_world_size(group)
            weight_global = torch.empty(
                vocab_size_global, ctx.hidden_size,
                dtype=weight_local_input.dtype,
                device=weight_local_input.device,
            )
            dist_all_gather_func(weight_global, weight_local_input.contiguous(), group=group)
            return weight_global
        
        @staticmethod
        def backward(ctx, grad_weight_global):
            group = ctx.group
            sequence_parallel = ctx.sequence_parallel
            vocab_size_local = ctx.vocab_size_local
            hidden_size = ctx.hidden_size
            
            if sequence_parallel:
                # Sequence parallel: reduce_scatter grad_weight
                # 每个rank计算了local loss，grad_weight需要reduce_scatter
                grad_weight_local = torch.empty(
                    vocab_size_local, hidden_size,
                    dtype=grad_weight_global.dtype,
                    device=grad_weight_global.device,
                )
                dist_reduce_scatter_func(
                    grad_weight_local,
                    grad_weight_global.contiguous(),
                    group=group,
                )
            else:
                # No sequence parallel: 每个rank都计算了完整loss
                # grad_weight被重复计算了world_size次，需要平均
                world_size = dist.get_world_size(group)
                grad_weight_global = grad_weight_global / world_size
                
                # Reduce_scatter to get partitioned gradient
                grad_weight_local = torch.empty(
                    vocab_size_local, hidden_size,
                    dtype=grad_weight_global.dtype,
                    device=grad_weight_global.device,
                )
                dist_reduce_scatter_func(
                    grad_weight_local,
                    grad_weight_global.contiguous(),
                    group=group,
                )
            
            return grad_weight_local, None, None
    
    # Custom autograd function to handle loss allgather with proper gradient
    class LossAllGatherWithGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss_local, group, rank, seq_len_local):
            """
            AllGather loss: [S/TP, B] -> [S, B]
            """
            ctx.group = group
            ctx.rank = rank
            ctx.seq_len_local = seq_len_local
            
            seq_len_global = seq_len_local * dist.get_world_size(group)
            batch_size = loss_local.size(1)
            
            loss_global = torch.empty(
                seq_len_global, batch_size,
                dtype=loss_local.dtype,
                device=loss_local.device,
            )
            dist_all_gather_func(loss_global, loss_local.contiguous(), group=group)
            return loss_global
        
        @staticmethod
        def backward(ctx, grad_loss_global):
            """
            Backward: Extract local part of gradient
            grad_loss_global: [S, B] -> grad_loss_local: [S/TP, B]
            """
            group = ctx.group
            rank = ctx.rank
            seq_len_local = ctx.seq_len_local
            
            # Extract local part (no communication needed)
            grad_loss_local = grad_loss_global[rank * seq_len_local : (rank + 1) * seq_len_local, :]
            
            return grad_loss_local, None, None, None
    
    weight_global = WeightAllGatherReduceScatter.apply(weight_local, tp_group, sequence_parallel)
    
    # Call Liger with LOCAL hidden_states and target
    hidden_2d = hidden_states_local.reshape(seq_len_local * batch_size, hidden_size)
    target_1d = target_local.reshape(-1)
    
    loss_1d, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
        hidden_2d,
        weight_global,
        target_1d,
        None, None, -100, 0.0, 0.0, "none", None, False, None, False, False,
    )
    
    loss_local = loss_1d.reshape(seq_len_local, batch_size)  # [S/TP, B]
    
    # AllGather loss to get full loss [S, B] with proper gradient handling
    if sequence_parallel:
        loss_global = LossAllGatherWithGrad.apply(loss_local, tp_group, rank, seq_len_local)
    else:
        loss_global = loss_local
    
    return loss_global


def print_rank0(rank, msg):
    if rank == 0:
        print(msg)


def run_test_forward_backward(lmce_func, hidden_states_cpu, weight_cpu, target_cpu, 
                               sequence_parallel, tp_group, device, rank):
    """Run forward and backward pass, return results on CPU with memory stats."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Move data to GPU
    hidden_states = hidden_states_cpu.to(device).requires_grad_(True)
    weight = weight_cpu.to(device).requires_grad_(True)
    target = target_cpu.to(device)
    
    # Forward
    loss = lmce_func(hidden_states, weight, target, sequence_parallel, tp_group)
    torch.cuda.synchronize()
    mem_after_fwd = torch.cuda.memory_allocated(device) / 1024**2
    
    # Backward
    loss.sum().backward()
    torch.cuda.synchronize()
    
    # Record peak memory
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2
    
    # Transfer results to CPU
    loss_cpu = loss.detach().cpu()
    grad_hidden_states_cpu = hidden_states.grad.clone().cpu() if hidden_states.grad is not None else None
    grad_weight_cpu = weight.grad.clone().cpu() if weight.grad is not None else None
    
    # Clean up GPU
    del hidden_states, weight, target, loss
    torch.cuda.empty_cache()
    
    return loss_cpu, grad_hidden_states_cpu, grad_weight_cpu, mem_after_fwd, mem_peak


def benchmark_performance(lmce_func, hidden_states_cpu, weight_cpu, target_cpu, 
                          sequence_parallel, tp_group, device, warmup=2, iters=5):
    """Benchmark forward+backward timing."""
    # Prepare data on GPU
    hidden_states = hidden_states_cpu.to(device)
    weight = weight_cpu.to(device)
    target = target_cpu.to(device)
    
    # Warmup
    for _ in range(warmup):
        h_copy = hidden_states.detach().requires_grad_(True)
        w_copy = weight.detach().requires_grad_(True)
        loss = lmce_func(h_copy, w_copy, target, sequence_parallel, tp_group)
        loss.sum().backward()
    
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        h_copy = hidden_states.detach().requires_grad_(True)
        w_copy = weight.detach().requires_grad_(True)
        loss = lmce_func(h_copy, w_copy, target, sequence_parallel, tp_group)
        loss.sum().backward()
    end_event.record()
    
    torch.cuda.synchronize()
    del hidden_states, weight, target
    return start_event.elapsed_time(end_event) / iters


def compare_results(name1, name2, loss1, grad_h1, grad_w1, loss2, grad_h2, grad_w2, rank):
    """Compare two versions' results."""
    print_rank0(rank, f"\n{'='*80}\nComparing {name1} vs {name2}\n{'='*80}")
    
    # Convert to float32 for comparison
    loss1 = loss1.float()
    loss2 = loss2.float()
    
    # Loss comparison
    loss_diff = torch.abs(loss1 - loss2)
    loss_abs_max = loss_diff.max().item()
    loss_abs_mean = loss_diff.mean().item()
    loss_rel_max = (loss_diff / (torch.abs(loss1) + 1e-8)).max().item()
    loss_allclose = torch.allclose(loss1, loss2, rtol=1e-2, atol=1e-3)
    
    print_rank0(rank, f"Forward Precision:")
    print_rank0(rank, f"  Loss abs diff: max={loss_abs_max:.2e}, mean={loss_abs_mean:.2e}")
    print_rank0(rank, f"  Loss rel diff: max={loss_rel_max:.2e}")
    print_rank0(rank, f"  torch.allclose: {loss_allclose}")
    
    # Gradient comparisons
    if grad_h1 is not None and grad_h2 is not None:
        grad_h1 = grad_h1.float()
        grad_h2 = grad_h2.float()
        
        grad_h_diff = torch.abs(grad_h1 - grad_h2)
        grad_h_abs_max = grad_h_diff.max().item()
        grad_h_abs_mean = grad_h_diff.mean().item()
        grad_h_magnitude = torch.maximum(torch.abs(grad_h1), torch.abs(grad_h2))
        grad_h_rel_diff = torch.where(grad_h_magnitude > 1e-3, 
                                       grad_h_diff / (grad_h_magnitude + 1e-8),
                                       torch.zeros_like(grad_h_diff))
        grad_h_rel_max = grad_h_rel_diff.max().item()
        grad_h_allclose = torch.allclose(grad_h1, grad_h2, rtol=5e-2, atol=5e-2)
        
        print_rank0(rank, f"Backward Precision (hidden_states grad):")
        print_rank0(rank, f"  Grad abs diff: max={grad_h_abs_max:.2e}, mean={grad_h_abs_mean:.2e}")
        print_rank0(rank, f"  Grad rel diff: max={grad_h_rel_max:.2e}")
        print_rank0(rank, f"  torch.allclose: {grad_h_allclose}")
        
        grad_h_pass = grad_h_allclose or (grad_h_abs_max < 5.0 and grad_h_abs_mean < 0.1)
    else:
        grad_h_pass = True
        print_rank0(rank, "Backward Precision (hidden_states grad): SKIPPED (gradient not available)")
    
    if grad_w1 is not None and grad_w2 is not None:
        grad_w1 = grad_w1.float()
        grad_w2 = grad_w2.float()
        
        grad_w_diff = torch.abs(grad_w1 - grad_w2)
        grad_w_abs_max = grad_w_diff.max().item()
        grad_w_abs_mean = grad_w_diff.mean().item()
        grad_w_magnitude = torch.maximum(torch.abs(grad_w1), torch.abs(grad_w2))
        grad_w_rel_diff = torch.where(grad_w_magnitude > 1e-3,
                                       grad_w_diff / (grad_w_magnitude + 1e-8),
                                       torch.zeros_like(grad_w_diff))
        grad_w_rel_max = grad_w_rel_diff.max().item()
        grad_w_allclose = torch.allclose(grad_w1, grad_w2, rtol=5e-2, atol=5e-2)
        
        print_rank0(rank, f"Backward Precision (weight grad):")
        print_rank0(rank, f"  Grad abs diff: max={grad_w_abs_max:.2e}, mean={grad_w_abs_mean:.2e}")
        print_rank0(rank, f"  Grad rel diff: max={grad_w_rel_max:.2e}")
        print_rank0(rank, f"  torch.allclose: {grad_w_allclose}")
        
        grad_w_pass = grad_w_allclose or (grad_w_abs_max < 5.0 and grad_w_abs_mean < 0.01)
    else:
        grad_w_pass = True
        print_rank0(rank, "Backward Precision (weight grad): SKIPPED (gradient not available)")
    
    # Pass/fail
    loss_pass = loss_allclose or (loss_abs_max < 5.0 and loss_abs_mean < 1.0)
    
    print_rank0(rank, f"\nResult:")
    print_rank0(rank, f"  Forward: {'✓ PASS' if loss_pass else '✗ FAIL'}")
    print_rank0(rank, f"  Backward (hidden_states): {'✓ PASS' if grad_h_pass else '✗ FAIL'}")
    print_rank0(rank, f"  Backward (weight): {'✓ PASS' if grad_w_pass else '✗ FAIL'}")
    
    overall_pass = loss_pass and grad_h_pass and grad_w_pass
    return overall_pass


def test_liger_distributed():
    """Multi-GPU distributed test."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    
    print_rank0(rank, f"{'='*80}\nLiger Distributed Test (TP={world_size})\n{'='*80}")
    
    # Initialize Tensor Parallel
    initialize_model_parallel(tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1)
    tp_group = torch.distributed.new_group(range(world_size))
    dist.barrier()
    
    # Config
    seq_len, batch_size = 4096, 8
    hidden_size = 1024
    vocab_size = 50257
    partition_vocab_size = vocab_size // world_size
    
    print_rank0(rank, f"\nConfig:")
    print_rank0(rank, f"  seq_len={seq_len}, batch={batch_size}")
    print_rank0(rank, f"  hidden_size={hidden_size}, vocab_size={vocab_size}")
    print_rank0(rank, f"  partition_vocab_size={partition_vocab_size}, tp={world_size}")
    
    # Calculate tensor sizes
    hidden_size_bf16 = seq_len * batch_size * hidden_size * 2 / 1024**2
    weight_partition_bf16 = partition_vocab_size * hidden_size * 2 / 1024**2
    weight_full_bf16 = vocab_size * hidden_size * 2 / 1024**2
    logits_partition_bf16 = seq_len * batch_size * partition_vocab_size * 2 / 1024**2
    
    print_rank0(rank, f"\nTensor sizes (bf16):")
    print_rank0(rank, f"  hidden_states [full]: {hidden_size_bf16:.2f} MB")
    print_rank0(rank, f"  weight [partition]: {weight_partition_bf16:.2f} MB")
    print_rank0(rank, f"  weight [full]: {weight_full_bf16:.2f} MB")
    print_rank0(rank, f"  logits [partition]: {logits_partition_bf16:.2f} MB")
    
    # Test both sequence_parallel modes
    for sequence_parallel in [False, True]:
        sp_mode = "sequence_parallel=True" if sequence_parallel else "sequence_parallel=False"
        print_rank0(rank, f"\n{'='*80}\nTesting {sp_mode}\n{'='*80}")
        
        # Create test data
        if sequence_parallel:
            # Sequence parallel: each rank has [S/TP, B, H]
            torch.manual_seed(42 + rank)
            seq_len_local = seq_len // world_size
            hidden_states_sp_cpu = torch.randn(seq_len_local, batch_size, hidden_size, dtype=torch.bfloat16)
            hidden_states_nosp_cpu = torch.randn(seq_len, batch_size, hidden_size, dtype=torch.bfloat16)
        else:
            # No sequence parallel: each rank has full [S, B, H]
            torch.manual_seed(42)
            hidden_states_sp_cpu = torch.randn(seq_len, batch_size, hidden_size, dtype=torch.bfloat16)
            hidden_states_nosp_cpu = hidden_states_sp_cpu.clone()
        
        # Weight: [V/TP, H] for vocab TP
        torch.manual_seed(42 + rank)
        weight_partition_cpu = torch.randn(partition_vocab_size, hidden_size, dtype=torch.bfloat16)
        
        # Labels: [S, B] (always full)
        torch.manual_seed(42)
        target_cpu = torch.randint(0, vocab_size, (seq_len, batch_size), dtype=torch.long)
        
        # ====================================================================
        # Test 1: Megatron Vocab TP
        # ====================================================================
        print_rank0(rank, f"\n{'='*80}\nTest 1: Megatron Vocab TP\n{'='*80}")
        print_rank0(rank, f"Running megatron vocab TP...")
        loss_megatron, grad_h_megatron, grad_w_megatron, mem_fwd_megatron, mem_peak_megatron = \
            run_test_forward_backward(
                megatron_vocab_tp, hidden_states_sp_cpu, weight_partition_cpu, target_cpu,
                sequence_parallel, tp_group, device, rank
            )
        print_rank0(rank, f"  Memory: after_fwd={mem_fwd_megatron:.2f}MB, peak={mem_peak_megatron:.2f}MB")
        print_rank0(rank, f"  Loss sum: {loss_megatron.sum().item():.6f}")
        
        # ====================================================================
        # Test 2: Triton Vocab TP
        # ====================================================================
        print_rank0(rank, f"\n{'='*80}\nTest 2: Triton Vocab TP\n{'='*80}")
        print_rank0(rank, f"Running triton vocab TP...")
        loss_triton, grad_h_triton, grad_w_triton, mem_fwd_triton, mem_peak_triton = \
            run_test_forward_backward(
                triton_vocab_tp, hidden_states_sp_cpu, weight_partition_cpu, target_cpu,
                sequence_parallel, tp_group, device, rank
            )
        print_rank0(rank, f"  Memory: after_fwd={mem_fwd_triton:.2f}MB, peak={mem_peak_triton:.2f}MB")
        print_rank0(rank, f"  Loss sum: {loss_triton.sum().item():.6f}")
        
        # ====================================================================
        # Test 3: Liger with FSDP Fully Shard
        # ====================================================================
        print_rank0(rank, f"\n{'='*80}\nTest 3: Liger FSDP Fully Shard (Optimized)\n{'='*80}")
        print_rank0(rank, f"Running liger FSDP fully shard...")
        loss_liger_fsdp, grad_h_liger_fsdp, grad_w_liger_fsdp, mem_fwd_liger_fsdp, mem_peak_liger_fsdp = \
            run_test_forward_backward(
                liger_fsdp_fully_shard, hidden_states_sp_cpu, weight_partition_cpu, target_cpu,
                sequence_parallel, tp_group, device, rank
            )
        print_rank0(rank, f"  Memory: after_fwd={mem_fwd_liger_fsdp:.2f}MB, peak={mem_peak_liger_fsdp:.2f}MB")
        print_rank0(rank, f"  Loss sum: {loss_liger_fsdp.sum().item():.6f}")
        
        # ====================================================================
        # Compare Results
        # ====================================================================
        compare_results(
            "megatron", "triton",
            loss_megatron, grad_h_megatron, grad_w_megatron,
            loss_triton, grad_h_triton, grad_w_triton,
            rank
        )
        
        # For Liger FSDP, grad_weight is already partitioned, can compare directly
        compare_results(
            "megatron", "liger_fsdp",
            loss_megatron, grad_h_megatron, grad_w_megatron,
            loss_liger_fsdp, grad_h_liger_fsdp, grad_w_liger_fsdp,
            rank
        )
        
        # ====================================================================
        # Memory Comparison
        # ====================================================================
        print_rank0(rank, f"\n{'='*80}\nMemory Usage Comparison ({sp_mode})\n{'='*80}")
        
        print_rank0(rank, f"\nMemory after forward:")
        print_rank0(rank, f"  megatron:       {mem_fwd_megatron:.2f} MB")
        print_rank0(rank, f"  triton:         {mem_fwd_triton:.2f} MB")
        print_rank0(rank, f"  liger_fsdp:     {mem_fwd_liger_fsdp:.2f} MB")
        
        print_rank0(rank, f"\nPeak memory:")
        print_rank0(rank, f"  megatron:       {mem_peak_megatron:.2f} MB")
        print_rank0(rank, f"  triton:         {mem_peak_triton:.2f} MB")
        print_rank0(rank, f"  liger_fsdp:     {mem_peak_liger_fsdp:.2f} MB")
        
        # ====================================================================
        # Performance Benchmarking
        # ====================================================================
        print_rank0(rank, f"\n{'='*80}\nPerformance Benchmarking ({sp_mode})\n{'='*80}")
        print_rank0(rank, "Benchmarking performance...")
        
        time_megatron = benchmark_performance(
            megatron_vocab_tp, hidden_states_sp_cpu, weight_partition_cpu, target_cpu,
            sequence_parallel, tp_group, device
        )
        time_triton = benchmark_performance(
            triton_vocab_tp, hidden_states_sp_cpu, weight_partition_cpu, target_cpu,
            sequence_parallel, tp_group, device
        )
        time_liger_fsdp = benchmark_performance(
            liger_fsdp_fully_shard, hidden_states_sp_cpu, weight_partition_cpu, target_cpu,
            sequence_parallel, tp_group, device
        )
        
        print_rank0(rank, f"\nPerformance Summary:")
        print_rank0(rank, f"  megatron:       {time_megatron:.2f} ms")
        print_rank0(rank, f"  triton:         {time_triton:.2f} ms")
        print_rank0(rank, f"  liger_fsdp:     {time_liger_fsdp:.2f} ms")
        
        dist.barrier()
    
    # Final summary
    print_rank0(rank, f"\n{'='*80}\nTest Complete (TP={world_size})\n{'='*80}")
    
    destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_liger_distributed()
