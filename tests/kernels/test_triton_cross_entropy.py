#!/usr/bin/env python
"""
Cross Entropy Tensor Parallel Distributed Precision Test with pytest

Test three versions:
1. non_fused_ce: vocab_parallel_cross_entropy
2. jit_fused_ce: fused_vocab_parallel_cross_entropy
3. triton_fused_ce: triton_fused_vocab_parallel_cross_entropy

Comparison: non_fused vs jit_fused, triton_fused vs non_fused, triton_fused vs jit_fused

Run: pytest test_triton_cross_entropy.py -v -s
"""

import sys
import json
import logging
import pytest
import torch
import torch.distributed as dist
import galvatron

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[Rank %(rank)s] %(message)s',
    force=True
)

from megatron.core.parallel_state import initialize_model_parallel, destroy_model_parallel
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.fusions.triton_fused_cross_entropy import triton_fused_vocab_parallel_cross_entropy


# ============================================================================
# Helper Functions
# ============================================================================


def non_fused_ce(logits, target, tp_group):
    return vocab_parallel_cross_entropy(logits, target)


def jit_fused_ce(logits, target, tp_group):
    return fused_vocab_parallel_cross_entropy(logits, target, half_entropy=False, tp_group=tp_group)


def triton_fused_ce(logits, target, tp_group):
    return triton_fused_vocab_parallel_cross_entropy(logits, target, tp_group=tp_group)


def print_rank0(rank, msg):
    """Print message only from rank 0."""
    if rank == 0:
        # Use both print and logging to ensure output is visible
        print(f"[Rank {rank}] {msg}", flush=True)
        logger = logging.getLogger(__name__)
        logger.info(msg)


def run_test_forward_backward(ce_func, logits_cpu, target_cpu, tp_group, device):
    """Run forward and backward pass, return results on CPU with memory stats."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    logits = logits_cpu.to(device).requires_grad_(True)
    target = target_cpu.to(device)
    
    # Forward
    loss = ce_func(logits, target, tp_group)
    torch.cuda.synchronize()
    mem_after_fwd = torch.cuda.memory_allocated(device) / 1024**2
    
    # Backward
    loss.sum().backward()
    torch.cuda.synchronize()
    
    # Record peak memory before transferring to CPU
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2
    
    # Transfer results to CPU
    loss_cpu = loss.detach().cpu()
    grad_cpu = logits.grad.clone().cpu()
    
    # Clean up GPU
    del logits, target, loss
    torch.cuda.empty_cache()
    
    return loss_cpu, grad_cpu, mem_after_fwd, mem_peak


def benchmark_performance(ce_func, logits_cpu, target_cpu, tp_group, device, warmup=20, iters=100):
    """Benchmark forward+backward timing (excluding data transfer)."""
    # Prepare data on GPU
    logits = logits_cpu.to(device)
    target = target_cpu.to(device)
    
    # Warmup
    for _ in range(warmup):
        logits_copy = logits.detach().requires_grad_(True)
        loss = ce_func(logits_copy, target, tp_group)
        loss.sum().backward()
    
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        logits_copy = logits.detach().requires_grad_(True)
        loss = ce_func(logits_copy, target, tp_group)
        loss.sum().backward()
    end_event.record()
    
    torch.cuda.synchronize()
    del logits, target
    return start_event.elapsed_time(end_event) / iters


def compare_results(name1, name2, loss1, grad1, loss2, grad2, rank):
    """Compare two versions' results."""
    print_rank0(rank, f"\n{'='*80}\nComparing {name1} and {name2}\n{'='*80}")
    
    # Loss comparison
    loss_diff = torch.abs(loss1 - loss2)
    loss_abs_max = loss_diff.max().item()
    loss_abs_mean = loss_diff.mean().item()
    loss_rel_max = (loss_diff / (torch.abs(loss1) + 1e-8)).max().item()
    
    # Gradient comparison
    grad_diff = torch.abs(grad1 - grad2)
    grad_abs_max = grad_diff.max().item()
    grad_abs_mean = grad_diff.mean().item()
    grad_rel_max = (grad_diff / (torch.abs(grad1) + 1e-8)).max().item()
    
    # torch.allclose comparison (for BF16: rtol=1e-2, atol=1e-3)
    loss_allclose = torch.allclose(loss1, loss2, rtol=1e-2, atol=1e-3)
    grad_allclose = torch.allclose(grad1, grad2, rtol=1e-2, atol=1e-3)
    
    print_rank0(rank, f"Forward Precision:")
    print_rank0(rank, f"  Loss abs diff: max={loss_abs_max:.2e}, mean={loss_abs_mean:.2e}")
    print_rank0(rank, f"  Loss rel diff: max={loss_rel_max:.2e}")
    print_rank0(rank, f"  torch.allclose: {loss_allclose}")
    
    print_rank0(rank, f"Backward Precision:")
    print_rank0(rank, f"  Grad abs diff: max={grad_abs_max:.2e}, mean={grad_abs_mean:.2e}")
    print_rank0(rank, f"  Grad rel diff: max={grad_rel_max:.2e}")
    print_rank0(rank, f"  torch.allclose: {grad_allclose}")
    
    # Pass/fail (use allclose as primary criterion)
    loss_pass = loss_allclose or (loss_abs_max < 1e-2 and loss_rel_max < 0.01)
    grad_pass = grad_allclose or (grad_abs_max < 1e-2 and grad_rel_max < 0.1)
    
    print_rank0(rank, f"\nResult:")
    print_rank0(rank, f"  Forward: {'PASS' if loss_pass else 'FAIL'}")
    print_rank0(rank, f"  Backward: {'PASS' if grad_pass else 'FAIL'}")

def _run_test(args):
    """Main test logic (runs in each distributed process)"""
    # Initialize distributed environment
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    
    # Setup logging for this process
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f'[Rank {rank}] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Parse arguments
    tp_size = args.get("tp_size", world_size)
    seq_len = args.get("seq_len", 1024)
    batch_size = args.get("batch_size", 8)
    vocab_size = args.get("vocab_size", 50257)
    model_config = args.get("model_config", "unknown")
    
    assert world_size == tp_size, f"world_size {world_size} != tp_size {tp_size}"
    
    print_rank0(rank, f"{'='*80}\nCross Entropy Test [{model_config}] (TP={tp_size})\n{'='*80}")
    sys.stdout.flush()
    
    # Initialize Tensor Parallel
    initialize_model_parallel(tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1)
    tp_group = torch.distributed.new_group(range(world_size))
    dist.barrier()
    
    # Config
    partition_vocab_size = vocab_size // tp_size
    print_rank0(rank, f"\nConfig: seq_len={seq_len}, batch={batch_size}, vocab={vocab_size}, tp={tp_size}")
    
    # Create test data on CPU
    torch.manual_seed(42 + rank)
    logits_cpu = torch.randn(seq_len, batch_size, partition_vocab_size, dtype=torch.bfloat16)
    torch.manual_seed(42)
    target_cpu = torch.randint(0, vocab_size, (seq_len, batch_size), dtype=torch.long)
    
    # Run tests
    print_rank0(rank, f"\n{'='*80}\nRunning Tests\n{'='*80}")
    print_rank0(rank, "Testing precision and memory consumption...")
    loss_nf, grad_nf, mem_fwd_nf, mem_peak_nf = run_test_forward_backward(
        non_fused_ce, logits_cpu, target_cpu, tp_group, device
    )
    print_rank0(rank, f"non_fused_ce - after_fwd: {mem_fwd_nf:.2f}MB, peak: {mem_peak_nf:.2f}MB")
    
    loss_jf, grad_jf, mem_fwd_jf, mem_peak_jf = run_test_forward_backward(
        jit_fused_ce, logits_cpu, target_cpu, tp_group, device
    )
    print_rank0(rank, f"jit_fused_ce - after_fwd: {mem_fwd_jf:.2f}MB, peak: {mem_peak_jf:.2f}MB")
    
    loss_tf, grad_tf, mem_fwd_tf, mem_peak_tf = run_test_forward_backward(
        triton_fused_ce, logits_cpu, target_cpu, tp_group, device
    )
    print_rank0(rank, f"triton_fused_ce - after_fwd: {mem_fwd_tf:.2f}MB, peak: {mem_peak_tf:.2f}MB")
    
    # Pairwise comparisons
    compare_results("non_fused_ce", "jit_fused_ce", loss_nf, grad_nf, loss_jf, grad_jf, rank)
    compare_results("triton_fused_ce", "non_fused_ce", loss_tf, grad_tf, loss_nf, grad_nf, rank)
    compare_results("triton_fused_ce", "jit_fused_ce", loss_tf, grad_tf, loss_jf, grad_jf, rank)
    
    # Memory comparison
    print_rank0(rank, f"\n{'='*80}\nMemory Usage Comparison\n{'='*80}")
    logits_size_bf16 = batch_size * seq_len * partition_vocab_size * 2 / 1024**2
    print_rank0(rank, f"Logits size bf16: {logits_size_bf16:.2f} MB")
    print_rank0(rank, f"\nMemory after forward:")
    print_rank0(rank, f"  non_fused_ce:    {mem_fwd_nf:.2f} MB")
    print_rank0(rank, f"  jit_fused_ce:    {mem_fwd_jf:.2f} MB")
    print_rank0(rank, f"  triton_fused_ce: {mem_fwd_tf:.2f} MB")
    print_rank0(rank, f"\nPeak memory:")
    print_rank0(rank, f"  non_fused_ce:    {mem_peak_nf:.2f} MB")
    print_rank0(rank, f"  jit_fused_ce:    {mem_peak_jf:.2f} MB")
    print_rank0(rank, f"  triton_fused_ce: {mem_peak_tf:.2f} MB")

    # Performance benchmarking
    print_rank0(rank, f"\n{'='*80}\nPerformance Benchmarking\n{'='*80}")
    
    print_rank0(rank, "Benchmarking performance...")
    time_nf = benchmark_performance(non_fused_ce, logits_cpu, target_cpu, tp_group, device)
    time_jf = benchmark_performance(jit_fused_ce, logits_cpu, target_cpu, tp_group, device)
    time_tf = benchmark_performance(triton_fused_ce, logits_cpu, target_cpu, tp_group, device)
    
    print_rank0(rank, f"\nPerformance Summary:")
    print_rank0(rank, f"  non_fused_ce:    {time_nf:.2f} ms (baseline)")
    print_rank0(rank, f"  jit_fused_ce:    {time_jf:.2f} ms ({time_nf/time_jf:.2f}x speedup)")
    print_rank0(rank, f"  triton_fused_ce: {time_tf:.2f} ms ({time_nf/time_tf:.2f}x speedup)")
    
    # Cleanup
    del loss_nf, loss_jf, loss_tf, grad_nf, grad_jf, grad_tf, logits_cpu, target_cpu
    torch.cuda.empty_cache()
    dist.barrier()
    
    print_rank0(rank, f"\n{'='*80}\nTest Complete (TP={tp_size})\n{'='*80}")
    
    destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize("tp_size,seq_len,batch_size,vocab_size,model_config", [    
    # (4, 1024, 8, 32000, "llama2"),
    (4, 1024, 8, 50257, "gpt2"),
    # (4, 1024, 8, 128256, "llama3"),
    (8, 4096, 8, 129280, "deepseek_v3.1"),
    (8, 4096, 8, 151936, "qwen3"),
])
def test_triton_cross_entropy(run_distributed, tp_size, seq_len, batch_size, vocab_size, model_config):
    """Pytest entry point for distributed cross entropy test"""
    args = {
        "tp_size": tp_size,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "model_config": model_config,
    }
    run_distributed("_run_test", tp_size, args, __file__)


if __name__ == "__main__":
    # Entry point for distributed processes
    func_name = sys.argv[1]
    args_json = sys.argv[2]
    args = json.loads(args_json)
    
    if func_name == "_run_test":
        _run_test(args)

