#!/usr/bin/env python
"""
Triton Kernels Precision Test with pytest

Test each Triton kernel's numerical precision:
1. tiled_max_reduction - Max computation
2. tiled_cross_entropy_forward - Forward statistics
3. tiled_cross_entropy_backward - Backward gradients

Run: pytest test_triton_cross_entropy_kernels.py -v -s
"""

import pytest
import torch
import galvatron


# ============================================================================
# Test Configurations
# ============================================================================

# Common test cases (seq_len, batch_size, vocab_size, model_config)
TEST_CASES = [
    # Basic test
    (1024, 8, 1000, "basic"),
    (4096, 8, 1000, "basic"),
    
    # LLaMA2 (vocab_size=32000)
    (1024, 1, 32000, "llama2"),
    (4096, 1, 32000, "llama2"),
    
    # GPT-2 (vocab_size=50257)
    (1024, 1, 50257, "gpt2"),
    (4096, 1, 50257, "gpt2"),
    
    # LLaMA3 (vocab_size=128256)
    (1024, 1, 128256, "llama3"),
    (4096, 1, 128256, "llama3"),
    
    # DeepSeek-V3.1 (vocab_size=129280)
    (1024, 1, 129280, "deepseek_v3.1"),
    (4096, 1, 129280, "deepseek_v3.1"),
    
    # Qwen3 (vocab_size=151936)
    (1024, 1, 151936, "qwen3"),
    (4096, 1, 151936, "qwen3"),
]

# Edge cases test (case_name, seq_len, batch_size, vocab_size)
EDGE_CASES = [
    # Small vocab test
    ("small_vocab", 10, 8, 1000),
    
    # Real model vocab sizes
    ("llama2_vocab", 10, 1, 32000),
    ("gpt2_vocab", 10, 1, 50257),
    ("llama3_vocab", 10, 1, 128256),
    ("deepseek_vocab", 10, 1, 129280),
    ("qwen3_vocab", 10, 1, 151936),
    
    # Extreme values
    ("extreme_values", 10, 8, 1000),
]


# ============================================================================
# Fixtures and Utilities
# ============================================================================

@pytest.fixture(scope="module")
def device():
    """Get CUDA device for testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda:0")


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset random seed before each test."""
    torch.manual_seed(42)


def check_precision(triton_val, torch_val, name, rtol=1e-2, atol=1e-3):
    """Check precision with both allclose and manual diff."""
    abs_diff = torch.abs(triton_val - torch_val)
    rel_diff = abs_diff / (torch.abs(torch_val) + 1e-8)
    
    allclose = torch.allclose(triton_val, torch_val, rtol=rtol, atol=atol)
    
    print(f"\n  {name}:")
    print(f"    abs diff: max={abs_diff.max().item():.2e}, mean={abs_diff.mean().item():.2e}")
    print(f"    rel diff: max={rel_diff.max().item():.2e}, mean={rel_diff.mean().item():.2e}")
    print(f"    allclose: {allclose}")
    
    passed = allclose or (abs_diff.max() < atol and rel_diff.max() < rtol)
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}]")
    
    assert passed, (
        f"{name} precision check failed: "
        f"max_abs={abs_diff.max().item():.2e}, "
        f"max_rel={rel_diff.max().item():.2e}"
    )
    
    return passed


# ============================================================================
# Test 1: Max Reduction Kernel
# ============================================================================


@pytest.mark.parametrize("seq_len,batch_size,vocab_size,model_config", TEST_CASES)
def test_max_reduction(device, seq_len, batch_size, vocab_size, model_config):
    """Test tiled_max_reduction precision."""
    dtype = torch.bfloat16
    print(f"\n{'='*80}")
    print(f"Test: Max Reduction [{model_config}]")
    print(f"Config: S={seq_len}, B={batch_size}, V={vocab_size}, dtype={dtype}")
    print(f"{'='*80}")
    
    from megatron.core.fusions.triton_fused_cross_entropy import tiled_max_reduction
    
    logits = torch.randn(seq_len, batch_size, vocab_size, device=device, dtype=dtype)
    
    max_triton = tiled_max_reduction(logits, BLOCK_SIZE=1024)
    max_torch = torch.max(logits.float(), dim=-1)[0]
    
    check_precision(max_triton, max_torch, "max", rtol=1e-3, atol=1e-2)


# ============================================================================
# Test 2: Forward Kernel
# ============================================================================


@pytest.mark.parametrize("seq_len,batch_size,vocab_size,model_config", TEST_CASES)
def test_forward(device, seq_len, batch_size, vocab_size, model_config):
    """Test tiled_cross_entropy_forward precision."""
    print(f"\n{'='*80}")
    print(f"Test: Forward [{model_config}]")
    print(f"Config: S={seq_len}, B={batch_size}, V={vocab_size}")
    print(f"{'='*80}")
    
    from megatron.core.fusions.triton_fused_cross_entropy import tiled_cross_entropy_forward
    from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
    
    logits = torch.randn(seq_len, batch_size, vocab_size, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, vocab_size, (seq_len, batch_size), device=device, dtype=torch.long)
    logits_max = torch.max(logits.float(), dim=-1)[0]
    
    # Triton version
    predicted_triton, sum_exp_triton = tiled_cross_entropy_forward(
        logits, target, logits_max, 0, vocab_size, BLOCK_SIZE=1024
    )
    
    # Baseline (PyTorch)
    logits_fp32 = logits.float().clone()
    (_, _, predicted_torch, sum_exp_torch, _) = VocabParallelCrossEntropy.calculate_predicted_logits(
        logits_fp32, target, logits_max, 0, vocab_size
    )
    
    # Check precision
    check_precision(predicted_triton, predicted_torch, "predicted", rtol=1e-3, atol=1e-2)
    check_precision(sum_exp_triton, sum_exp_torch, "sum_exp", rtol=1e-3, atol=1e-2)


# ============================================================================
# Test 3: Backward Kernel
# ============================================================================


@pytest.mark.parametrize("seq_len,batch_size,vocab_size,model_config", TEST_CASES)
def test_backward(device, seq_len, batch_size, vocab_size, model_config):
    """Test tiled_cross_entropy_backward precision."""
    print(f"\n{'='*80}")
    print(f"Test: Backward [{model_config}]")
    print(f"Config: S={seq_len}, B={batch_size}, V={vocab_size}")
    print(f"{'='*80}")
    
    from megatron.core.fusions.triton_fused_cross_entropy import tiled_cross_entropy_backward
    from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
    
    logits = torch.randn(seq_len, batch_size, vocab_size, device=device, dtype=torch.bfloat16)
    target = torch.randint(0, vocab_size, (seq_len, batch_size), device=device, dtype=torch.long)
    grad_output = torch.randn(seq_len, batch_size, device=device, dtype=torch.float32)
    
    # Prepare intermediate values using baseline
    logits_fp32 = logits.float().clone()
    logits_max = torch.max(logits_fp32, dim=-1)[0]
    
    (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
        VocabParallelCrossEntropy.calculate_predicted_logits(logits_fp32, target, logits_max, 0, vocab_size)
    )
    
    softmax_torch, _ = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
        exp_logits.clone(), predicted_logits, sum_exp_logits
    )
    
    (grad_2d, arange_1d, softmax_update, grad_input) = (
        VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax_torch, target_mask)
    )
    
    grad_torch = VocabParallelCrossEntropy.calculate_gradients(
        grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
    ).to(torch.bfloat16)
    
    # Triton version
    grad_triton = tiled_cross_entropy_backward(
        logits, target, logits_max, sum_exp_logits, grad_output, 0, vocab_size, BLOCK_SIZE=1024
    )
    
    # Check precision (backward requires looser tolerance)
    check_precision(grad_triton.float(), grad_torch.float(), "gradient", rtol=1e-2, atol=5e-2)


# ============================================================================
# Test 4: Edge Cases
# ============================================================================


@pytest.mark.parametrize("case_name,seq_len,batch_size,vocab_size", EDGE_CASES)
def test_edge_cases_max(device, case_name, seq_len, batch_size, vocab_size):
    """Test edge cases for max reduction."""
    print(f"\n{'='*80}")
    print(f"Test: Edge Case - {case_name} (S={seq_len}, B={batch_size}, V={vocab_size})")
    print(f"{'='*80}")
    
    from megatron.core.fusions.triton_fused_cross_entropy import tiled_max_reduction
    
    logits = torch.randn(seq_len, batch_size, vocab_size, device=device, dtype=torch.bfloat16)
    
    if case_name == "extreme_values":
        logits = logits * 10
        logits[0, 0, 0] = 100.0
        logits[1, 1, 1] = -100.0
    
    max_triton = tiled_max_reduction(logits, BLOCK_SIZE=1024)
    max_torch = torch.max(logits.float(), dim=-1)[0]
    
    allclose = torch.allclose(max_triton, max_torch, rtol=1e-2, atol=1e-2)
    print(f"\n  allclose: {allclose}")
    status = "PASS" if allclose else "FAIL"
    print(f"  [{status}]")
    
    assert allclose, f"Edge case {case_name} failed"


def test_boundary_targets(device):
    """Test boundary target indices."""
    print(f"\n{'='*80}")
    print(f"Test: Boundary Targets (vocab=1000)")
    print(f"{'='*80}")
    
    from megatron.core.fusions.triton_fused_cross_entropy import tiled_cross_entropy_forward
    
    logits = torch.randn(10, 1, 1000, device=device, dtype=torch.bfloat16)
    target = torch.zeros(10, 1, device=device, dtype=torch.long)
    target[1, :] = 999
    
    logits_max = torch.max(logits.float(), dim=-1)[0]
    predicted, sum_exp = tiled_cross_entropy_forward(logits, target, logits_max, 0, 1000, BLOCK_SIZE=1024)
    
    finite = torch.isfinite(predicted).all() and torch.isfinite(sum_exp).all()
    positive = (sum_exp > 0).all()
    
    print(f"\n  finite: {finite}, sum_exp > 0: {positive}")
    status = "PASS" if (finite and positive) else "FAIL"
    print(f"  [{status}]")
    
    assert finite, "Predicted or sum_exp has non-finite values"
    assert positive, "Sum_exp has non-positive values"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])