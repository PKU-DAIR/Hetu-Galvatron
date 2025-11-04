#!/usr/bin/env python
"""
Triton Kernels Precision Test

Test each Triton kernel's numerical precision:
1. tiled_max_reduction - Max computation
2. tiled_cross_entropy_forward - Forward statistics
3. tiled_cross_entropy_backward - Backward gradients

Run: python test_triton_cross_entropy_kernels_debug.py
"""

import torch
import galvatron


def check_precision(triton_val, torch_val, name, rtol=1e-2, atol=1e-3):
    """Check precision with both allclose and manual diff."""
    abs_diff = torch.abs(triton_val - torch_val)
    rel_diff = abs_diff / (torch.abs(torch_val) + 1e-8)
    
    allclose = torch.allclose(triton_val, torch_val, rtol=rtol, atol=atol)
    
    print(f"  {name}:")
    print(f"    abs diff: max={abs_diff.max().item():.2e}, mean={abs_diff.mean().item():.2e}")
    print(f"    rel diff: max={rel_diff.max().item():.2e}, mean={rel_diff.mean().item():.2e}")
    print(f"    allclose: {allclose}")
    
    passed = allclose or (abs_diff.max() < atol and rel_diff.max() < rtol)
    print(f"    {'PASS' if passed else 'FAIL'}")
    
    return passed


def test_max_reduction():
    """Test tiled_max_reduction precision."""
    print(f"\n{'='*80}\nTest 1: tiled_max_reduction\n{'='*80}")
    
    from megatron.core.fusions.triton_fused_cross_entropy import tiled_max_reduction
    
    device = torch.device("cuda:0")
    test_cases = [
        (128, 4, 1000, torch.bfloat16),
        (1024, 8, 12564, torch.bfloat16),
        (2048, 16, 50257, torch.bfloat16),
        (4096, 2, 128256, torch.bfloat16),
    ]
    
    all_passed = True
    for seq_len, batch_size, vocab_size, dtype in test_cases:
        print(f"\nCase: S={seq_len}, B={batch_size}, V={vocab_size}, dtype={dtype}")
        
        torch.manual_seed(42)
        logits = torch.randn(seq_len, batch_size, vocab_size, device=device, dtype=dtype)
        
        max_triton = tiled_max_reduction(logits, BLOCK_SIZE=1024)
        max_torch = torch.max(logits.float(), dim=-1)[0]
        
        passed = check_precision(max_triton, max_torch, "max", rtol=1e-3, atol=1e-2)
        all_passed = all_passed and passed
    
    return all_passed


def test_forward():
    """Test tiled_cross_entropy_forward precision."""
    print(f"\n{'='*80}\nTest 2: tiled_cross_entropy_forward\n{'='*80}")
    
    from megatron.core.fusions.triton_fused_cross_entropy import tiled_cross_entropy_forward
    from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
    
    device = torch.device("cuda:0")
    test_cases = [(128, 4, 1000), (1024, 8, 12564), (2048, 16, 50257)]
    
    all_passed = True
    for seq_len, batch_size, vocab_size in test_cases:
        print(f"\nCase: S={seq_len}, B={batch_size}, V={vocab_size}")
        
        torch.manual_seed(42)
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
        pred_pass = check_precision(predicted_triton, predicted_torch, "predicted", rtol=1e-3, atol=1e-2)
        sum_pass = check_precision(sum_exp_triton, sum_exp_torch, "sum_exp", rtol=1e-3, atol=1e-2)
        
        all_passed = all_passed and pred_pass and sum_pass
    
    return all_passed


def test_backward():
    """Test tiled_cross_entropy_backward precision."""
    print(f"\n{'='*80}\nTest 3: tiled_cross_entropy_backward\n{'='*80}")
    
    from megatron.core.fusions.triton_fused_cross_entropy import tiled_cross_entropy_backward
    from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
    
    device = torch.device("cuda:0")
    test_cases = [(128, 4, 1000), (1024, 8, 12564), (512, 16, 50257)]
    
    all_passed = True
    for seq_len, batch_size, vocab_size in test_cases:
        print(f"\nCase: S={seq_len}, B={batch_size}, V={vocab_size}")
        
        torch.manual_seed(42)
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
        passed = check_precision(grad_triton.float(), grad_torch.float(), "gradient", rtol=1e-2, atol=5e-2)
        all_passed = all_passed and passed
    
    return all_passed


def test_edge_cases():
    """Test edge cases."""
    print(f"\n{'='*80}\nTest 4: Edge Cases\n{'='*80}")
    
    from megatron.core.fusions.triton_fused_cross_entropy import (
        tiled_max_reduction, tiled_cross_entropy_forward
    )
    
    device = torch.device("cuda:0")
    
    test_configs = [
        ("Small vocab (V < BLOCK_SIZE)", 10, 4, 512),
        ("Non-divisible vocab", 10, 4, 50257),
        ("Extreme values", 10, 4, 1000),
    ]
    
    all_passed = True
    for name, seq_len, batch_size, vocab_size in test_configs:
        print(f"\n{name}: S={seq_len}, B={batch_size}, V={vocab_size}")
        
        torch.manual_seed(42)
        logits = torch.randn(seq_len, batch_size, vocab_size, device=device, dtype=torch.bfloat16)
        
        if "Extreme" in name:
            logits = logits * 10
            logits[0, 0, 0] = 100.0
            logits[1, 1, 1] = -100.0
        
        max_triton = tiled_max_reduction(logits, BLOCK_SIZE=1024)
        max_torch = torch.max(logits.float(), dim=-1)[0]
        
        allclose = torch.allclose(max_triton, max_torch, rtol=1e-2, atol=1e-2)
        print(f"  allclose: {allclose}")
        print(f"  {'PASS' if allclose else 'FAIL'}")
        all_passed = all_passed and allclose
    
    # Test boundary targets
    print(f"\nBoundary targets: vocab=1000")
    torch.manual_seed(42)
    logits = torch.randn(10, 4, 1000, device=device, dtype=torch.bfloat16)
    target = torch.zeros(10, 4, device=device, dtype=torch.long)
    target[1, :] = 999
    
    logits_max = torch.max(logits.float(), dim=-1)[0]
    predicted, sum_exp = tiled_cross_entropy_forward(logits, target, logits_max, 0, 1000, BLOCK_SIZE=1024)
    
    finite = torch.isfinite(predicted).all() and torch.isfinite(sum_exp).all()
    positive = (sum_exp > 0).all()
    print(f"  finite: {finite}, sum_exp > 0: {positive}")
    print(f"  {'PASS' if (finite and positive) else 'FAIL'}")
    all_passed = all_passed and finite and positive
    
    return all_passed


def main():
    """Run all precision tests."""
    print(f"\n{'='*80}\nTriton Kernels Precision Test Suite\n{'='*80}")
    
    tests = [
        ("max_reduction", test_max_reduction),
        ("forward", test_forward),
        ("backward", test_backward),
        ("edge_cases", test_edge_cases),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print(f"\n{'='*80}\nSummary\n{'='*80}")
    for name, passed in results.items():
        print(f"  {name:20s}: {'PASS' if passed else 'FAIL'}")
    
    all_passed = all(results.values())
    print(f"\n{'='*80}")
    print(f"{'All tests passed!' if all_passed else 'Some tests failed'}")
    print(f"{'='*80}\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
