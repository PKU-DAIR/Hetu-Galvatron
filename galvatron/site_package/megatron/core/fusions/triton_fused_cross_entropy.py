from typing import Tuple

import torch
import triton
import triton.language as tl

from megatron.core.tensor_parallel.utils import VocabUtility



# ============================================================================
# Triton Kernels for Memory-Optimized Cross Entropy
# ============================================================================

@triton.jit
def _tiled_max_kernel(
    # Input pointers
    logits_ptr,  # [S, B, V] bf16
    # Output pointers
    max_ptr,     # [S, B] fp32
    # Metadata
    seq_len,
    batch_size,
    vocab_size,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tile-wise max reduction: avoid creating full fp32 tensor
    
    Each program handles one token position, computing max tile-wise over vocab dimension
    fp32 only exists in SRAM, not in global memory
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate token position
    batch_idx = pid % batch_size
    seq_idx = pid // batch_size
    
    # Skip invalid positions
    if seq_idx >= seq_len:
        return
    
    # Initialize max to -inf
    max_val = float('-inf')
    
    # Tile-wise processing over vocab dimension
    for vocab_offset in range(0, vocab_size, BLOCK_SIZE):
        # Calculate current tile range
        vocab_indices = vocab_offset + tl.arange(0, BLOCK_SIZE)
        mask = vocab_indices < vocab_size
        
        # Calculate logits address
        logits_offset = seq_idx * batch_size * vocab_size + batch_idx * vocab_size + vocab_indices
        
        # Load bf16 logits and convert to fp32 (in SRAM)
        logits_bf16 = tl.load(logits_ptr + logits_offset, mask=mask, other=float('-inf'))
        logits_fp32 = logits_bf16.to(tl.float32)
        
        # Calculate max within tile (in SRAM)
        tile_max = tl.max(logits_fp32)
        
        # Update global max (in register)
        max_val = tl.maximum(max_val, tile_max)
    
    # Store result
    token_offset = seq_idx * batch_size + batch_idx
    tl.store(max_ptr + token_offset, max_val)


@triton.jit
def _tiled_cross_entropy_forward_kernel(
    # Input pointers
    logits_ptr,  # [S, B, V] bf16
    target_ptr,  # [S, B] int64
    logits_max_ptr,  # [S, B] fp32 (input, already all-reduced)
    # Output pointers
    predicted_logits_ptr,  # [S, B] fp32
    sum_exp_logits_ptr,  # [S, B] fp32
    # Metadata
    seq_len,
    batch_size,
    vocab_size,
    vocab_start_idx,
    vocab_end_idx,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tile-wise forward kernel: compute statistics, avoid storing full fp32 exp_logits
    
    Each program handles one token position (seq_idx, batch_idx)
    Process vocab dimension in tiles, keep fp32 only in SRAM
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate token position
    batch_idx = pid % batch_size
    seq_idx = pid // batch_size
    
    # Skip invalid positions
    if seq_idx >= seq_len:
        return
    
    # Load target and max for this token
    token_offset = seq_idx * batch_size + batch_idx
    target = tl.load(target_ptr + token_offset)
    logits_max = tl.load(logits_max_ptr + token_offset)
    
    # Initialize accumulators (fp32 scalars)
    sum_exp = 0.0
    predicted_logit = 0.0
    
    # Tile-wise processing over vocab dimension
    for vocab_offset in range(0, vocab_size, BLOCK_SIZE):
        # Calculate current tile range
        vocab_indices = vocab_offset + tl.arange(0, BLOCK_SIZE)
        mask = vocab_indices < vocab_size
        
        # Calculate logits address
        logits_offset = seq_idx * batch_size * vocab_size + batch_idx * vocab_size + vocab_indices
        
        # Load bf16 logits and convert to fp32 (in SRAM)
        logits_bf16 = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0)
        logits_fp32 = logits_bf16.to(tl.float32)
        
        # Calculate exp(logits - max) (in SRAM)
        exp_logits = tl.exp(logits_fp32 - logits_max)
        
        # Accumulate sum_exp
        sum_exp += tl.sum(tl.where(mask, exp_logits, 0.0))
        
        # Check if target is in current tile
        global_vocab_indices = vocab_start_idx + vocab_indices
        target_in_tile = (global_vocab_indices == target) & mask
        
        # Extract predicted logit: (logits[target] - max)
        # Note: saving the original logit after subtracting max, not exp
        predicted_logit += tl.sum(tl.where(target_in_tile, logits_fp32 - logits_max, 0.0))
    
    # Store statistics
    tl.store(predicted_logits_ptr + token_offset, predicted_logit)
    tl.store(sum_exp_logits_ptr + token_offset, sum_exp)


@triton.jit
def _tiled_cross_entropy_backward_kernel(
    # Input pointers
    logits_ptr,  # [S, B, V] bf16 (original input)
    target_ptr,  # [S, B] int64
    logits_max_ptr,  # [S, B] fp32
    sum_exp_logits_ptr,  # [S, B] fp32 (all-reduced)
    grad_output_ptr,  # [S, B] fp32
    # Output pointers
    grad_logits_ptr,  # [S, B, V] bf16
    # Metadata
    seq_len,
    batch_size,
    vocab_size,
    vocab_start_idx,
    vocab_end_idx,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tile-wise backward kernel: recompute exp and calculate gradients
    
    Gradient formula: grad = grad_output * (softmax - onehot(target))
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate token position
    batch_idx = pid % batch_size
    seq_idx = pid // batch_size
    
    if seq_idx >= seq_len:
        return
    
    # Load metadata for this token
    token_offset = seq_idx * batch_size + batch_idx
    target = tl.load(target_ptr + token_offset)
    logits_max = tl.load(logits_max_ptr + token_offset)
    sum_exp = tl.load(sum_exp_logits_ptr + token_offset)
    grad_out = tl.load(grad_output_ptr + token_offset)
    
    # Tile-wise processing over vocab dimension
    for vocab_offset in range(0, vocab_size, BLOCK_SIZE):
        vocab_indices = vocab_offset + tl.arange(0, BLOCK_SIZE)
        mask = vocab_indices < vocab_size
        
        # Calculate address
        logits_offset = seq_idx * batch_size * vocab_size + batch_idx * vocab_size + vocab_indices
        
        # Recompute exp (from bf16, in SRAM)
        logits_bf16 = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0)
        logits_fp32 = logits_bf16.to(tl.float32)
        exp_logits = tl.exp(logits_fp32 - logits_max)
        
        # Calculate softmax
        softmax = exp_logits / sum_exp
        
        # Calculate onehot
        global_vocab_indices = vocab_start_idx + vocab_indices
        onehot = (global_vocab_indices == target).to(tl.float32)
        
        # Calculate gradient: grad_output * (softmax - onehot)
        grad = grad_out * (softmax - onehot)
        
        # Convert to bf16 and store back
        grad_bf16 = grad.to(tl.bfloat16)
        tl.store(grad_logits_ptr + logits_offset, grad_bf16, mask=mask)


def tiled_max_reduction(
    vocab_parallel_logits: torch.Tensor,  # [S, B, V/TP] bf16
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """
    Use Triton's tile-wise max reduction, avoid full fp32 tensor
    
    Key optimizations:
    - bf16 → fp32 conversion only happens in SRAM
    - Don't create full fp32 vocab_parallel_logits
    
    Args:
        vocab_parallel_logits: bf16 [S, B, V/TP]
        BLOCK_SIZE: tile size for vocab dimension
    
    Returns:
        logits_max: fp32 [S, B]
    """
    seq_len, batch_size, vocab_size = vocab_parallel_logits.shape
    device = vocab_parallel_logits.device
    
    # Allocate output buffer (only need [S, B], very small)
    logits_max = torch.empty(seq_len, batch_size, dtype=torch.float32, device=device)
    
    # Each program processes one token
    grid = (seq_len * batch_size,)
    
    _tiled_max_kernel[grid](
        vocab_parallel_logits,
        logits_max,
        seq_len,
        batch_size,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return logits_max


def tiled_cross_entropy_forward(
    vocab_parallel_logits: torch.Tensor,  # [S, B, V/TP] bf16
    target: torch.Tensor,  # [S, B] int64
    logits_max: torch.Tensor,  # [S, B] fp32 (already all-reduced)
    vocab_start_idx: int,
    vocab_end_idx: int,
    BLOCK_SIZE: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use Triton's tile-wise forward, avoid full fp32 tensor
    
    Returns:
        predicted_logits: [S, B] fp32 - (logits[target] - max)
        sum_exp_logits: [S, B] fp32 - sum(exp(logits - max))
    """
    seq_len, batch_size, vocab_size = vocab_parallel_logits.shape
    device = vocab_parallel_logits.device
    
    # Allocate output buffers (only statistics, much smaller than exp_logits)
    predicted_logits = torch.zeros(seq_len, batch_size, dtype=torch.float32, device=device)
    sum_exp_logits = torch.zeros(seq_len, batch_size, dtype=torch.float32, device=device)
    
    # Each program processes one token
    grid = (seq_len * batch_size,)
    
    _tiled_cross_entropy_forward_kernel[grid](
        vocab_parallel_logits,
        target,
        logits_max,
        predicted_logits,
        sum_exp_logits,
        seq_len,
        batch_size,
        vocab_size,
        vocab_start_idx,
        vocab_end_idx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return predicted_logits, sum_exp_logits


def tiled_cross_entropy_backward(
    vocab_parallel_logits: torch.Tensor,  # [S, B, V/TP] bf16
    target: torch.Tensor,  # [S, B] int64
    logits_max: torch.Tensor,  # [S, B] fp32
    sum_exp_logits: torch.Tensor,  # [S, B] fp32 (all-reduced)
    grad_output: torch.Tensor,  # [S, B] fp32
    vocab_start_idx: int,
    vocab_end_idx: int,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """
    Use Triton's tile-wise backward, recompute exp
    
    Returns:
        grad_logits: [S, B, V/TP] bf16
    """
    seq_len, batch_size, vocab_size = vocab_parallel_logits.shape
    device = vocab_parallel_logits.device
    
    # Allocate gradient buffer
    grad_logits = torch.empty_like(vocab_parallel_logits)
    
    # Each program processes one token
    grid = (seq_len * batch_size,)
    
    _tiled_cross_entropy_backward_kernel[grid](
        vocab_parallel_logits,
        target,
        logits_max,
        sum_exp_logits,
        grad_output,
        grad_logits,
        seq_len,
        batch_size,
        vocab_size,
        vocab_start_idx,
        vocab_end_idx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return grad_logits


# ============================================================================
# Optimized AutoGrad Function using Triton Kernels
# ============================================================================

class _VocabParallelCrossEntropyTritonFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group):
        """
        Memory-Optimized Forward using Triton Kernels
        
        Core optimizations:
        1. Use Triton tile-wise max, avoid full fp32 tensor
        2. Use Triton tile-wise forward, only output statistics
        3. Only save bf16 logits + statistics for backward
        
        Memory peak:
        - Original: bf16 + fp32
        - Optimized: bf16 only
        - Savings: fp32 (67%)
        """
        # Step 1: Use Triton kernel to compute local max (tile-wise, no fp32 peak)
        # Key: bf16 → fp32 conversion only happens in SRAM, don't create full fp32 tensor
        logits_max = tiled_max_reduction(
            vocab_parallel_logits,  # bf16 [S, B, V/TP]
            BLOCK_SIZE=1024,
        )  # Returns fp32 [S, B], but no fp32 peak during process!
        
        # Step 2: All-Reduce max
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        
        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, tp_group.rank(), tp_group.size()
        )
        
        # Step 3: Use Triton kernel to compute statistics (tile-wise, no fp32 peak)
        # Output only two small [S*B] tensors, not huge [S, B, V] exp_logits
        predicted_logits, sum_exp_logits = tiled_cross_entropy_forward(
            vocab_parallel_logits,  # bf16 [S, B, V/TP]
            target,                  # int64 [S, B]
            logits_max,             # fp32 [S, B]
            vocab_start_index,
            vocab_end_index,
            BLOCK_SIZE=1024,
        )
        
        # Step 4: All-Reduce statistics (small data amount)
        torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        
        # Step 5: Calculate loss
        # predicted_logits = logits[target] - max
        # sum_exp_logits = sum(exp(logits - max))
        # loss = -log(softmax[target]) 
        #      = -log(exp(predicted_logits) / sum_exp_logits)
        #      = -predicted_logits + log(sum_exp_logits)
        #      = log(sum_exp_logits) - predicted_logits
        loss = torch.log(sum_exp_logits) - predicted_logits
        
        # Step 6: Save minimal information for backward
        # Key: only save bf16 logits, don't save fp32 exp_logits!
        ctx.save_for_backward(
            vocab_parallel_logits,  # bf16 [S, B, V/TP]
            target,                  # int64 [S, B] - small
            logits_max,             # fp32 [S, B] - small
            sum_exp_logits,         # fp32 [S, B] - small
        )
        ctx.vocab_start_index = vocab_start_index
        ctx.vocab_end_index = vocab_end_index
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Memory-Optimized Backward using Triton Kernels
        
        Core optimizations:
        1. Recompute exp from saved bf16 logits (tile-wise)
        2. Use Triton kernel, fp32 only in SRAM, not in global memory
        3. Directly output bf16 gradients
        """
        # Restore saved tensors
        vocab_parallel_logits, target, logits_max, sum_exp_logits = ctx.saved_tensors
        vocab_start_index = ctx.vocab_start_index
        vocab_end_index = ctx.vocab_end_index
        
        # Ensure grad_output is contiguous, grad_output may be broadcast tensor (stride=0), need to convert to normal layout
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        
        # Use Triton kernel to compute gradients (tile-wise recompute exp)
        grad_logits = tiled_cross_entropy_backward(
            vocab_parallel_logits,  # bf16 [S, B, V/TP]
            target,                  # int64 [S, B]
            logits_max,             # fp32 [S, B]
            sum_exp_logits,         # fp32 [S, B]
            grad_output,            # fp32 [S, B]
            vocab_start_index,
            vocab_end_index,
            BLOCK_SIZE=1024,
        )
        
        # grad_logits is already bf16, return directly
        return grad_logits, None, None


def triton_fused_vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group,
) -> torch.Tensor:
    """
    Triton-optimized Vocab Parallel Cross Entropy with minimal memory footprint
    
    Core optimizations:
    1. Use Triton tile-wise max, avoid fp32 peak when computing max
    2. Use Triton tile-wise forward, only output statistics, avoid fp32 exp_logits
    3. Forward: only save bf16 logits + statistics
    4. Backward: recompute exp_logits, tile-wise processing
    
    Args:
        vocab_parallel_logits: Tensor parallel logits [S, B, V/TP] in bf16
        target: Target labels [S, B] in int64
        tp_group: Tensor parallel process group
    
    Returns:
        loss: Cross entropy loss [S, B] in fp32
    
    Memory optimization (example: V=50257, S=1024, B=8, TP=4):
        Original (with fp32 peak):
          - bf16 logits: 196 MB
          - fp32 logits (when computing max): 393 MB
          - fp32 exp_logits (forward): 393 MB
          Peak: 589 MB
        
        Optimized version (no fp32 peak):
          - bf16 logits: 196 MB
          - statistics buffers: ~8 MB
          Peak: 196 MB
        
        Savings: 393 MB (67% reduction!)
    """
    return _VocabParallelCrossEntropyTritonFused.apply(vocab_parallel_logits, target, tp_group)



