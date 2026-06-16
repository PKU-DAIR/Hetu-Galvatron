
import torch
import torch.nn.functional as F
import warnings
from typing import Tuple

from galvatron.core.runtime.tensor_parallel.utils import VocabUtility
from galvatron.core.runtime.utils.utils import is_te_min_version

###### BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))


@torch.compile
def geglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return (y_1 * 0.5 * (1.0 + torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1)))) * y_2


@torch.compile
def bias_geglu(bias, y):
    y = y + bias
    return geglu(y)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def geglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    tanh_out = torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * y_1 * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * y_1 * y_1)) + 0.5 * (
        1 + tanh_out
    )
    return torch.cat(((g * y_2) * ff, g * (y_1 * 0.5 * (1.0 + tanh_out))), -1)


@torch.compile
def bias_geglu_back(g, y, bias):
    y = y + bias
    return geglu_back(g, y)


class BiasGeGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_geglu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_geglu_back(grad_output, input, bias)
        return tmp, tmp


class GeGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return geglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = geglu_back(grad_output, input[0])
        return tmp


def bias_geglu_impl(input, bias):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasGeGLUFunction.apply(input, bias)
    else:
        output = GeGLUFunction.apply(input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


# BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))


@torch.compile
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp

    # This is required to make Sphinx happy :-(
    @classmethod
    def apply(cls, *args, **kwargs):
        return super().apply(*args, **kwargs)


bias_gelu_impl = GeLUFunction.apply


@torch.compile
def swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@torch.compile
def bias_swiglu(y, bias):
    y = y + bias
    return swiglu(y)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def swiglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )


@torch.compile
def bias_swiglu_back(g, y, bias):
    y = y + bias
    return swiglu_back(g, y)


class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_swiglu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = bias_swiglu_back(grad_output, input, bias)
        return tmp, tmp, None


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return swiglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = swiglu_back(grad_output, input)
        return tmp, None


def bias_swiglu_impl(input, bias, fp8_input_store=False):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(input, bias, fp8_input_store)
    else:
        output = SwiGLUFunction.apply(input, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


# bias_swiglu_impl = BiasSwiGLUFunction.apply
# swiglu_impl = SwiGLUFunction.apply

# TODO: Add support for fused RoPE from TE
try:

    from transformer_engine.pytorch.attention import FusedRoPEFunc

    def fused_apply_rotary_pos_emb(
        t: torch.Tensor, freqs: torch.Tensor, transpose_output_memory: bool = False
    ) -> torch.Tensor:
        """Apply rotary positional embedding to input tensor T in `sbhd` format."""
        if transpose_output_memory:
            warnings.warn(
                "transpose_output_memory is not supported by TE's fused RoPE and will be ignored."
            )
        return FusedRoPEFunc.apply(t, freqs, "sbhd")

    def fused_apply_rotary_pos_emb_thd(
        t: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor T in `thd` format with CP support.
        """
        if is_te_min_version("1.12.0", check_equality=True):
            return FusedRoPEFunc.apply(t, freqs, "thd", cu_seqlens, cp_size, cp_rank)
        else:
            return FusedRoPEFunc.apply(t, freqs, "thd", cu_seqlens)

except Exception:
    pass

# Fused Vocab Parallel Cross Entropy


class VocabParallelCrossEntropy:
    """
    Computes the Cross Entropy Loss splitting the Vocab size across tensor parallel
    ranks. This implementation is used in both fused and unfused cross entropy implementations
    """

    @staticmethod
    def calculate_logits_max(
        vocab_parallel_logits: torch.Tensor,
        half_entropy: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates logits_max."""

        if not half_entropy:
            vocab_parallel_logits = vocab_parallel_logits.float()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates predicted logits."""

        # In-place subtraction reduces memory pressure.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits

    @staticmethod
    def calculate_cross_entropy_loss(
        exp_logits: torch.Tensor, predicted_logits: torch.Tensor, sum_exp_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates cross entropy loss."""

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        return exp_logits, loss

    @staticmethod
    def prepare_gradient_calculation_operands(
        softmax: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare gradient calculation operands."""

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        return grad_2d, arange_1d, softmax_update, grad_input

    @staticmethod
    def calculate_gradients(
        grad_2d: torch.Tensor,
        arange_1d: torch.Tensor,
        masked_target_1d: torch.Tensor,
        softmax_update: torch.Tensor,
        grad_input: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates gradients."""

        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input


@torch.compile
def calculate_logits_max(vocab_parallel_logits: torch.Tensor, half_entropy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the maximum logits of the predicted tokens.
    """

    vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
        vocab_parallel_logits, half_entropy
    )

    return vocab_parallel_logits, logits_max


@torch.compile
def calculate_predicted_logits(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    logits_max: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the predicted logits for the tokens.
    """
    (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
        VocabParallelCrossEntropy.calculate_predicted_logits(
            vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
        )
    )

    predicted_logits_sum_exp_logits = torch.cat((predicted_logits, sum_exp_logits))

    return target_mask, masked_target_1d, predicted_logits_sum_exp_logits, exp_logits


@torch.compile
def calculate_cross_entropy_loss(
    exp_logits: torch.Tensor, predicted_logits_sum_exp_logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the final cross entropy loss for the tokens.
    """
    split_val = predicted_logits_sum_exp_logits.size()[0] // 2
    predicted_logits, sum_exp_logits = torch.split(predicted_logits_sum_exp_logits, split_val)

    exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
        exp_logits, predicted_logits, sum_exp_logits
    )

    return exp_logits, loss


@torch.compile
def calculate_gradients(
    softmax: torch.Tensor,
    grad_output: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the logits gradients scaled based on the CE loss
    """
    (grad_2d, arange_1d, softmax_update, grad_input) = (
        VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
    )

    grad_input = VocabParallelCrossEntropy.calculate_gradients(
        grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
    )

    grad_input = grad_input.to(torch.bfloat16)

    return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, half_entropy, tp_group):
        """
        Forward implementation for the cross entropy loss.
        """
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits, half_entropy)
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, tp_group.rank(), tp_group.size()
        )

        (target_mask, masked_target_1d, predicted_logits_sum_exp_logits, exp_logits) = (
            calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
            )
        )

        # All reduce is needed to get the chunks from other GPUs.
        # In the fused case, tensors are batches to invoke a single
        # AllReduce call
        torch.distributed.all_reduce(
            predicted_logits_sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )

        exp_logits, loss = calculate_cross_entropy_loss(exp_logits, predicted_logits_sum_exp_logits)

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward implementation for the cross entropy loss.
        """
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        grad_input = calculate_gradients(softmax, grad_output, target_mask, masked_target_1d)

        return grad_input, None, None, None


def fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target, half_entropy, tp_group):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]
        tp_group: the tensor parallel group over which to all reduce

    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, half_entropy, tp_group)


# ── Non-fused reference implementation ────────────────────────────────────────

class _VocabParallelCrossEntropyNonFused(torch.autograd.Function):
    """Non-fused (two separate all-reduces) vocab-parallel CE.

    Serves as a float32 reference baseline; outputs are compared against the
    fused and Triton-fused variants in precision tests.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group):
        vocab_parallel_logits = vocab_parallel_logits.float()
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, tp_group.rank(), tp_group.size()
        )

        (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
            VocabParallelCrossEntropy.calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
            )
        )

        torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        (grad_2d, arange_1d, softmax_update, grad_input) = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
        )
        grad_input = VocabParallelCrossEntropy.calculate_gradients(
            grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
        )
        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group):
    """Non-fused vocab-parallel cross entropy (fp32, two all-reduces).

    Used as the reference baseline in precision tests.

    Args:
        vocab_parallel_logits: ``[S, B, V/TP]`` (any dtype, upcast to fp32 internally)
        target: ``[S, B]`` int64
        tp_group: tensor-parallel process group
    Returns:
        loss: ``[S, B]`` fp32
    """
    return _VocabParallelCrossEntropyNonFused.apply(vocab_parallel_logits, target, tp_group)
