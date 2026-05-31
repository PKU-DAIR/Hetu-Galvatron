"""User-facing entry points for HF dyn_bsz data: ``get_batch``."""

from functools import partial
from typing import Callable, Dict, List, Tuple

import torch

import galvatron.core.runtime.parallel_state as parallel_state

from .loss_func import hf_loss_func
from .utils import get_batch_on_this_tp, get_batch_on_this_cp_rank


def get_hf_batch(
    data_iterator,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Callable]:
    """Fetch one global step of HF dyn_bsz data and bind per-microbatch loss masks.

    Returns:
        ``(input_ids, kwargs, loss_fn)`` where
          - ``input_ids``: ``(1, total_T)`` packed tokens (placeholder zeros on
            middle PP stages).
          - ``kwargs``: ``{labels, cu_seqlens, cu_seqlens_chunks}`` forwarded to
            the model / chunker.
          - ``loss_fn``: ``hf_packing_loss_func`` partially bound to the list
            of per-microbatch ``loss_masks`` slices.
    """
    args = parallel_state.get_args()
    chunks = args.train.chunks

    # Middle PP stages don't need data; return a small placeholder.
    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return torch.zeros([1, 1], device="cuda"), {}, None

    batch = get_batch_on_this_tp(data_iterator)
    # CP currently a no-op; plug a dyn_bsz CP shard here if/when needed.
    batch = get_batch_on_this_cp_rank(batch)

    loss_masks = batch.pop("loss_masks")
    cu_seqlens = batch.get("cu_seqlens")
    cu_seqlens_chunks = batch.get("cu_seqlens_chunks")
    cp_size = parallel_state.get_vocab_cp_world_size()

    micro_loss_masks: List[torch.Tensor] = []
    if cu_seqlens_chunks is not None:
        # dyn_bsz: micro-batch boundaries given by cu_seqlens_chunks.
        assert cu_seqlens is not None, "cu_seqlens_chunks present but cu_seqlens missing"
        assert cu_seqlens_chunks.numel() == chunks + 1, (
            f"cu_seqlens_chunks length {cu_seqlens_chunks.numel()} != chunks+1 ({chunks + 1}); "
            f"producer's num_microbatches must equal train.chunks"
        )
        for i in range(chunks):
            t_start = int(cu_seqlens[int(cu_seqlens_chunks[i].item())].item()) // cp_size
            t_end = int(cu_seqlens[int(cu_seqlens_chunks[i + 1].item())].item()) // cp_size
            micro_loss_masks.append(loss_masks[:, t_start:t_end])
    elif cu_seqlens is not None:
        # packing: slice the token dim into ``chunks`` equal sample groups.
        num_samples = cu_seqlens.numel() - 1
        assert num_samples % chunks == 0, (
            f"Number of samples {num_samples} must be divisible by chunks {chunks}"
        )
        samples_per_chunk = num_samples // chunks
        for i in range(chunks):
            t_start = int(cu_seqlens[i * samples_per_chunk].item()) // cp_size
            t_end = int(cu_seqlens[(i + 1) * samples_per_chunk].item()) // cp_size
            micro_loss_masks.append(loss_masks[:, t_start:t_end])
    else:
        # padding: slice along the batch dim.
        B = loss_masks.shape[0]
        assert B % chunks == 0, f"Batch size {B} must be divisible by chunks {chunks}"
        per_chunk = B // chunks
        micro_loss_masks = [loss_masks[i * per_chunk : (i + 1) * per_chunk] for i in range(chunks)]

    effective_tokens = int(loss_masks.sum().item())

    kwargs: Dict[str, torch.Tensor] = {"labels": batch["labels"]}
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    if cu_seqlens_chunks is not None:
        kwargs["cu_seqlens_chunks"] = cu_seqlens_chunks
    
    return (
        batch["input_ids"],
        kwargs,
        partial(hf_loss_func, micro_loss_masks=micro_loss_masks, effective_tokens=effective_tokens, chunks=args.train.chunks),
    )
