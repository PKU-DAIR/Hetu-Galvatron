"""User-facing entry points: build the data iterator and fetch micro-batches."""

from functools import partial
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader

import galvatron.core.runtime.parallel_state as parallel_state

from .collator import text_collate_fn
from .dataset import build_dummy_text_dataset
from .utils import get_text_batch_on_this_tp_rank, get_text_batch_on_this_cp_rank, text_loss_func


def get_dummy_text_data_iterator(
    dp_rank: int = None,
    dp_world_size: int = None,
    global_batch_size: int = 32,
    dataset_size: int = 512,
    sequence_length: int = 4096,
    vocab_size: int = 151936,
    sample_mode: str = "fix_length",
    collate_mode: str = "padding",
    min_sequence_length: int = 16,
    align_to: int = 8,
):
    """
    Build an infinite data iterator for dummy text training.

    Args:
        dp_rank: Current data-parallel rank.
        dp_world_size: Total number of data-parallel ranks.
        global_batch_size: Global batch size across all DP ranks.
        dataset_size: Number of synthetic samples in the base dataset.
        sequence_length: Maximum sequence length per sample.
        vocab_size: Vocabulary size for input_ids.
        sample_mode: ``fix_length`` or ``varlen_length``.
        collate_mode: ``padding`` or ``pack``.
        min_sequence_length: Minimum sequence length when using ``varlen_length``.
        align_to: Sequence lengths will be multiples of this value.

    Returns:
        An iterator that yields batches indefinitely.
    """
    assert dp_rank is not None and dp_world_size is not None, "dp_rank and dp_world_size should be provided together"

    dataset = build_dummy_text_dataset(
        size=dataset_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        sample_mode=sample_mode,
        collate_mode=collate_mode,
        min_sequence_length=min_sequence_length,
        align_to=align_to,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=global_batch_size // dp_world_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: text_collate_fn(b, collate_mode=collate_mode),
    )

    def infinite_iterator():
        while True:
            yield from loader

    return infinite_iterator()


def get_dummy_text_batch(
    data_iterator,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Callable]:
    args = parallel_state.get_args()
    batch_size = args.train.global_batch_size // parallel_state.get_vocab_dp_world_size()
    chunks = args.train.chunks

    assert batch_size >= chunks, (
        f"global_batch_size {args.train.global_batch_size} // dp_world_size {parallel_state.get_vocab_dp_world_size()} "
        f"must be >= chunks {chunks}"
    )
    assert batch_size % chunks == 0, (
        f"global_batch_size {args.train.global_batch_size} // dp_world_size {parallel_state.get_vocab_dp_world_size()} "
        f"must be divisible by chunks {chunks}"
    )

    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return torch.zeros([batch_size, 1], device="cuda"), {}, None

    batch = get_text_batch_on_this_tp_rank(data_iterator)
    batch = get_text_batch_on_this_cp_rank(batch)

    loss_masks = batch.pop("loss_masks")
    cu_seqlens = batch.get("cu_seqlens")

    # Chunk loss_masks for the per-microbatch loss function.
    if cu_seqlens is None:
        # padding mode: slice along the batch dim.
        B = loss_masks.shape[0]
        assert B % chunks == 0, f"Batch size {B} must be divisible by chunks {chunks}"
        per_chunk = B // chunks
        micro_loss_masks = [loss_masks[i * per_chunk : (i + 1) * per_chunk] for i in range(chunks)]
    else:
        # pack mode: slice along the token dim by sample boundaries.
        num_samples = cu_seqlens.numel() - 1
        assert num_samples % chunks == 0, f"Number of samples {num_samples} must be divisible by chunks {chunks}"
        samples_per_chunk = num_samples // chunks
        micro_loss_masks = []
        for i in range(chunks):
            t_start = int(cu_seqlens[i * samples_per_chunk].item())
            t_end = int(cu_seqlens[(i + 1) * samples_per_chunk].item())
            micro_loss_masks.append(loss_masks[:, t_start:t_end])

    return (
        batch.get('input_ids'),
        {
            'labels': batch.get("labels"),
            'cu_seqlens': batch.get("cu_seqlens"),
        },
        partial(text_loss_func, micro_loss_masks=micro_loss_masks),
    )
