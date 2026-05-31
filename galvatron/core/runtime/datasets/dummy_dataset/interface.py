"""User-facing entry points: build the data iterator and fetch micro-batches."""

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import galvatron.core.runtime.parallel_state as parallel_state

from .collator import text_collate_fn
from .dataset import build_dummy_text_dataset
from .dyn_bsz import build_dynamic_batch_dataset, dynamic_batch_collate_fn
from .utils import get_text_batch_on_this_tp_rank, get_text_batch_on_this_cp_rank, text_loss_func


def get_dummy_text_data_iterator(
    args = None,
    dataset_size: int = 512,
):
    """
    Build an infinite data iterator for dummy text training.

    ``align_to`` is computed internally as ``sp_size * cp_size * 2``.
    """
    dp_rank = parallel_state.get_vocab_dp_rank()
    dp_world_size = parallel_state.get_vocab_dp_world_size()

    dataset = build_dummy_text_dataset(
        size=dataset_size,
        sequence_length=args.train.seq_length,
        vocab_size=args.model.vocab_size,
        sample_mode=args.data.dummy_sample_mode,
        collate_mode=args.data.dummy_collate_mode,
        min_sequence_length=args.train.seq_length // 2,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.train.global_batch_size // dp_world_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: text_collate_fn(b, collate_mode=args.data.dummy_collate_mode),
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
    cp_size = parallel_state.get_vocab_cp_world_size()
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
            t_start = int(cu_seqlens[i * samples_per_chunk].item()) // cp_size
            t_end = int(cu_seqlens[(i + 1) * samples_per_chunk].item()) // cp_size
            micro_loss_masks.append(loss_masks[:, t_start:t_end])

    effective_tokens = int(loss_masks.sum().item())

    return (
        batch.get('input_ids'),
        {
            'labels': batch.get("labels"),
            'cu_seqlens': batch.get("cu_seqlens"),
        },
        partial(text_loss_func, micro_loss_masks=micro_loss_masks, effective_tokens=effective_tokens, chunks=chunks),
    )


def get_dynamic_batch_data_iterator(
    args,
    length_file: Optional[str] = None,
    dataset_size: int = None,
):
    """
    Build an infinite data iterator for dynamic-batch-size training.

    ``align_to`` is computed internally as ``sp_size * cp_size * 2``.

    Each item produced by the underlying ``DynamicBatchDataset`` packs as many
    variable-length samples (from ``length_file``) as fit within
    ``token_capacity = global_batch_size / dp_world_size / chunks * sequence_length``.
    The ``DataLoader`` then groups ``chunks`` such items into one batch, i.e.
    one global step's worth of tokens on this DP rank.

    If ``length_file`` is None, falls back to the default txt file shipped
    alongside the dyn_bsz dataset module (see ``dyn_bsz.DEFAULT_LENGTH_FILE``).
    """
    dp_rank = parallel_state.get_vocab_dp_rank()
    dp_world_size = parallel_state.get_vocab_dp_world_size()
    global_batch_size = args.train.global_batch_size
    chunks = args.train.chunks
    vocab_size = args.model.vocab_size
    sequence_length = args.train.seq_length

    assert dp_rank is not None and dp_world_size is not None, "dp_rank and dp_world_size should be provided together"
    assert global_batch_size % dp_world_size == 0, (
        f"global_batch_size ({global_batch_size}) must be divisible by dp_world_size ({dp_world_size})"
    )
    per_dp_batch = global_batch_size // dp_world_size
    assert per_dp_batch % chunks == 0, (
        f"global_batch_size / dp_world_size ({per_dp_batch}) must be divisible by chunks ({chunks})"
    )
    token_capacity = (per_dp_batch // chunks) * sequence_length

    build_kwargs = dict(
        max_seq_length=sequence_length,
        token_capacity=token_capacity,
        vocab_size=vocab_size,
        size=dataset_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )
    if length_file is not None:
        build_kwargs["length_file"] = length_file
    dataset = build_dynamic_batch_dataset(**build_kwargs)

    loader = DataLoader(
        dataset,
        batch_size=chunks,
        shuffle=False,
        num_workers=0,
        collate_fn=dynamic_batch_collate_fn,
    )

    def infinite_iterator():
        while True:
            yield from loader

    return infinite_iterator()


def get_dynamic_batch(
    data_iterator,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Callable]:
    """
    Fetch one global step worth of dyn_bsz data and prepare the per-microbatch
    loss-mask slices.

    Returns:
        (input_ids, kwargs, loss_fn) where
          - input_ids: (1, total_T) packed tokens (or a placeholder zeros for
            middle PP stages).
          - kwargs:    {"labels", "cu_seqlens", "cu_seqlens_chunks"} forwarded
            to the model / chunker.
          - loss_fn:   ``text_loss_func`` partially bound to the list of
            per-microbatch ``loss_masks`` slices.
    """
    args = parallel_state.get_args()
    chunks = args.train.chunks

    # Middle PP stages don't need data; return a small placeholder.
    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return torch.zeros([1, 1], device="cuda"), {}, None

    batch = get_text_batch_on_this_tp_rank(data_iterator)
    batch = get_text_batch_on_this_cp_rank(batch)

    loss_masks = batch.pop("loss_masks")
    cu_seqlens = batch["cu_seqlens"]
    cu_seqlens_chunks = batch["cu_seqlens_chunks"]
    assert cu_seqlens is not None and cu_seqlens_chunks is not None, (
        "get_dynamic_batch expects a dyn_bsz batch carrying both cu_seqlens and cu_seqlens_chunks"
    )
    assert cu_seqlens_chunks.numel() == chunks + 1, (
        f"cu_seqlens_chunks length {cu_seqlens_chunks.numel()} != chunks+1 ({chunks + 1}); "
        f"DataLoader.batch_size must equal chunks"
    )

    # Slice loss_masks per micro-batch along the token dim using chunk boundaries.
    cp_size = parallel_state.get_vocab_cp_world_size()
    micro_loss_masks: List[torch.Tensor] = []
    for i in range(chunks):
        t_start = int(cu_seqlens[int(cu_seqlens_chunks[i].item())].item()) // cp_size
        t_end = int(cu_seqlens[int(cu_seqlens_chunks[i + 1].item())].item()) // cp_size
        micro_loss_masks.append(loss_masks[:, t_start:t_end])

    effective_tokens = int(loss_masks.sum().item())

    return (
        batch["input_ids"],
        {
            "labels": batch["labels"],
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_chunks": cu_seqlens_chunks,
        },
        partial(text_loss_func, micro_loss_masks=micro_loss_masks, effective_tokens=effective_tokens, chunks=chunks),
    )
