from __future__ import annotations

from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from galvatron.core.runtime import parallel_state
from galvatron.core.runtime.datasets.huggingface.collator import get_collate_fn
from galvatron.core.runtime.datasets.huggingface.dataset import (
    build_hf_dataset,
    build_hf_iterable_dataset,
)
from galvatron.core.runtime.datasets.huggingface.prefetch_strategy import (
    build_multiprocess_batch_generator,
)


def is_hf_dataset_built_on_rank():
    """True on ranks that should construct HF data iterators (PP first/last + vocab-TP rank 0)."""
    return (
        parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage()
    ) and parallel_state.get_vocab_tensor_parallel_rank() == 0


def get_hf_train_valid_test_data_iterators(args):
    """Build train/valid/test iterators for HuggingFace data.

    Call only on ranks where :func:`is_hf_dataset_built_on_rank` is true; other
    ranks should use ``(None, None, None)`` in the training script before
    ``torch.distributed.barrier``.

    Paths:

    - Train: ``args.data.train_data_path``
    - Valid: ``args.data.valid_data_path`` (optional)
    - Test: ``args.data.test_data_path`` (optional)

    Mode: ``args.data.hf_data_mode`` — ``prefetch`` | ``iterable`` | ``mapping``.
    """
    mode = getattr(args.data, "hf_data_mode", "prefetch")
    dp_world_size = parallel_state.get_vocab_data_parallel_world_size()
    dp_rank = parallel_state.get_vocab_data_parallel_rank()
    batch_size = args.train.global_batch_size // dp_world_size
    text_keys = args.data.hf_text_keys or "text"
    shuffle_buffer = getattr(args.data, "hf_shuffle_buffer_size", 0) or 0
    seed = getattr(args.train, "seed", 42)

    train_path = args.data.train_data_path
    valid_path = args.data.valid_data_path
    test_path = args.data.test_data_path

    if train_path is None:
        raise ValueError("args.data.train_data_path is required when data_source is hf.")

    device = torch.device("cuda", args.local_rank)

    train_it, valid_it, test_it = None, None, None
    if mode == "prefetch":
        train_it = _build_prefetch_loader(args, train_path, "train", text_keys, shuffle_buffer, batch_size)
        if valid_path:
            valid_it = _build_prefetch_loader(args, valid_path, "train", text_keys, 0, batch_size)
        if test_path:
            test_it = _build_prefetch_loader(args, test_path, "train", text_keys, 0, batch_size)
        return (
            iter(train_it),
            iter(valid_it) if valid_it is not None else None,
            iter(test_it) if test_it is not None else None,
        )

    if mode == "iterable":
        train_it = _build_iterable_loader(args, train_path, shuffle_buffer, seed, text_keys, device, batch_size, shuffle=True)
        if valid_path:
            valid_it = _build_iterable_loader(args, valid_path, 0, seed, text_keys, device, batch_size, shuffle=False)
        if test_path:
            test_it = _build_iterable_loader(args, test_path, 0, seed, text_keys, device, batch_size, shuffle=False)
        return (
            iter(train_it),
            iter(valid_it) if valid_it is not None else None,
            iter(test_it) if test_it is not None else None,
        )

    if mode == "mapping":
        train_it = _build_mapping_loader(
            args, train_path, text_keys, batch_size, dp_world_size, dp_rank, shuffle=True
        )
        if valid_path:
            valid_it = _build_mapping_loader(args, valid_path, text_keys, batch_size, dp_world_size, dp_rank, shuffle=False)
        test_it = _build_mapping_loader(args, test_path, text_keys, batch_size, dp_world_size, dp_rank, shuffle=False)
        return (
            iter(train_it),
            iter(valid_it) if valid_it is not None else None,
            iter(test_it) if test_it is not None else None,
        )

    raise ValueError(
        f"Unknown hf_data_mode {mode!r}; expected prefetch, iterable, or mapping."
    )


def _build_prefetch_loader(
    args,
    path: Optional[str],
    split: str,
    text_keys,
    shuffle_buffer: int,
    batch_size: int,
):
    if path is None:
        return None

    return build_multiprocess_batch_generator(
        train_ds_path=path,
        split=split,
        text_keys=text_keys,
        shuffle_buffer_size=shuffle_buffer,
        seed=getattr(args.train, "seed", 42),
        num_workers=getattr(args.data, "hf_num_workers", 0),
        batch_size=batch_size,
        prefetch_factor=getattr(args.data, "hf_prefetch_factor", 2),
    )


def _build_iterable_loader(
    args,
    path: Optional[str],
    shuffle_buffer: int,
    seed: int,
    text_keys,
    device: torch.device,
    batch_size: int,
    shuffle: bool,
):
    if path is None:
        return None

    ds = build_hf_iterable_dataset(
        train_ds_path=path,
        split="train",
        text_keys=text_keys,
        device=device,
        shuffle_buffer_size=shuffle_buffer if shuffle else 0,
        seed=seed,
    )
    collate = get_collate_fn()
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )


def _build_mapping_loader(
    args,
    path: Optional[str],
    text_keys,
    batch_size: int,
    dp_world_size: int,
    dp_rank: int,
    *,
    shuffle: bool,
):
    if path is None:
        return None
        
    ds = build_hf_dataset(path, text_keys=text_keys, split="train")
    sampler = DistributedSampler(
        ds,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=shuffle,
        seed=getattr(args.train, "seed", 42),
    )
    collate = get_collate_fn()
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=getattr(args.data, "hf_num_workers", 0),
        pin_memory=True,
    )
