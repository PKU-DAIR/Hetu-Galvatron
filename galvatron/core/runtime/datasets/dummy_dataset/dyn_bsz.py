import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

import galvatron.core.runtime.parallel_state as parallel_state

from .dataset import IGNORE_INDEX, DPAwareDataset


DEFAULT_LENGTH_FILE = os.path.join(os.path.dirname(__file__), "seq_lengths.txt")


class VarLenTextDataset(Dataset):
    """
    Variable-length text dataset whose per-sample sequence lengths are read
    from a txt file (one integer length per line).

    - Samples whose recorded length exceeds ``max_seq_length`` are dropped
      at construction time.
    - Token tensors are NOT materialized in ``__init__``; each sample is
      generated lazily inside ``__getitem__`` using a per-index seeded
      generator, so samples are deterministic given their index.
    - Returned ``input_ids`` always have their real length (no padding).

    Args:
        length_file: Path to a txt file with one integer length per line.
            Defaults to ``DEFAULT_LENGTH_FILE`` (``seq_lengths.txt`` shipped
            alongside this module) when ``None``.
        max_seq_length: Upper bound on sequence length. Samples longer than
            this are filtered out.
        vocab_size: Vocabulary size for the random token ids.
        align_to: If > 1, sample lengths are rounded down to a multiple of
            this value (after the max-length filter). Samples that round
            down to 0 are dropped.
        size: Optional cap on the number of samples to keep (after filtering).
    """

    def __init__(
        self,
        length_file: Optional[str] = None,
        max_seq_length: int = 4096,
        vocab_size: int = 32000,
        size: Optional[int] = None,
    ):
        sp_size = parallel_state.get_vocab_tp_sp_world_size()
        cp_size = parallel_state.get_vocab_cp_world_size()
        align_to = sp_size * cp_size * 2
        assert max_seq_length >= 1, f"max_seq_length must be >= 1, got {max_seq_length}"
        assert align_to >= 1, f"align_to must be >= 1, got {align_to}"
        assert max_seq_length % align_to == 0, f"max_seq_length ({max_seq_length}) must be a multiple of align_to ({align_to})"

        self.length_file = length_file if length_file is not None else DEFAULT_LENGTH_FILE
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.align_to = align_to

        self.lengths: List[int] = self._load_and_filter_lengths()
        if size is not None:
            self.lengths = self.lengths[:size]
        self.size = len(self.lengths)
        assert self.size > 0, f"No samples remain after filtering with max_seq_length={max_seq_length}"

    def _load_and_filter_lengths(self) -> List[int]:
        kept: List[int] = []
        with open(self.length_file, "r") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                length = int(raw)
                if length > self.max_seq_length:
                    continue
                if self.align_to > 1:
                    length = (length // self.align_to) * self.align_to
                if length < 1:
                    continue
                kept.append(length)
        return kept

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index = index % self.size
        real_seq_len = self.lengths[index]

        generator = torch.Generator().manual_seed(index)
        input_ids = torch.randint(0, self.vocab_size, (real_seq_len,), generator=generator, dtype=torch.long)

        labels = torch.full_like(input_ids, IGNORE_INDEX, dtype=torch.long)
        if real_seq_len >= 2:
            labels[: real_seq_len - 1] = input_ids[1:real_seq_len]

        loss_masks = (labels != IGNORE_INDEX).to(torch.float32)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_masks": loss_masks,
        }


class DynamicBatchDataset(Dataset):
    """
    Dynamic-batch-size dataset built on top of a ``VarLenTextDataset``.

    At construction time, samples from the underlying dataset are walked in
    order and greedily concatenated into buckets such that each bucket's
    total token count is ``<= token_capacity`` (typically computed as
    ``micro_batch_size * seq_length`` at train time). The current bucket is
    closed (and a new one started) as soon as the next sample would push
    the total past ``token_capacity``. The resulting list of buckets defines
    the mapping from this dataset's index to a list of indices in the
    underlying dataset.

    In ``__getitem__``, the items returned by the underlying dataset for the
    bucket's indices are concatenated along the sequence dimension.

    Args:
        dataset: The underlying ``VarLenTextDataset`` whose ``lengths`` drive
            the greedy packing.
        token_capacity: Per-bucket token budget (upper bound, inclusive).
    """

    def __init__(self, dataset: VarLenTextDataset, token_capacity: int):
        assert token_capacity >= 1, f"token_capacity must be >= 1, got {token_capacity}"
        assert token_capacity >= dataset.max_seq_length, (
            f"token_capacity ({token_capacity}) must be >= dataset.max_seq_length "
            f"({dataset.max_seq_length}) so that every single sample can fit in a bucket"
        )

        self.dataset = dataset
        self.token_capacity = token_capacity
        self.buckets: List[List[int]] = self._build_buckets()
        self.size = len(self.buckets)

    def _build_buckets(self) -> List[List[int]]:
        buckets: List[List[int]] = []
        current: List[int] = []
        current_total = 0
        for index, length in enumerate(self.dataset.lengths):
            if current and current_total + length > self.token_capacity:
                buckets.append(current)
                current = []
                current_total = 0
            current.append(index)
            current_total += length
        if current:
            buckets.append(current)
        return buckets

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index = index % self.size
        indices = self.buckets[index]
        items = [self.dataset[i] for i in indices]
        lengths = torch.tensor([it["input_ids"].shape[0] for it in items], dtype=torch.int32)
        cu_seqlens = torch.zeros(len(items) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
        return {
            "input_ids": torch.cat([it["input_ids"] for it in items], dim=0),
            "labels": torch.cat([it["labels"] for it in items], dim=0),
            "loss_masks": torch.cat([it["loss_masks"] for it in items], dim=0),
            "cu_seqlens": cu_seqlens,
        }


def dynamic_batch_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for ``DynamicBatchDataset``.

    Each item in ``batch`` is one bucket (one micro-batch worth of tokens)
    with a per-bucket ``cu_seqlens`` describing its internal sample boundaries.
    This function concatenates the buckets along the token dim and merges the
    cu_seqlens into two arrays:

    - ``cu_seqlens``: shape (total_inner_samples + 1,), monotonic, covers
      every inner-sample boundary across the whole pack. Use this for the
      model's attention (flash-attn varlen, etc.).
    - ``cu_seqlens_chunks``: shape (len(batch) + 1,), indices INTO
      ``cu_seqlens`` that mark micro-batch (bucket) boundaries. Use this for
      slicing the pack into ``chunks`` micro-batches:

        token range of chunk i:
            cu_seqlens[cu_seqlens_chunks[i]] : cu_seqlens[cu_seqlens_chunks[i+1]]
        inner-sample boundaries within chunk i:
            cu_seqlens[cu_seqlens_chunks[i] : cu_seqlens_chunks[i+1] + 1]
    """
    input_ids = torch.cat([s["input_ids"] for s in batch], dim=0).unsqueeze(0)    # (1, total_T)
    labels = torch.cat([s["labels"] for s in batch], dim=0).unsqueeze(0)          # (1, total_T)
    loss_masks = torch.cat([s["loss_masks"] for s in batch], dim=0).unsqueeze(0)  # (1, total_T)

    cu_seqlens_parts: List[torch.Tensor] = [torch.zeros(1, dtype=torch.int32)]
    chunk_sample_offsets: List[int] = [0]
    running_token_offset = 0
    running_sample_count = 0
    for s in batch:
        cs = s["cu_seqlens"].to(torch.int32)
        n_inner = cs.numel() - 1
        cu_seqlens_parts.append(cs[1:] + running_token_offset)
        running_token_offset += int(cs[-1].item())
        running_sample_count += n_inner
        chunk_sample_offsets.append(running_sample_count)

    cu_seqlens = torch.cat(cu_seqlens_parts, dim=0)
    cu_seqlens_chunks = torch.tensor(chunk_sample_offsets, dtype=torch.int32)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_masks,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_chunks": cu_seqlens_chunks,
    }


def build_dynamic_batch_dataset(
    length_file: Optional[str] = None,
    max_seq_length: int = 4096,
    token_capacity: int = 4096,
    vocab_size: int = 32000,
    size: Optional[int] = None,
    dp_rank: int = 0,
    dp_world_size: int = 1,
) -> Dataset:
    base_dataset = VarLenTextDataset(
        length_file=length_file,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size,
        size=size,
    )
    packed_dataset = DynamicBatchDataset(
        dataset=base_dataset,
        token_capacity=token_capacity,
    )
    return DPAwareDataset(
        dataset=packed_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )
