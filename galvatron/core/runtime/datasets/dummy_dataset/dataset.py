from typing import Dict

import torch
from torch.utils.data import Dataset

import galvatron.core.runtime.parallel_state as parallel_state


# Ignore index for labels (no loss on padding / last position).
IGNORE_INDEX = 0 # TODO: set another value


class DummyTextDataset(Dataset):
    """
    Dummy text dataset.

    Two modes:
        - ``fix_length``: every sample has length ``sequence_length``.
        - ``varlen_length``: each sample has a random length in
          ``[min_sequence_length, sequence_length]``, aligned up to a multiple of ``align_to``.

    All samples are pre-generated in __init__, one at a time.
    Each sample is deterministic given its index (independent of global RNG).
    """

    def __init__(
        self,
        size: int = 16,
        sequence_length: int = 4096,
        vocab_size: int = 32000,
        sample_mode: str = "fix_length",
        collate_mode: str = "padding",
        min_sequence_length: int = 16,
    ):
        sp_size = parallel_state.get_vocab_tp_sp_world_size()
        cp_size = parallel_state.get_vocab_cp_world_size()
        align_to = sp_size * cp_size * 2
        assert sample_mode in ("fix_length", "varlen_length"), f"sample_mode must be 'fix_length' or 'varlen_length', got {sample_mode!r}"
        assert collate_mode in ("padding", "pack"), f"collate_mode must be 'padding' or 'pack', got {collate_mode!r}"
        assert sequence_length % align_to == 0, f"sequence_length ({sequence_length}) must be a multiple of align_to ({align_to})"
        assert 1 <= min_sequence_length <= sequence_length, f"min_sequence_length ({min_sequence_length}) must be in [1, sequence_length={sequence_length}]"

        self.size = size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.sample_mode = sample_mode
        self.collate_mode = collate_mode
        self.min_sequence_length = min_sequence_length
        self.align_to = align_to

        self.data = [self._make_sample(index) for index in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index = index % self.size

        if self.collate_mode == "padding" and self.sample_mode == "varlen_length":
            real_seq_len = self.data[index].shape[0]
            input_ids = torch.full((self.sequence_length,), IGNORE_INDEX, dtype=torch.long)  # pad to max length with IGNORE_INDEX
            input_ids[:real_seq_len] = self.data[index]
        else:
            input_ids = self.data[index]
            real_seq_len = input_ids.shape[0]

        # labels are next-token-shifted; positions without a real next token are IGNORE_INDEX.
        labels = torch.full_like(input_ids, IGNORE_INDEX, dtype=torch.long)
        if real_seq_len >= 2:
            labels[: real_seq_len - 1] = input_ids[1:real_seq_len]

        loss_masks = (labels != IGNORE_INDEX).to(torch.float32)

        return {
            "input_ids": input_ids,    # (seq_len,)
            "labels": labels,          # (seq_len,)
            "loss_masks": loss_masks,  # (seq_len,)
        }

    def _make_sample(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(index)
        seq_len = self._sample_length(generator)
        return torch.randint(0, self.vocab_size, (seq_len,), generator=generator, dtype=torch.long)

    def _sample_length(self, generator: torch.Generator) -> int:
        if self.sample_mode == "fix_length":
            return self.sequence_length
        # varlen_length: align min up, align max down, then pick a random multiple.
        low_units = (self.min_sequence_length + self.align_to - 1) // self.align_to
        high_units = self.sequence_length // self.align_to
        return int(torch.randint(low_units, high_units + 1, (1,), generator=generator).item()) * self.align_to
    

class DPAwareDataset(Dataset):
    """
    Data-Parallel aware dataset wrapper.

    Partitions the underlying dataset by ``dp_rank`` so that each DP rank only
    sees its own slice. No ``DistributedSampler`` is required.

    The total dataset size must be identical across all DP ranks.
    """

    def __init__(
        self,
        dataset: Dataset,
        dp_rank: int,
        dp_world_size: int,
    ):
        self.dataset = dataset
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.full_length = len(dataset)

    def __len__(self) -> int:
        # ceil division so that every rank has the same number of steps
        return (self.full_length + self.dp_world_size - 1) // self.dp_world_size

    def __getitem__(self, index: int):
        global_index = index * self.dp_world_size + self.dp_rank
        if global_index >= self.full_length:
            # Wrap around to keep iteration lengths identical across ranks.
            global_index = global_index % self.full_length
        return self.dataset[global_index]


def build_dummy_text_dataset(
    size: int = 16,
    sequence_length: int = 4096,
    vocab_size: int = 32000,
    sample_mode: str = "fix_length",
    collate_mode: str = "padding",
    min_sequence_length: int = 16,
    dp_rank: int = 0,
    dp_world_size: int = 1,
) -> Dataset:
    base_dataset = DummyTextDataset(
        size=size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        sample_mode=sample_mode,
        collate_mode=collate_mode,
        min_sequence_length=min_sequence_length,
    )
    return DPAwareDataset(
        dataset=base_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )
