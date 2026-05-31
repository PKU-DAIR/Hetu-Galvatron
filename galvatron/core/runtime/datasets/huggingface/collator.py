from itertools import accumulate
from typing import List, Tuple, Callable

import torch
from torch import Tensor
from galvatron.core.runtime.parallel_state import get_args, get_tokenizer
import galvatron.core.runtime.parallel_state as parallel_state

# from .loss_func import hf_packing_loss_func


# Ignore index for labels (no loss on padding / last position).
IGNORE_INDEX = -100


class PackingCollator:
    """
    Fixed-shape packing collator for HF data. ``batch`` is a flat list of 1D
    token tensors whose length is a multiple of ``chunks``; consecutive
    ``len(batch) // chunks`` samples form one micro-batch chunk.

    Each sample is padded to a multiple of ``vocab_cp_size * 2`` tokens so
    that zigzag CP splitting is aligned.  Padding tokens use ``labels =
    IGNORE_INDEX`` and ``loss_masks = 0`` and are appended to the sample
    they belong to in ``cu_seqlens``.

    Output dict mirrors ``DynamicBatchCollator`` but without
    ``cu_seqlens_chunks`` — chunk boundaries are implicit since every chunk
    holds the same number of real samples.
    """

    def __init__(self, chunks: int):
        assert chunks >= 1, f"chunks must be >= 1, got {chunks}"
        self.chunks = chunks

    def __call__(self, batch: List[Tensor]) -> dict:
        assert len(batch) % self.chunks == 0, (
            f"batch length {len(batch)} must be a multiple of chunks {self.chunks}"
        )
        samples_per_chunk = len(batch) // self.chunks

        cp_size = parallel_state.get_vocab_cp_world_size()
        cp_pad_multiple = cp_size * 2 if cp_size > 1 else 1
        tp_sp_size = parallel_state.get_vocab_tp_sp_world_size()

        input_ids_parts: List[Tensor] = []
        labels_parts: List[Tensor] = []
        loss_masks_parts: List[Tensor] = []
        sample_lengths: List[int] = []
        dtype = None

        for chunk_idx in range(self.chunks):
            chunk_start = len(sample_lengths)

            for i in range(samples_per_chunk):
                t = batch[chunk_idx * samples_per_chunk + i].view(-1)
                n_orig = t.numel()
                dtype = t.dtype

                input_ids_parts.append(t)
                labels = torch.empty_like(t)
                labels[:-1] = t[1:]
                labels[-1] = IGNORE_INDEX
                labels_parts.append(labels)
                loss_masks_parts.append(torch.ones_like(t, dtype=torch.float32))

                if cp_pad_multiple > 1:
                    rem = n_orig % cp_pad_multiple
                    if rem != 0:
                        pad_len = cp_pad_multiple - rem
                        input_ids_parts.append(torch.zeros(pad_len, dtype=dtype))
                        labels_parts.append(torch.full((pad_len,), IGNORE_INDEX, dtype=dtype))
                        loss_masks_parts.append(torch.zeros(pad_len, dtype=torch.float32))
                        n_orig += pad_len

                sample_lengths.append(n_orig)

            # Pad chunk total to multiple of tp_sp_size
            chunk_tokens = sum(sample_lengths[chunk_start:])
            rem = chunk_tokens % tp_sp_size
            if rem != 0:
                pad_len = tp_sp_size - rem
                input_ids_parts.append(torch.zeros(pad_len, dtype=dtype))
                labels_parts.append(torch.full((pad_len,), IGNORE_INDEX, dtype=dtype))
                loss_masks_parts.append(torch.zeros(pad_len, dtype=torch.float32))
                sample_lengths[-1] += pad_len

        input_ids = torch.cat(input_ids_parts, dim=0).unsqueeze(0).contiguous()
        labels = torch.cat(labels_parts, dim=0).unsqueeze(0).contiguous()
        loss_masks = torch.cat(loss_masks_parts, dim=0).unsqueeze(0).contiguous()

        cu_seqlens = torch.tensor(list(accumulate(sample_lengths, initial=0)), dtype=torch.int32)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_masks": loss_masks,
            "cu_seqlens": cu_seqlens,
        }

class PaddingCollator:
    def __init__(self, seq_length: int, pad_token_id: int = 0):
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        args = get_args()
        self.use_flash_attn = getattr(args.train, "use_flash_attn", False)

    def __call__(self, batch: List[Tensor]) -> dict:
        padded_ids = []
        valid_lens = []
        for t in batch:
            t = t.view(-1)
            if t.numel() >= self.seq_length:
                t = t[: self.seq_length]
                valid_lens.append(self.seq_length)
            else:
                pad_len = self.seq_length - t.numel()
                valid_lens.append(t.numel())
                t = torch.cat(
                    [
                        t,
                        torch.full(
                            (pad_len,),
                            self.pad_token_id,
                            dtype=t.dtype
                        ),
                    ],
                    dim=0,
                )
            padded_ids.append(t)

        input_ids = torch.stack(padded_ids, dim=0)
        valid_lens_t = torch.tensor(valid_lens, dtype=torch.long)
        # True = real token positions; avoids conflating pad fill with real token id (e.g. pad=0 vs content id 0).
        padding_mask = torch.arange(self.seq_length, dtype=torch.long).unsqueeze(0) < valid_lens_t.unsqueeze(1)

        labels = torch.full_like(
            input_ids, IGNORE_INDEX, dtype=input_ids.dtype
        )
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = IGNORE_INDEX
        labels[~padding_mask] = IGNORE_INDEX

        loss_masks = (labels != IGNORE_INDEX).float()
        position_ids = torch.arange(
            self.seq_length, dtype=torch.long
        ).unsqueeze(0).expand(input_ids.size(0), -1).contiguous()

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_masks": loss_masks,
            "position_ids": position_ids,
        }

        if not self.use_flash_attn:
            batch_size, seq_length = input_ids.size()
            causal = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)
            pad_key_mask = (~padding_mask).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_length, seq_length)
            result["attention_mask"] = pad_key_mask | causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        return result

class DynamicBatchCollator:
    """
    Dyn-bsz pack collator for HF data. Input is a list of ``num_microbatches``
    buckets, each a list of variable-length 1D token tensors. Each bucket is
    packed into one micro-batch.  Each sample is padded to a multiple of
    ``vocab_cp_size * 2`` and each bucket is padded to a multiple of
    ``vocab_tp_sp_size``.  Padding is appended to the last sample in the
    bucket with ``loss_masks = 0`` and ``labels = IGNORE_INDEX``.

    Output dict:

    - ``input_ids`` / ``labels`` / ``loss_masks``: ``(1, total_T)`` tensors.
    - ``cu_seqlens``: ``List[int]`` of length ``N+1``.
    - ``cu_seqlens_chunks``: ``List[int]`` of length ``num_microbatches+1``,
      indices into ``cu_seqlens`` marking micro-batch ends.

    ``cu_seqlens`` variants are kept as plain Python lists to stay queue-
    friendly (cheap pickle, no torch shared-memory churn). Convert to int32
    tensors on the consumer side if downstream code needs them.
    """

    def __init__(self):
        pass

    def __call__(self, buckets: List[List[Tensor]]) -> dict:
        cp_size = parallel_state.get_vocab_cp_world_size()
        cp_pad_multiple = cp_size * 2 if cp_size > 1 else 1
        tp_sp_size = parallel_state.get_vocab_tp_sp_world_size()

        input_ids_parts: List[Tensor] = []
        labels_parts: List[Tensor] = []
        loss_masks_parts: List[Tensor] = []
        sample_lengths: List[int] = []
        chunk_sample_offsets: List[int] = [0]
        dtype = None

        for bucket in buckets:
            bucket_start = len(sample_lengths)

            for t in bucket:
                n_orig = t.numel()
                dtype = t.dtype

                input_ids_parts.append(t)
                labels = torch.empty_like(t)
                labels[:-1] = t[1:]
                labels[-1] = IGNORE_INDEX
                labels_parts.append(labels)
                loss_masks_parts.append(torch.ones_like(t, dtype=torch.float32))

                if cp_pad_multiple > 1:
                    rem = n_orig % cp_pad_multiple
                    if rem != 0:
                        pad_len = cp_pad_multiple - rem
                        input_ids_parts.append(torch.zeros(pad_len, dtype=dtype))
                        labels_parts.append(torch.full((pad_len,), IGNORE_INDEX, dtype=dtype))
                        loss_masks_parts.append(torch.zeros(pad_len, dtype=torch.float32))
                        n_orig += pad_len

                sample_lengths.append(n_orig)

            # Pad bucket total to multiple of tp_sp_size
            bucket_tokens = sum(sample_lengths[bucket_start:])
            rem = bucket_tokens % tp_sp_size
            if rem != 0:
                pad_len = tp_sp_size - rem
                input_ids_parts.append(torch.zeros(pad_len, dtype=dtype))
                labels_parts.append(torch.full((pad_len,), IGNORE_INDEX, dtype=dtype))
                loss_masks_parts.append(torch.zeros(pad_len, dtype=torch.float32))
                sample_lengths[-1] += pad_len

            chunk_sample_offsets.append(len(sample_lengths))

        input_ids = torch.cat(input_ids_parts, dim=0).unsqueeze(0).contiguous()
        labels = torch.cat(labels_parts, dim=0).unsqueeze(0).contiguous()
        loss_masks = torch.cat(loss_masks_parts, dim=0).unsqueeze(0).contiguous()

        cu_seqlens: List[int] = list(accumulate(sample_lengths, initial=0))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_masks": loss_masks,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_chunks": chunk_sample_offsets,
        }


def get_collate_fn():
    args = get_args()
    if args.data.hf_data_mode == 'dyn_bsz':
        return DynamicBatchCollator()
    if args.data.hf_collator_mode == "packing":
        chunks = args.train.chunks # accumulation steps
        return PackingCollator(chunks=chunks)
    elif args.data.hf_collator_mode == "padding":
        tokenizer = get_tokenizer()
        pad_token_id = None
        try:
            pad_token_id = tokenizer._tokenizer.pad_token_id
        except Exception as e:
            print(f"Failed to get pad_token_id from tokenizer: {e}")

        if pad_token_id is None:
            pad_token_id = 0
        elif isinstance(pad_token_id, int) and pad_token_id < 0:
            pad_token_id = 0
        return PaddingCollator(seq_length=args.train.seq_length, pad_token_id=pad_token_id)