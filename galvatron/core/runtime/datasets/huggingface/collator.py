from itertools import accumulate
from typing import List, Tuple, Callable

import torch
from torch import Tensor
from galvatron.core.runtime.parallel_state import get_args, get_tokenizer

# from .loss_func import hf_packing_loss_func


# Ignore index for labels (no loss on padding / last position).
IGNORE_INDEX = -100


class PackingCollator:
    """
    Fixed-shape packing collator for HF data. ``batch`` is a flat list of 1D
    token tensors whose length is a multiple of ``chunks``; consecutive
    ``len(batch) // chunks`` samples form one micro-batch chunk and each
    chunk is padded to a multiple of ``align_to`` (default 8) tokens. The
    padding is appended to the last sample's segment so it belongs to the
    same sequence in ``cu_seqlens``, with ``labels = IGNORE_INDEX`` and
    ``loss_masks = 0``.

    Output dict mirrors ``DynamicBatchCollator`` but without
    ``cu_seqlens_chunks`` — chunk boundaries are implicit since every chunk
    holds the same number of real samples.
    """

    def __init__(self, chunks: int, align_to: int = 8):
        assert chunks >= 1, f"chunks must be >= 1, got {chunks}"
        assert align_to >= 1, f"align_to must be >= 1, got {align_to}"
        self.chunks = chunks
        self.align_to = align_to

    def __call__(self, batch: List[Tensor]) -> dict:
        assert len(batch) % self.chunks == 0, (
            f"batch length {len(batch)} must be a multiple of chunks {self.chunks}"
        )
        samples_per_chunk = len(batch) // self.chunks

        input_ids_parts: List[Tensor] = []
        labels_parts: List[Tensor] = []
        loss_masks_parts: List[Tensor] = []
        sample_lengths: List[int] = []

        for c in range(self.chunks):
            chunk_tokens = 0
            chunk_dtype = None
            for i in range(samples_per_chunk):
                t = batch[c * samples_per_chunk + i].view(-1)
                input_ids_parts.append(t)
                labels = torch.empty_like(t, dtype=t.dtype)
                labels[:-1] = t[1:]
                labels[-1] = IGNORE_INDEX
                labels_parts.append(labels)
                loss_masks_parts.append(torch.ones_like(t, dtype=torch.float32))
                sample_lengths.append(t.numel())
                chunk_tokens += t.numel()
                chunk_dtype = t.dtype

            # Pad this chunk up to a multiple of ``align_to`` tokens, appended
            # to the last sample's segment.
            remainder = chunk_tokens % self.align_to
            if remainder != 0 and chunk_tokens > 0:
                pad_len = self.align_to - remainder
                input_ids_parts.append(torch.zeros(pad_len, dtype=chunk_dtype))
                labels_parts.append(torch.full((pad_len,), IGNORE_INDEX, dtype=chunk_dtype))
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
    packed into one micro-batch and then padded so that its token count is a
    multiple of ``align_to`` (default 8); the padding is appended to the last
    sample's segment in ``cu_seqlens``, with ``loss_masks = 0`` and
    ``labels = IGNORE_INDEX`` to keep it out of the loss.

    Output dict:

    - ``input_ids`` / ``labels`` / ``loss_masks``: ``(1, total_T)`` tensors,
      where ``total_T`` is the sum of per-bucket aligned token counts.
    - ``cu_seqlens``: ``List[int]`` of length ``N+1``, every inner-sample
      boundary (including the per-bucket pad slot when present) across the
      whole pack.
    - ``cu_seqlens_chunks``: ``List[int]`` of length ``num_microbatches+1``,
      indices into ``cu_seqlens`` marking micro-batch ends.

    ``cu_seqlens`` variants are kept as plain Python lists to stay queue-
    friendly (cheap pickle, no torch shared-memory churn). Convert to int32
    tensors on the consumer side if downstream code needs them.
    """

    def __init__(self, align_to: int = 8):
        assert align_to >= 1, f"align_to must be >= 1, got {align_to}"
        self.align_to = align_to

    def __call__(self, buckets: List[List[Tensor]]) -> dict:
        input_ids_parts: List[Tensor] = []
        labels_parts: List[Tensor] = []
        loss_masks_parts: List[Tensor] = []
        sample_lengths: List[int] = []
        chunk_sample_offsets: List[int] = [0]

        for bucket in buckets:
            bucket_tokens = 0
            bucket_dtype = None
            for t in bucket:
                input_ids_parts.append(t)
                labels = torch.empty_like(t, dtype=t.dtype)
                labels[:-1] = t[1:]
                labels[-1] = IGNORE_INDEX
                labels_parts.append(labels)
                loss_masks_parts.append(torch.ones_like(t, dtype=torch.float32))
                sample_lengths.append(t.numel())
                bucket_tokens += t.numel()
                bucket_dtype = t.dtype

            # Pad the bucket up to a multiple of ``align_to`` tokens, appended
            # to the last sample's segment.
            remainder = bucket_tokens % self.align_to
            if remainder != 0 and bucket_tokens > 0:
                pad_len = self.align_to - remainder
                input_ids_parts.append(torch.zeros(pad_len, dtype=bucket_dtype))
                labels_parts.append(torch.full((pad_len,), IGNORE_INDEX, dtype=bucket_dtype))
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