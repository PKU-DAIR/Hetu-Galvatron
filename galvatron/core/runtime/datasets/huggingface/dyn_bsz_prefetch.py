"""Dyn-bsz prefetch pipeline for HF data.

Mirrors the producer/consumer architecture in ``prefetch_strategy.py`` but is
intentionally standalone — no shared ``PrefetchBuffer`` / ``PrefetchConfig`` /
``MultiprocessBatchGenerator``. The shapes and semantics here only make sense
for dyn_bsz, so they live in their own dedicated types.

Shared-memory layout per ring-buffer entry: only ``input_ids`` and ``labels``
(both large, ``(1, num_microbatches * token_capacity)``). The ``cu_seqlens``
and ``cu_seqlens_chunks`` arrays are small (a few KB at most), so they travel
through the multiprocessing queue alongside the metadata; this avoids
pre-allocating a loose ``max_inner_samples`` upper bound.
"""

import multiprocessing
import os
import signal
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.multiprocessing as tmp
from torch import Tensor

from galvatron.core.runtime.datasets.huggingface.utils import (
    get_text_from_example,
    tokenize_text,
    split_into_chunks,
)
from galvatron.core.runtime.parallel_state import (
    get_args,
    get_tokenizer,
    get_vocab_dp_world_size as get_data_parallel_world_size,
    get_vocab_dp_rank as get_data_parallel_rank,
)
from galvatron.core.runtime.datasets.huggingface.dataset import build_hf_streaming_dataset
from galvatron.core.runtime.datasets.huggingface.collator import (
    IGNORE_INDEX,
    get_collate_fn,
)

_STOP_SIGNAL = None

# Globals consumed by ``_tokenize`` in the forked producer / pool workers.
_g_tokenizer = None
_g_eos_token_id = None
_g_text_keys = None
_g_max_seq_len = None


def _tokenize(example: dict) -> List[List[int]]:
    text = get_text_from_example(example, _g_text_keys)
    if text is None:
        return []
    token_ids = tokenize_text(text, _g_tokenizer, _g_eos_token_id)
    return split_into_chunks(token_ids, _g_max_seq_len)


@dataclass
class DynBszPrefetchConfig:
    train_ds_path: str
    split: str
    text_keys: Union[str, List[str]]
    shuffle_buffer_size: int
    seed: int
    num_workers: int
    imap_chunksize: int
    dp_rank: int
    dp_world_size: int
    seq_length: int
    token_capacity: int
    num_microbatches: int
    # Minimum #segments in the long-lived buffer before we start draining
    # micro-batches out of it (in conjunction with ``token >= token_capacity``).
    # Bigger => better first-fit packing, longer warm-up to first batch.
    buffer_ready_threshold: int = 0


class DynBszPrefetchBuffer:
    """Shared-memory ring buffer for dyn_bsz.

    Only ``input_ids`` and ``labels`` are pre-allocated (the big tensors).
    Each entry holds one global step worth of tokens:
    ``(1, num_microbatches * token_capacity)``.

    ``cu_seqlens`` and ``cu_seqlens_chunks`` are NOT in the buffer; the
    producer ships them through the queue, since they are small and have a
    sample-count-dependent shape that we'd rather not bound a priori.
    """

    def __init__(self, num_entries: int, token_capacity: int, num_microbatches: int):
        self.num_entries = num_entries
        self.token_capacity = token_capacity
        self.num_microbatches = num_microbatches

        max_packed = token_capacity * num_microbatches
        self.entries = []
        for _ in range(num_entries):
            entry = {
                "input_ids": torch.zeros(1, max_packed, dtype=torch.long),
                "labels": torch.zeros(1, max_packed, dtype=torch.long),
                "loss_masks": torch.zeros(1, max_packed, dtype=torch.float32),
            }
            for t in entry.values():
                t.share_memory_()  # shared memory: producer subprocess writes, main process reads
            self.entries.append(entry)

    def write(self, entry_idx: int, input_ids: Tensor, labels: Tensor, loss_masks: Tensor) -> int:
        entry = self.entries[entry_idx]
        packed_len = input_ids.size(1)
        assert packed_len <= entry["input_ids"].size(1), (
            f"dyn_bsz packed_len {packed_len} exceeds buffer capacity "
            f"{entry['input_ids'].size(1)}"
        )
        entry["input_ids"][:, :packed_len].copy_(input_ids)
        entry["labels"][:, :packed_len].copy_(labels)
        entry["loss_masks"][:, :packed_len].copy_(loss_masks)
        return packed_len

    def read(self, entry_idx: int, packed_len: int) -> Tuple[Tensor, Tensor, Tensor]:
        entry = self.entries[entry_idx]
        return (
            entry["input_ids"][:, :packed_len].clone(),
            entry["labels"][:, :packed_len].clone(),
            entry["loss_masks"][:, :packed_len].clone(),
        )


def _build_segments_iter(
    dataset,
    prefetch_config: DynBszPrefetchConfig,
) -> Tuple[object, Optional[multiprocessing.pool.Pool]]:
    """Parallel tokenizer pool; yields per-doc ``List[List[int]]`` (segments)."""
    num_workers = prefetch_config.num_workers
    imap_chunksize = prefetch_config.imap_chunksize
    data_iter = iter(dataset)

    if num_workers <= 0:
        return map(_tokenize, data_iter), None

    # fork (not spawn) so workers inherit the _g_* globals _tokenize reads.
    fork_ctx = multiprocessing.get_context("fork")
    pool = fork_ctx.Pool(processes=num_workers)
    segments_iter = pool.imap(_tokenize, data_iter, chunksize=imap_chunksize)
    return segments_iter, pool


def dyn_bsz_batch_producer(
    prefetch_config: DynBszPrefetchConfig,
    buffer: DynBszPrefetchBuffer,
    out_queue,
    stop_event,
    entries_sema,
):
    """Producer: greedy-pack tokenizer chunks into ``num_microbatches`` buckets
    (each ``<= token_capacity`` tokens), collate, write to shared-memory entry,
    enqueue ``(entry_idx, packed_len, cu_seqlens, cu_seqlens_chunks)``. The
    loss func is constant (``hf_packing_loss_func``) so the consumer attaches
    it directly without going through the queue.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    collate_fn = get_collate_fn()  # expects args.data.hf_collator_mode == "dyn_bsz"

    dataset = build_hf_streaming_dataset(
        train_ds_path=prefetch_config.train_ds_path,
        split=prefetch_config.split,
        shuffle_buffer_size=prefetch_config.shuffle_buffer_size,
        seed=prefetch_config.seed,
        dp_rank=prefetch_config.dp_rank,
        dp_world_size=prefetch_config.dp_world_size,
    )
    segments_iter, pool = _build_segments_iter(dataset, prefetch_config)

    token_capacity = prefetch_config.token_capacity
    num_microbatches = prefetch_config.num_microbatches
    buffer_ready_threshold = prefetch_config.buffer_ready_threshold
    write_idx = 0

    # Long-lived segment buffer (shared across docs/micro-batches).
    # Items here are 1D token tensors, with cumulative token count tracked
    # in ``segment_buffer_tokens`` for an O(1) "is buffer heavy?" check.
    segment_buffer: List[Tensor] = []
    segment_buffer_tokens = 0

    # Accumulator for the current global step. Each entry is one micro-batch
    # (a list of segments greedily extracted from segment_buffer).
    buckets: List[List[Tensor]] = []

    def _extract_micro_batch() -> List[Tensor]:
        """First-fit greedy: walk segment_buffer in order, pick segments that
        fit into ``token_capacity``; segments that don't fit are left behind
        for later micro-batches (this is what makes the long-lived buffer pay
        off vs. simple per-doc bucketing).
        """
        nonlocal segment_buffer_tokens
        picked: List[Tensor] = []
        total = 0
        remove_idx: List[int] = []
        for idx, seg in enumerate(segment_buffer):
            seg_len = seg.numel()
            if total + seg_len > token_capacity:
                continue
            picked.append(seg)
            total += seg_len
            remove_idx.append(idx)
            if total >= token_capacity:
                break
        remove_set = set(remove_idx)
        segment_buffer[:] = [s for i, s in enumerate(segment_buffer) if i not in remove_set]
        segment_buffer_tokens -= total
        return picked

    def _flush(buckets: List[List[Tensor]]) -> None:
        nonlocal write_idx
        batch = collate_fn(buckets)
        entries_sema.acquire()
        if stop_event.is_set():
            return
        entry_idx = write_idx % buffer.num_entries
        packed_len = buffer.write(
            entry_idx,
            batch["input_ids"],
            batch["labels"],
            batch["loss_masks"],
        )
        # cu_seqlens / cu_seqlens_chunks are already List[int] from the
        # collator, so they pickle cheaply through the queue (no torch
        # shared-memory churn).
        out_queue.put((
            entry_idx,
            packed_len,
            batch["cu_seqlens"],
            batch["cu_seqlens_chunks"],
        ))
        write_idx += 1

    try:
        # ``doc_segments``: one HF sample (a single ``str`` doc) after
        # tokenize + split_into_chunks => List[List[int]]; each ``segment``
        # is a List[int] (token ids) of length <= seq_length.
        for doc_segments in segments_iter:
            if stop_event.is_set():
                break
            if not doc_segments:
                continue
            for segment in doc_segments:
                token_tensor = torch.tensor(segment, dtype=torch.long)
                segment_buffer.append(token_tensor)
                segment_buffer_tokens += token_tensor.numel()

                # Extract one micro-batch if the buffer is "heavy enough":
                # both segment-count threshold (for packing quality) and
                # token-count threshold (so the picked micro-batch can fill
                # token_capacity) must be satisfied.
                if (
                    len(segment_buffer) >= buffer_ready_threshold
                    and segment_buffer_tokens >= token_capacity
                ):
                    micro_batch = _extract_micro_batch()
                    if micro_batch:
                        buckets.append(micro_batch)
                        if len(buckets) == num_microbatches:
                            _flush(buckets)
                            buckets = []
                            if stop_event.is_set():
                                break
        # Trailing partial step (incomplete num_microbatches or leftover
        # segments in segment_buffer) is dropped.
    except Exception:
        traceback.print_exc()
    finally:
        out_queue.put(_STOP_SIGNAL)
        if pool is not None:
            pool.terminate()
            pool.join()


class DynBszMultiprocessBatchGenerator:
    """Fork-based dyn_bsz batch generator."""

    def __init__(self, prefetch_config: DynBszPrefetchConfig, buffer: DynBszPrefetchBuffer):
        self.prefetch_config = prefetch_config
        self.buffer = buffer
        self._process: Optional[multiprocessing.Process] = None
        self._queue = None
        self._stop_event = None
        self._entries_sema = None
        self._closed = False
        self._prev_sigterm = None

    def __iter__(self):
        self._closed = False
        ctx = tmp.get_context("fork")

        self._queue = ctx.SimpleQueue()
        self._stop_event = ctx.Event()
        self._entries_sema = ctx.BoundedSemaphore(self.buffer.num_entries)

        self._process = ctx.Process(
            target=dyn_bsz_batch_producer,
            args=(
                self.prefetch_config,
                self.buffer,
                self._queue,
                self._stop_event,
                self._entries_sema,
            ),
            daemon=False,
        )
        self._process.start()
        self._install_sigterm_handler()

        try:
            while True:
                msg = self._queue.get()
                if msg is _STOP_SIGNAL:
                    break

                entry_idx, packed_len, cu_seqlens_list, cu_seqlens_chunks_list = msg
                input_ids, labels, loss_masks = self.buffer.read(entry_idx, packed_len)
                self._entries_sema.release()
                yield {
                    'input_ids': input_ids,
                    'cu_seqlens': torch.tensor(cu_seqlens_list, dtype=torch.int32),
                    'cu_seqlens_chunks': torch.tensor(cu_seqlens_chunks_list, dtype=torch.int32),
                    'labels': labels,
                    'loss_masks': loss_masks,
                }
        finally:
            self._cleanup()

    def _install_sigterm_handler(self):
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            self._cleanup()
            prev = self._prev_sigterm
            if callable(prev) and prev not in (signal.SIG_IGN, signal.SIG_DFL):
                prev(signum, frame)
            else:
                raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _handler)

    def _cleanup(self):
        if self._closed:
            return
        self._closed = True

        if self._prev_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            except (OSError, ValueError):
                pass

        if self._stop_event is not None:
            self._stop_event.set()
        if self._entries_sema is not None:
            for _ in range(self.buffer.num_entries):
                try:
                    self._entries_sema.release()
                except ValueError:
                    break
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()

    def stop(self):
        self._cleanup()


def build_dyn_bsz_batch_generator(
    train_ds_path: str,
    split: str = "train",
    text_keys: Union[str, List[str]] = "text",
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
    num_workers: int = 0,
    prefetch_factor: int = 4,
    imap_chunksize: int = 4,
    buffer_ready_threshold: Optional[int] = None,
):
    """Build a dyn_bsz prefetch generator.

    ``token_capacity = global_batch_size // dp_world_size // chunks * seq_length``,
    mirroring ``dummy_dataset.dyn_bsz.get_dynamic_batch_data_iterator``.
    """
    args = get_args()
    tokenizer = get_tokenizer()
    dp_rank = get_data_parallel_rank()
    dp_world_size = get_data_parallel_world_size()

    global _g_tokenizer, _g_eos_token_id, _g_text_keys, _g_max_seq_len
    _g_tokenizer = tokenizer
    try:
        _g_eos_token_id = tokenizer.eod
    except Exception as e:
        print(f"Failed to get eos_token_id from tokenizer: {e}")
    _g_text_keys = text_keys
    _g_max_seq_len = args.train.seq_length

    global_batch_size = args.train.global_batch_size
    num_microbatches = args.train.chunks
    seq_length = args.train.seq_length

    assert global_batch_size % dp_world_size == 0, (
        f"global_batch_size ({global_batch_size}) must be divisible by dp_world_size ({dp_world_size})"
    )
    per_dp_batch = global_batch_size // dp_world_size
    assert per_dp_batch % num_microbatches == 0, (
        f"global_batch_size / dp_world_size ({per_dp_batch}) must be divisible by chunks ({num_microbatches})"
    )
    token_capacity = (per_dp_batch // num_microbatches) * seq_length

    if buffer_ready_threshold is None:
        # Default: 2x the expected segment count for one global step
        # (per_dp_batch ≈ #segments per global step when most segments are
        # near seq_length). Tune higher for better packing, lower for less
        # warm-up latency.
        buffer_ready_threshold = max(1, 2 * per_dp_batch)

    buffer = DynBszPrefetchBuffer(
        num_entries=prefetch_factor,
        token_capacity=token_capacity,
        num_microbatches=num_microbatches,
    )

    prefetch_config = DynBszPrefetchConfig(
        train_ds_path=train_ds_path,
        split=split,
        text_keys=text_keys,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        num_workers=num_workers,
        imap_chunksize=imap_chunksize,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        seq_length=seq_length,
        token_capacity=token_capacity,
        num_microbatches=num_microbatches,
        buffer_ready_threshold=buffer_ready_threshold,
    )

    return DynBszMultiprocessBatchGenerator(
        prefetch_config=prefetch_config,
        buffer=buffer,
    )


