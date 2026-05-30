import multiprocessing
import os
import signal
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
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
from galvatron.core.runtime.datasets.huggingface.collator import get_collate_fn

_STOP_SIGNAL = None
_g_tokenizer = None
_g_eos_token_id = None
_g_text_keys = None
_g_max_seq_len = None

@dataclass
class PrefetchConfig:
    train_ds_path: str
    split: str
    text_keys: Union[str, List[str]]
    shuffle_buffer_size: int
    seed: int
    num_workers: int
    batch_size: int
    imap_chunksize: int
    dp_rank: int
    dp_world_size: int
    seq_length: int

def _tokenize(example: dict) -> List[List[int]]:
    text = get_text_from_example(example, _g_text_keys)
    if text is None:
        return []
    token_ids = tokenize_text(text, _g_tokenizer, _g_eos_token_id)
    return split_into_chunks(token_ids, _g_max_seq_len)


class PrefetchBuffer:
    MODE_PADDING = "padding"
    MODE_PACKING = "packing"

    def __init__(
        self,
        num_entries: int,
        max_batch_size: int,
        seq_length: int,
        mode: str = "padding",
        use_flash_attn: bool = True,
    ):
        self.num_entries = num_entries
        self.max_batch_size = max_batch_size
        self.seq_length = seq_length
        self.mode = mode
        self.use_flash_attn = use_flash_attn

        self.entries = [self._alloc_entry() for _ in range(num_entries)]

    def _alloc_entry(self) -> dict:
        bs, sl = self.max_batch_size, self.seq_length

        if self.mode == self.MODE_PADDING:
            entry = {
                "input_ids": torch.zeros(bs, sl, dtype=torch.long),
                "labels": torch.zeros(bs, sl, dtype=torch.long),
                "loss_masks": torch.zeros(bs, sl, dtype=torch.float),
                "position_ids": torch.zeros(bs, sl, dtype=torch.long),
                "valid_lens": torch.zeros(bs, dtype=torch.long),
            }
        else:  # packing
            max_packed = bs * sl
            entry = {
                "input_ids": torch.zeros(1, max_packed, dtype=torch.long),
                "labels": torch.zeros(1, max_packed, dtype=torch.long),
                "loss_masks": torch.zeros(1, max_packed, dtype=torch.float),
                "cu_seqlens": torch.zeros(bs + 1, dtype=torch.int32),
            }

        for t in entry.values():
            t.share_memory_()
        return entry

    def write_padding(self, entry_idx: int, batch: dict, valid_lens: Tensor) -> int:
        """Copy a PaddingCollator dict into entry_idx"""
        entry = self.entries[entry_idx]
        actual_bs = batch["input_ids"].size(0)
        entry["input_ids"][:actual_bs].copy_(batch["input_ids"])
        entry["labels"][:actual_bs].copy_(batch["labels"])
        entry["loss_masks"][:actual_bs].copy_(batch["loss_masks"])
        entry["position_ids"][:actual_bs].copy_(batch["position_ids"])
        entry["valid_lens"][:actual_bs].copy_(valid_lens)
        return actual_bs

    def read_padding(self, entry_idx: int, actual_bs: int) -> dict:
        """Clone a padding batch from entry_idx"""
        entry = self.entries[entry_idx]
        result = {
            "input_ids": entry["input_ids"][:actual_bs].clone(),
            "labels": entry["labels"][:actual_bs].clone(),
            "loss_masks": entry["loss_masks"][:actual_bs].clone(),
            "position_ids": entry["position_ids"][:actual_bs].clone(),
        }
        if not self.use_flash_attn:
            vl = entry["valid_lens"][:actual_bs].clone()
            sl = self.seq_length
            padding_mask = (
                torch.arange(sl, dtype=torch.long).unsqueeze(0) < vl.unsqueeze(1)
            )
            causal = torch.triu(
                torch.ones(sl, sl, dtype=torch.bool), diagonal=1,
            )
            pad_key = (
                (~padding_mask)
                .unsqueeze(1).unsqueeze(2)
                .expand(actual_bs, 1, sl, sl)
            )
            result["attention_mask"] = (
                pad_key
                | causal.unsqueeze(0).unsqueeze(0).expand(actual_bs, 1, sl, sl)
            )
        return result

    def write_packing(self, entry_idx: int, batch: tuple) -> Tuple[int, int]:
        """Copy a PackingCollator tuple into entry_idx"""
        packed_len = batch["input_ids"].size(1)
        num_seqs = batch["cu_seqlens"].size(0)

        entry = self.entries[entry_idx]
        entry["input_ids"][:, :packed_len].copy_(batch["input_ids"])
        entry["labels"][:, :packed_len].copy_(batch["labels"])
        entry["loss_masks"][:, :packed_len].copy_(batch["loss_masks"])
        entry["cu_seqlens"][:num_seqs].copy_(batch["cu_seqlens"])

        return packed_len, num_seqs

    def read_packing(self, entry_idx: int, packed_len: int, num_seqs: int) -> tuple:
        """Clone a packing batch from entry_idx"""
        entry = self.entries[entry_idx]
        result = {
            "input_ids": entry["input_ids"][:, :packed_len].clone(),
            "labels": entry["labels"][:, :packed_len].clone(),
            "loss_masks": entry["loss_masks"][:, :packed_len].clone(),
            "cu_seqlens": entry["cu_seqlens"][:num_seqs].clone(),
        }
        return result


def _build_chunk_iterator(
    dataset,
    prefetch_config: PrefetchConfig,
) -> Tuple[object, Optional[multiprocessing.pool.Pool]]:
    num_workers = prefetch_config.num_workers
    imap_chunksize = prefetch_config.imap_chunksize
    data_iter = iter(dataset)

    if num_workers <= 0:
        return map(_tokenize, data_iter), None

    fork_ctx = multiprocessing.get_context("fork")
    pool = fork_ctx.Pool(processes=num_workers)
    chunk_iter = pool.imap(
        _tokenize,
        data_iter,
        chunksize=imap_chunksize,
    )
    return chunk_iter, pool


def batch_producer(
    prefetch_config: PrefetchConfig,
    buffer: PrefetchBuffer,
    out_queue,
    stop_event,
    entries_sema,
):
    """Runs in the fork-started producer process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    collate_fn = get_collate_fn()

    dataset = build_hf_streaming_dataset(
        train_ds_path=prefetch_config.train_ds_path,
        split=prefetch_config.split,
        shuffle_buffer_size=prefetch_config.shuffle_buffer_size,
        seed=prefetch_config.seed,
        dp_rank=prefetch_config.dp_rank,
        dp_world_size=prefetch_config.dp_world_size,
    )
    chunk_iter, pool = _build_chunk_iterator(dataset, prefetch_config)

    batch_size = prefetch_config.batch_size
    seq_length = buffer.seq_length

    is_padding = buffer.mode == PrefetchBuffer.MODE_PADDING
    write_idx = 0

    def _flush(samples: List[Tensor]) -> None:
        """Collate *samples*, acquire an entry, write to buffer, enqueue metadata."""
        nonlocal write_idx
        batch = collate_fn(samples)
        entries_sema.acquire()
        if stop_event.is_set():
            return
        entry_idx = write_idx % buffer.num_entries
        if is_padding:
            vl = torch.tensor(
                [min(s.numel(), seq_length) for s in samples],
                dtype=torch.long,
            )
            actual_bs = buffer.write_padding(entry_idx, batch, vl)
            out_queue.put((entry_idx, actual_bs))
        else:
            packed_len, num_seqs = buffer.write_packing(entry_idx, batch)
            out_queue.put((entry_idx, packed_len, num_seqs))
        write_idx += 1

    # main loop
    try:
        samples: List[Tensor] = []

        for chunks in chunk_iter:
            if stop_event.is_set():
                break
            if not chunks:
                continue

            for chunk in chunks:
                samples.append(torch.tensor(chunk, dtype=torch.long))
                if len(samples) == batch_size:
                    _flush(samples)
                    samples = []
                    if stop_event.is_set():
                        break

        if samples and not stop_event.is_set():
            _flush(samples)

    except Exception:
        traceback.print_exc()
    finally:
        out_queue.put(_STOP_SIGNAL)
        if pool is not None:
            pool.terminate()
            pool.join()


class MultiprocessBatchGenerator:
    """Fork-based batch generator backed by a shared-memory prefetch buffer."""

    def __init__(self, prefetch_config: PrefetchConfig, buffer: PrefetchBuffer):
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
            target=batch_producer,
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

        is_padding = self.buffer.mode == PrefetchBuffer.MODE_PADDING

        try:
            while True:
                msg = self._queue.get()
                if msg is _STOP_SIGNAL:
                    break

                if is_padding:
                    entry_idx, actual_bs = msg
                    batch = self.buffer.read_padding(entry_idx, actual_bs)
                else:
                    entry_idx, packed_len, num_seqs = msg
                    batch = self.buffer.read_packing(entry_idx, packed_len, num_seqs)

                self._entries_sema.release()
                yield batch
        finally:
            self._cleanup()

    # lifecycle
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


def build_multiprocess_batch_generator(
    train_ds_path: str,
    split: str = "train",
    text_keys: Union[str, List[str]] = "text",
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
    num_workers: int = 0,
    batch_size: int = 1,
    prefetch_factor: int = 4,
    imap_chunksize: int = 4,
):
    """Build a prefetch-buffer-backed batch generator.
    Must be called after Megatron globals are initialised so that the forked producer inherits them."""

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

    mode = args.data.hf_collator_mode
    use_flash_attn = getattr(args.train, "use_flash_attn", False)

    buffer = PrefetchBuffer(
        num_entries=prefetch_factor,
        max_batch_size=batch_size,
        seq_length=args.train.seq_length,
        mode=mode,
        use_flash_attn=use_flash_attn,
    )

    prefetch_config = PrefetchConfig(
        train_ds_path=train_ds_path,
        split=split,
        text_keys=text_keys,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        num_workers=num_workers,
        batch_size=batch_size,
        imap_chunksize=imap_chunksize,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        seq_length=args.train.seq_length,
    )

    return MultiprocessBatchGenerator(
        prefetch_config=prefetch_config,
        buffer=buffer,
    )
