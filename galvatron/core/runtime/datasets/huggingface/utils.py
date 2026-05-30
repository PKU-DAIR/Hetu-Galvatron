from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.distributed as dist

import galvatron.core.runtime.parallel_state as parallel_state


def get_data_files(data_paths: Union[str, List[str]]):
    data_files = []
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    for data_path in data_paths:
        if data_path.startswith("hdfs://"):
            raise NotImplementedError("HDFS dataset path is not supported yet.")
        else:
            data_path = Path(data_path)
            if data_path.is_dir():
                data_files.extend([str(file_path) for file_path in data_path.iterdir() if file_path.is_file()])
            elif data_path.is_file():
                data_files.append(str(data_path))
            else:
                raise FileNotFoundError(f"Dataset {data_path} not exists.")
    if not data_files:
        raise ValueError(f"No data files found for data_paths={data_paths!r}")
    
    file_extension = Path(data_files[0]).suffix.lstrip(".").lower()
    if file_extension == "jsonl":
        file_extension = "json"
    if file_extension not in ["parquet", "json", "csv", "arrow"]:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    return data_files, file_extension


def get_text_from_example(example: dict, text_keys: Union[str, List[str]]):
    if isinstance(text_keys, str):
        return example.get(text_keys)
    for key in text_keys:
        if key in example:
            return example[key]
    return None


def tokenize_text(text, tokenizer, eos_token_id) -> List[int]:
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        # Prefer tokenizer-native defaults (HF fast tokenizer path),
        # and only fallback to legacy bos/eos kwargs when required.
        try:
            ids = tokenizer.tokenize(text)
        except TypeError:
            ids = tokenizer.tokenize(text, bos=False, eos=False)
        if not isinstance(ids, list):
            ids = list(ids)
    if eos_token_id is not None:
        ids = ids + [eos_token_id]
    return ids


def split_into_chunks(token_ids: List[int], chunk_len: int):
    chunks = []
    for i in range(0, len(token_ids), chunk_len):
        chunk = token_ids[i : i + chunk_len]
        if chunk:
            chunks.append(chunk)
    return chunks


def get_batch_on_this_tp(data_iterator) -> Dict[str, torch.Tensor]:
    """TP broadcast for HF batches.

    Mirrors ``dummy_dataset.utils.get_text_batch_on_this_tp_rank``: handles
    both packing-style batches (with ``cu_seqlens``) and dyn_bsz batches
    (with both ``cu_seqlens`` and ``cu_seqlens_chunks``), and falls back to
    plain padded batches when neither is present.

    Metadata is broadcast as ``[B, T, num_samples, num_chunks_plus_1]``
    where ``num_samples == 0`` signals "no cu_seqlens" and
    ``num_chunks_plus_1 == 0`` signals "no cu_seqlens_chunks". Only PP
    first/last + vocab-TP rank 0 hold a real ``data_iterator``; other
    ranks pass ``None`` and just receive the broadcast.
    """
    assert (
        parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage()
    ), "get_batch_on_this_tp should only be called on TP ranks of the first or last PP stage"

    tp_group = parallel_state.get_vocab_tp_sp_comm_group().group
    tp_rank = parallel_state.get_vocab_tp_sp_rank()
    src_rank = parallel_state.get_vocab_tp_sp_src_rank()

    if tp_rank == 0:
        batch = next(data_iterator)
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)
        loss_masks = batch["loss_masks"].cuda(non_blocking=True)
        B, T = input_ids.shape

        cu_seqlens = batch.get("cu_seqlens")
        if cu_seqlens is None:
            num_samples = 0
        else:
            cu_seqlens = cu_seqlens.cuda(non_blocking=True)
            num_samples = cu_seqlens.numel() - 1

        cu_seqlens_chunks = batch.get("cu_seqlens_chunks")
        if cu_seqlens_chunks is None:
            num_chunks_plus_1 = 0
        else:
            cu_seqlens_chunks = cu_seqlens_chunks.cuda(non_blocking=True)
            num_chunks_plus_1 = cu_seqlens_chunks.numel()

        shape_meta = torch.tensor(
            [B, T, num_samples, num_chunks_plus_1], dtype=torch.long
        ).cuda(non_blocking=True)
    else:
        shape_meta = torch.empty(4, dtype=torch.long).cuda(non_blocking=True)

    dist.broadcast(shape_meta, src=src_rank, group=tp_group)
    B, T, num_samples, num_chunks_plus_1 = shape_meta.tolist()
    has_cu_seqlens = num_samples > 0
    has_cu_seqlens_chunks = num_chunks_plus_1 > 0

    if tp_rank != 0:
        input_ids = torch.empty((B, T), dtype=torch.long).cuda(non_blocking=True)
        labels = torch.empty((B, T), dtype=torch.long).cuda(non_blocking=True)
        loss_masks = torch.empty((B, T), dtype=torch.float32).cuda(non_blocking=True)
        cu_seqlens = (
            torch.empty(num_samples + 1, dtype=torch.int32).cuda(non_blocking=True)
            if has_cu_seqlens
            else None
        )
        cu_seqlens_chunks = (
            torch.empty(num_chunks_plus_1, dtype=torch.int32).cuda(non_blocking=True)
            if has_cu_seqlens_chunks
            else None
        )

    dist.broadcast(input_ids, src=src_rank, group=tp_group)
    dist.broadcast(labels, src=src_rank, group=tp_group)
    dist.broadcast(loss_masks, src=src_rank, group=tp_group)
    if has_cu_seqlens:
        dist.broadcast(cu_seqlens, src=src_rank, group=tp_group)
    if has_cu_seqlens_chunks:
        dist.broadcast(cu_seqlens_chunks, src=src_rank, group=tp_group)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_masks,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_chunks": cu_seqlens_chunks,
    }


def get_batch_on_this_cp_rank(batch):
    pass