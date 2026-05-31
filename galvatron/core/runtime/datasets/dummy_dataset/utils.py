from typing import Dict, List

import torch
import torch.distributed as dist

import galvatron.core.runtime.parallel_state as parallel_state


def chunk_batch(batch: Dict[str, torch.Tensor], chunks: int) -> List[Dict[str, torch.Tensor]]:
    cu_seqlens = batch["cu_seqlens"]

    if cu_seqlens is None:
        # padding mode: slice along the batch dim.
        B, _ = batch["input_ids"].shape
        assert B % chunks == 0, f"Batch size {B} must be divisible by chunks {chunks}"

        micro_batches = []
        for i in range(chunks):
            start = i * (B // chunks)
            end = (i + 1) * (B // chunks)
            micro_batches.append({
                "input_ids": batch["input_ids"][start:end],
                "labels": batch["labels"][start:end],
                "loss_masks": batch["loss_masks"][start:end],
                "cu_seqlens": None,
            })
        return micro_batches

    # pack mode: batch['input_ids'] is (1, total_T); split by sample boundaries in cu_seqlens.
    num_samples = cu_seqlens.numel() - 1
    assert num_samples % chunks == 0, f"Number of samples {num_samples} must be divisible by chunks {chunks}"
    samples_per_chunk = num_samples // chunks

    micro_batches = []
    for i in range(chunks):
        s_start = i * samples_per_chunk
        s_end = (i + 1) * samples_per_chunk
        t_start = int(cu_seqlens[s_start].item())
        t_end = int(cu_seqlens[s_end].item())
        micro_cu_seqlens = (cu_seqlens[s_start : s_end + 1] - cu_seqlens[s_start]).contiguous()
        micro_batches.append({
            "input_ids": batch["input_ids"][:, t_start:t_end],
            "labels": batch["labels"][:, t_start:t_end],
            "loss_masks": batch["loss_masks"][:, t_start:t_end],
            "cu_seqlens": micro_cu_seqlens,
        })
    return micro_batches


def text_loss_func(label:List, ce_loss: List[torch.Tensor], micro_loss_masks: List[torch.Tensor],
                   effective_tokens: int = None, chunks: int = 1):
    loss_mask = micro_loss_masks[0]
    micro_loss_masks.pop(0)

    ce_loss = ce_loss[0] # adapt to the original signature of loss_func which returns a list of ce_loss for each microbatch, but actually we only have one microbatch's ce_loss here since chunking is done outside in the training loop
    ce_loss = ce_loss.view(-1).float()

    loss_mask = loss_mask.view(-1).float()
    if effective_tokens is not None:
        # effective_tokens is per-micro-batch. Outside the training loop the
        # micro-batch loss is divided by ``chunks`` for gradient accumulation,
        # so we multiply by chunks here to recover the true per-token loss.
        local_loss = torch.sum(ce_loss * loss_mask) / effective_tokens * chunks
    else:
        local_loss = torch.sum(ce_loss * loss_mask) / loss_mask.sum()  # token level loss

    ce_sum = torch.sum(ce_loss * loss_mask).detach()
    ce_count = loss_mask.sum().detach()
    stas = torch.stack([ce_sum, ce_count])
    torch.distributed.all_reduce(
        stas, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_vocab_dp_comm_group().group
    )
    loss_reduced = stas[0] / stas[1]

    return local_loss, loss_reduced


def get_text_batch_on_this_tp_rank(data_iterator):
    """
    TP broadcast for text batches. Handles three batch shapes:

    - padding mode: ``cu_seqlens`` is None.
    - pack mode (fixed micro-batch count): ``cu_seqlens`` present, no
      ``cu_seqlens_chunks``.
    - dyn_bsz pack mode: both ``cu_seqlens`` and ``cu_seqlens_chunks`` present;
      ``cu_seqlens_chunks`` marks micro-batch (chunk) boundaries into
      ``cu_seqlens``.

    Metadata is broadcast as a 4-tuple ``[B, T, num_samples, num_chunks_plus_1]``,
    where ``num_samples == 0`` signals padding mode and
    ``num_chunks_plus_1 == 0`` signals "no cu_seqlens_chunks".
    """
    # Middle PP stages don't need the raw batch.
    assert (
        parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage()
    ), "get_text_batch_on_this_tp_rank should only be called on TP ranks of the first or last PP stage"

    tp_group = parallel_state.get_vocab_tp_sp_comm_group().group
    tp_rank = parallel_state.get_vocab_tp_sp_rank()
    src_rank = parallel_state.get_vocab_tp_sp_src_rank()  # global rank of TP-SP rank 0

    # Step 1: src rank reads data and packs shape metadata.
    if tp_rank == 0:
        batch = next(data_iterator)
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)
        loss_masks = batch["loss_masks"].cuda(non_blocking=True)
        B, T = batch["input_ids"].shape
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

    # Step 2: broadcast shape.
    dist.broadcast(shape_meta, src=src_rank, group=tp_group)
    B, T, num_samples, num_chunks_plus_1 = shape_meta.tolist()
    has_cu_seqlens = num_samples > 0
    has_cu_seqlens_chunks = num_chunks_plus_1 > 0

    # Step 3: non-src ranks allocate receive buffers.
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

    # Step 4: broadcast data tensors.
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


def get_text_batch_on_this_cp_rank(batch):
    """Slice batch along sequence dimension for context parallel group.

    For packed (varlen) batches with ``cu_seqlens``, applies zigzag CP
    splitting per-sample and keeps global ``cu_seqlens`` unchanged (zigzag
    ring attention handles the division internally).

    For non-packed batches, applies uniform zigzag CP splitting.
    """
    cp_size = parallel_state.get_vocab_cp_world_size()
    if cp_size <= 1:
        return batch

    cp_rank = parallel_state.get_vocab_cp_rank()
    cu_seqlens = batch.get("cu_seqlens")

    if cu_seqlens is not None:
        # --- Packed (varlen) mode: zigzag per-sample ---
        device = batch["input_ids"].device

        indices_list = []
        for i in range(len(cu_seqlens) - 1):
            N = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
            offset = cu_seqlens[i].item()
            seg = N // (2 * cp_size)
            front = torch.arange(cp_rank * seg, (cp_rank + 1) * seg, device=device)
            back = torch.arange(N - (cp_rank + 1) * seg, N - cp_rank * seg, device=device)
            indices_list.append(torch.cat([front, back]) + offset)
        local_indices = torch.cat(indices_list)

        for key in ("input_ids", "labels", "loss_masks"):
            if key in batch and batch[key] is not None:
                batch[key] = batch[key][..., local_indices].contiguous()

    else:
        # --- Non-packed mode: uniform zigzag CP split ---
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != "attention_mask" else 2
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor(
                    [cp_rank, (2 * cp_size - cp_rank - 1)],
                    device="cpu", pin_memory=True,
                ).cuda(non_blocking=True)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                batch[key] = val

    return batch
