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


def text_loss_func(label:List, ce_loss: List[torch.Tensor], micro_loss_masks: List[torch.Tensor]):
    loss_mask = micro_loss_masks[0]
    micro_loss_masks.pop(0)

    ce_loss = ce_loss[0] # adapt to the original signature of loss_func which returns a list of ce_loss for each microbatch, but actually we only have one microbatch's ce_loss here since chunking is done outside in the training loop
    ce_loss = ce_loss.view(-1).float()
    
    # import torch
    # rank = torch.distributed.get_rank()
    # print(f'Rank {rank}, ce.loss.shape: {ce_loss.shape}, loss_mask.shape: {loss_mask.shape}', flush=True)

    loss_mask = loss_mask.view(-1).float()
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
    # Middle PP stages don't need the raw batch.
    assert (
        parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage()
    ), "get_text_batch_on_this_tp_rank should only be called on TP ranks of the first or last PP stage"

    tp_group = parallel_state.get_vocab_tp_sp_comm_group().group
    tp_rank = parallel_state.get_vocab_tp_sp_rank()
    src_rank = parallel_state.get_vocab_tp_sp_src_rank()  # global rank of TP-SP rank 0

    # Step 1: src rank reads data and packs shape metadata.
    # shape_meta = [B, T, num_samples]; num_samples == 0 signals padding mode (no cu_seqlens).
    if tp_rank == 0:
        batch = next(data_iterator)
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)
        loss_masks = batch["loss_masks"].cuda(non_blocking=True)
        B, T = batch["input_ids"].shape
        cu_seqlens = batch["cu_seqlens"]
        if cu_seqlens is None:
            num_samples = 0
        else:
            cu_seqlens = cu_seqlens.cuda(non_blocking=True)
            num_samples = cu_seqlens.numel() - 1
        shape_meta = torch.tensor([B, T, num_samples], dtype=torch.long).cuda(non_blocking=True)
    else:
        shape_meta = torch.empty(3, dtype=torch.long).cuda(non_blocking=True)

    # Step 2: broadcast shape.
    dist.broadcast(shape_meta, src=src_rank, group=tp_group)
    B, T, num_samples = shape_meta.tolist()
    has_cu_seqlens = num_samples > 0

    # Step 3: non-src ranks allocate receive buffers.
    if tp_rank != 0:
        input_ids = torch.empty((B, T), dtype=torch.long).cuda(non_blocking=True)
        labels = torch.empty((B, T), dtype=torch.long).cuda(non_blocking=True)
        loss_masks = torch.empty((B, T), dtype=torch.float32).cuda(non_blocking=True)
        if has_cu_seqlens:
            cu_seqlens = torch.empty(num_samples + 1, dtype=torch.int32).cuda(non_blocking=True)
        else:
            cu_seqlens = None

    # Step 4: broadcast data tensors.
    dist.broadcast(input_ids, src=src_rank, group=tp_group)
    dist.broadcast(labels, src=src_rank, group=tp_group)
    dist.broadcast(loss_masks, src=src_rank, group=tp_group)
    if has_cu_seqlens:
        dist.broadcast(cu_seqlens, src=src_rank, group=tp_group)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_masks,
        "cu_seqlens": cu_seqlens,
    }


def get_text_batch_on_this_cp_rank(batch):
    return batch  # TODO
