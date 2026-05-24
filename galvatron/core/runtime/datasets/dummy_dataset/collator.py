from typing import Dict, List

import torch


def text_collate_fn(batch: List[Dict[str, torch.Tensor]], collate_mode: str = "padding") -> Dict[str, torch.Tensor]:
    if collate_mode == "padding":
        input_ids = torch.stack([sample["input_ids"] for sample in batch], dim=0)    # (B, seq_length)
        labels = torch.stack([sample["labels"] for sample in batch], dim=0)          # (B, seq_length)
        loss_masks = torch.stack([sample["loss_masks"] for sample in batch], dim=0)  # (B, seq_length)
        cu_seqlens = None
    elif collate_mode == "pack":
        input_ids = torch.cat([sample["input_ids"] for sample in batch], dim=0).unsqueeze(0)    # (1, total_T)
        labels = torch.cat([sample["labels"] for sample in batch], dim=0).unsqueeze(0)          # (1, total_T)
        loss_masks = torch.cat([sample["loss_masks"] for sample in batch], dim=0).unsqueeze(0)  # (1, total_T)
        lengths = torch.tensor([s["input_ids"].shape[0] for s in batch], dtype=torch.int32)
        cu_seqlens = torch.zeros(len(batch) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)  # (num_samples + 1,)
    else:
        raise ValueError(f"collate_mode must be 'padding' or 'pack', got {collate_mode!r}")

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_masks,
        "cu_seqlens": cu_seqlens,
    }
