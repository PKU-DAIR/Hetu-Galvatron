import torch
from typing import List, Tuple, Callable
from torch import Tensor
from galvatron.core.runtime.utils.utils import average_losses_across_data_parallel_group, average_losses_across_context_parallel_group
from galvatron.core.runtime.parallel_state import get_args, get_tokenizer


# Ignore index for labels (no loss on padding / last position).
IGNORE_INDEX = -100

def hf_packing_loss_func(labels_list, output_tensor_list):
    loss = output_tensor_list[0].float()
    loss_mask = torch.ones_like(loss).float()
    loss = torch.sum(loss.view(-1) * loss_mask.view(-1)) / loss_mask.sum().clamp(min=1)
    averaged_loss = average_losses_across_data_parallel_group([loss])
    averaged_loss = average_losses_across_context_parallel_group(averaged_loss)
    return loss, averaged_loss.squeeze().clone().detach()
    
class PackingCollator:
    def __call__(self, batch: List[Tensor]) -> Tuple[Tensor, dict, Callable]:
        # batch: list of 1D tensors (input_ids per chunk)
        input_ids_list = [t.view(-1) for t in batch]
        labels_list = []
        for t in input_ids_list:
            labels = torch.empty_like(t, dtype=t.dtype)
            labels[:-1] = t[1:]
            labels[-1] = IGNORE_INDEX
            labels_list.append(labels)

        cu_seqlens = torch.zeros(
            len(batch) + 1, dtype=torch.int32
        )
        cu_seqlens[0] = 0
        for i in range(1, len(cu_seqlens)):
            cu_seqlens[i] = cu_seqlens[i - 1] + input_ids_list[i - 1].numel()

        input_ids = torch.cat(input_ids_list, dim=0).view(1, -1).contiguous()
        labels = torch.cat(labels_list, dim=0).view(1, -1).contiguous()
        return (
            input_ids,
            {
                "cu_seqlens": cu_seqlens,
                "attention_mask": None,
                "labels": labels,
                "rotary_embedding": None,
            },
            hf_packing_loss_func,
        )

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

        loss_mask = (labels != IGNORE_INDEX).float()
        position_ids = torch.arange(
            self.seq_length, dtype=torch.long
        ).unsqueeze(0).expand(input_ids.size(0), -1).contiguous()

        result = {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        if not self.use_flash_attn:
            batch_size, seq_length = input_ids.size()
            causal = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)
            pad_key_mask = (~padding_mask).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_length, seq_length)
            result["attention_mask"] = pad_key_mask | causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        return result

def get_collate_fn():
    args = get_args()
    if args.data.hf_collator_mode == "packing":
        return PackingCollator()
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