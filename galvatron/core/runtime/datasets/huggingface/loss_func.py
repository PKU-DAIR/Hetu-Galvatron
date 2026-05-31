"""Loss functions for HF data pipelines.

- ``hf_padding_loss_func``: per-microbatch loss for padding-mode batches.
- ``hf_packing_loss_func``: per-microbatch loss for packing / dyn_bsz batches.

Both functions expect ``micro_loss_masks`` to be pre-bound via
``functools.partial`` to a list whose head is popped once per micro-batch.
"""

from typing import List

import torch

import galvatron.core.runtime.parallel_state as parallel_state
from galvatron.core.runtime.utils.utils import (
    average_losses_across_context_parallel_group,
    average_losses_across_data_parallel_group,
)

def hf_loss_func(
    label: List,
    ce_loss: List[torch.Tensor],
    micro_loss_masks: List[torch.Tensor],
    effective_tokens: int,
    chunks: int,
):
    loss_mask = micro_loss_masks[0]
    micro_loss_masks.pop(0)

    # Model output is wrapped as [ce_loss]; unwrap.
    ce_loss = ce_loss[0]
    ce_loss = ce_loss.view(-1).float()

    loss_mask = loss_mask.view(-1).float()
    # local_loss = torch.sum(ce_loss * loss_mask) / loss_mask.sum()  # token-level loss
    # effective_tokens is per-micro-batch. Outside the training loop the
    # micro-batch loss is divided by ``chunks`` for gradient accumulation,
    # so we multiply by chunks here to recover the true per-token loss.
    local_loss = torch.sum(ce_loss * loss_mask) / effective_tokens * chunks

    ce_sum = torch.sum(ce_loss * loss_mask).detach()
    ce_count = loss_mask.sum().detach()
    stas = torch.stack([ce_sum, ce_count])
    torch.distributed.all_reduce(
        stas,
        op=torch.distributed.ReduceOp.SUM,
        group=parallel_state.get_vocab_dp_comm_group().group,
    )
    loss_reduced = stas[0] / stas[1]

    return local_loss, loss_reduced
