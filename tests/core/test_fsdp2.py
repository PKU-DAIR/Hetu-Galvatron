"""FSDP2 (fully_shard) correctness test — Galvatron distributed vs HF baseline.

Verifies that FSDP2 training with zero2 / zero3 produces identical loss to a
single-GPU HuggingFace GPT-2 baseline over 3 steps, for both synchronous and
asynchronous gradient reduction.

Usage (8 GPUs):
    pytest tests/core/test_fsdp2.py -v -s
"""

import os
import sys

# Ensure the repo root is on sys.path for subprocess-based distributed runs.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytest
import torch
import json
from typing import Dict, Any
from torch.optim import Adam
from torch.amp import autocast
from torch.nn import CrossEntropyLoss

from tests.utils.init_dist import init_dist_env
from tests.utils.runtime_args import make_test_args
from galvatron.core.runtime.parallel_state import set_args, set_global_memory_buffer
from galvatron.core.runtime.models.builder import build_model
from galvatron.core.runtime.datasets import RandomTokenDataset, random_collate_fn
from galvatron.utils.training_utils import set_seed, distributed_dataloader
from galvatron.tools.checkpoint_convert_h2g import convert_checkpoints_gpt
from transformers import GPT2Config, GPT2LMHeadModel


# ---------------------------------------------------------------------------
# Distributed test body
# ---------------------------------------------------------------------------

def _run_test(test_args: Dict[str, Any]):
    rank, world_size = init_dist_env()
    parallel_config = test_args["parallel_config"]
    mixed_precision = test_args["mixed_precision"]
    async_grad_reduce = test_args["async_grad_reduce"]
    checkpoint_dir = test_args["checkpoint_dir"]
    num_steps = test_args["num_steps"]
    seed = test_args["seed"]
    global_bsz = parallel_config["global_bsz"]

    device = torch.device("cuda", rank)
    set_seed(seed)

    args = make_test_args(
        rank=rank,
        world_size=world_size,
        checkpoint_load=checkpoint_dir["converted"],
        mixed_precision=mixed_precision,
        async_grad_reduce=async_grad_reduce,
        galvatron_config_path=parallel_config,
        global_batch_size=global_bsz,
        chunks=parallel_config["chunks"],
        seed=seed,
        fsdp_version="fsdp2",
    )
    set_args(args)
    set_global_memory_buffer()

    hf_config = GPT2Config(
        n_embd=args.model.hidden_size,
        n_layer=args.model.num_layers,
        n_head=args.model.num_attention_heads,
        n_positions=args.train.seq_length,
        n_inner=args.model.ffn_hidden_size,
        vocab_size=args.model.vocab_size,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )

    # Rank world_size-1 builds the HF baseline, saves and converts it so all
    # ranks load from the same initial weights via the checkpoint loading path.
    if rank == world_size - 1:
        baseline_model = GPT2LMHeadModel(hf_config)
        baseline_optimizer = Adam(
            baseline_model.parameters(),
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
        )
        baseline_model.save_pretrained(checkpoint_dir["baseline"])
        convert_checkpoints_gpt(checkpoint_dir["baseline"], checkpoint_dir["converted"])
        baseline_model = baseline_model.to(device)

    torch.distributed.barrier()

    model = build_model(args)
    optimizer = Adam(
        model.parameters(),
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
    )

    trainloader = distributed_dataloader(
        dataset=RandomTokenDataset(args.model.vocab_size, args.train.seq_length, size=256),
        global_bsz=global_bsz,
        shuffle=True,
        group=model.dp_groups_whole[0].group,
        collate_fn=random_collate_fn,
    )

    for i, batch in enumerate(trainloader):
        tokens, kwargs, loss_func = batch
        input_ids = tokens
        batch = [input_ids]

        dp_group = model.dp_groups_whole[0].group
        dp_world_size = torch.distributed.get_world_size(dp_group)

        # all_gather is a collective – all ranks must participate
        gathered_input_ids = [torch.zeros_like(input_ids) for _ in range(dp_world_size)]
        gathered_labels = [torch.zeros_like(kwargs["labels"]) for _ in range(dp_world_size)]
        torch.distributed.all_gather(gathered_input_ids, input_ids, group=dp_group)
        torch.distributed.all_gather(gathered_labels, kwargs["labels"], group=dp_group)

        loss = model.forward_backward(batch, i, None, loss_func=loss_func, **kwargs)
        optimizer.step()
        optimizer.zero_grad()

        if loss is not None:
            loss = torch.tensor(loss, device=device, dtype=torch.float)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)

        if rank == world_size - 1:
            full_batch = torch.cat(gathered_input_ids, dim=0)
            full_labels = torch.cat(gathered_labels, dim=0)
            cast_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float
            with autocast(device_type="cuda", dtype=cast_dtype):
                logits = baseline_model(input_ids=full_batch).logits
                baseline_loss = CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    full_labels.view(-1).to(logits.device),
                )

            baseline_loss.backward()
            baseline_optimizer.step()
            baseline_optimizer.zero_grad()
        else:
            baseline_loss = torch.tensor(0.0, device=device, dtype=torch.float)
            loss = torch.tensor(0.0, device=device, dtype=torch.float)

        torch.distributed.broadcast(baseline_loss, src=world_size - 1)
        torch.distributed.broadcast(loss, src=world_size - 1)

        assert torch.allclose(loss, baseline_loss, rtol=5e-3), (
            f"Loss mismatch at iteration {i}: {loss} vs {baseline_loss}"
        )

        torch.distributed.barrier()
        if i == num_steps - 1:
            break


# ---------------------------------------------------------------------------
# Pytest parametrize
# ---------------------------------------------------------------------------

@pytest.mark.distributed
@pytest.mark.parallel
@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("mixed_precision", ["bf16"])
@pytest.mark.parametrize("parallel_config", (
    {
        "pp_deg": 1,
        "tp_sizes_enc": "1,1,1,1",
        "tp_consecutive_flags": "1,1,1,1",
        "cp_sizes_enc": "1,1,1,1",
        "dp_types_enc": "0,0,0,0",
        "use_sp": "0,0,0,0",
        "checkpoint": "0,0,0,0",
        "global_bsz": 16,
        "chunks": 2,
        "pp_division": "4",
        "pipeline_type": "pipedream_flush",
        "default_dp_type": "zero2",
        "vtp": 1,
        "vsp": 0,
    },
    {
        "pp_deg": 1,
        "tp_sizes_enc": "1,1,1,1",
        "tp_consecutive_flags": "1,1,1,1",
        "cp_sizes_enc": "1,1,1,1",
        "dp_types_enc": "0,0,0,0",
        "use_sp": "0,0,0,0",
        "checkpoint": "0,0,0,0",
        "global_bsz": 16,
        "chunks": 2,
        "pp_division": "4",
        "pipeline_type": "pipedream_flush",
        "default_dp_type": "zero3",
        "vtp": 1,
        "vsp": 0,
    },
))
@pytest.mark.parametrize("async_grad_reduce", [False, True])
def test_fsdp2_correctness(
    run_distributed, world_size, parallel_config,
    mixed_precision, async_grad_reduce, checkpoint_dir,
):
    """Test FSDP2 (zero2 / zero3) training correctness against a baseline HF model."""
    config = {
        "parallel_config": parallel_config,
        "num_steps": 3,
        "seed": 42,
        "checkpoint_dir": checkpoint_dir,
        "mixed_precision": mixed_precision,
        "async_grad_reduce": async_grad_reduce,
    }

    run_distributed(
        func_name="_run_test",
        world_size=world_size,
        args=config,
        script=__file__,
    )


# ---------------------------------------------------------------------------
# torchrun / subprocess entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_file.py <function_name> <json_args>")
        sys.exit(1)

    func_name = sys.argv[1]
    test_args = json.loads(sys.argv[2])

    if func_name == "_run_test":
        _run_test(test_args)
    else:
        print(f"Unknown function: {func_name}")
        sys.exit(1)
