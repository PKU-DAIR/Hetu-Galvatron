"""Distributed training entry point for GPT.

Usage:
  torchrun ... train_dist.py <config.yaml> [hydra overrides...]
"""

import os
import sys

import torch

from galvatron.core.arguments import load_with_hydra
from galvatron.core.runtime.optimizer.utils import clip_grad_norm, get_optimizer_and_param_scheduler
from galvatron.core.runtime.models.builder import build_model, get_runtime_profiler
from galvatron.core.runtime.dataloader import get_batch, get_train_valid_test_data_iterators
from galvatron.core.runtime.utils.utils import set_megatron_args_for_dataset
from galvatron.core.runtime.checkpoint.llama_adapter import save_llama_module
from galvatron.core.runtime.initialize import initialize_galvatron, _print_args
from galvatron.core.runtime.datasets.huggingface.dataloader import (
    get_hf_train_valid_test_data_iterators,
    is_hf_dataset_built_on_rank,
)
from galvatron.utils import print_loss
from galvatron.utils.hf_config_adapter import resolve_model_config
from galvatron.core.runtime.checkpoint.llama_adapter import save_llama_module

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    resolve_model_config(args)
    model = build_model(args)

    data_source = getattr(args.data, "data_source", "megatron")

    if data_source == "hf" and getattr(args.train, "use_flash_attn", False):
        args.data.create_attention_mask_in_dataloader = False

    if local_rank == 0:
        print("Creating dataset...")
        _print_args(args)

    set_megatron_args_for_dataset(args)

    if data_source == "hf":
        if is_hf_dataset_built_on_rank():
            train_data_iterator, valid_data_iterator, test_data_iterator = get_hf_train_valid_test_data_iterators(args)
        else:
            train_data_iterator, valid_data_iterator, test_data_iterator = None, None, None
        torch.distributed.barrier()
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()

    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)

    path = os.path.dirname(os.path.abspath(__file__))
    start_iter = args.train.iteration
    end_iter = max(start_iter + 1, args.train.train_iters - 1)
    profiler = get_runtime_profiler(args, path, start_iter=start_iter, end_iter=end_iter)
    profiler.profile_memory(0, "After creating model")

    if local_rank == 0:
        print("Start training...")

    for iter_idx in range(getattr(args.train, "iteration", 0), args.train.train_iters):
        tokens, kwargs, loss_func = get_batch(train_data_iterator)

        profiler.profile_time_start(iter_idx)
        profiler.profile_memory(iter_idx, "Before Forward")

        loss = model.forward_backward([tokens], iter_idx, profiler, loss_func=loss_func, **kwargs)

        profiler.profile_memory(iter_idx, "After Backward")

        grad_norm = clip_grad_norm(model, args.train.clip_grad)
        optimizer.step()
        opt_param_scheduler.step(increment=args.train.global_batch_size)

        profiler.profile_memory(iter_idx, "After optimizer_step")
        optimizer.zero_grad()

        # print_loss(args, loss, -1, iter_idx)

        profiler.post_profile_memory(iter_idx)

        lr = optimizer.param_groups[0]["lr"]
        profiler.profile_time_end(iter_idx, loss, lr, grad_norm)

        if args.ckpt.save is not None and args.ckpt.save_interval is not None and (iter_idx + 1) % args.ckpt.save_interval == 0:
            save_llama_module(args.ckpt.save, model, optimizer, opt_param_scheduler, iter_idx + 1, args)

        torch.distributed.barrier()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path, overrides = sys.argv[1], sys.argv[2:]
        sys.argv = [sys.argv[0]]
        args = load_with_hydra(config_path, overrides=overrides, mode="train_dist")
    else:
        raise ValueError("Usage: python train_dist.py <config_path> [overrides...]")
    initialize_galvatron(args)
    train(args)
