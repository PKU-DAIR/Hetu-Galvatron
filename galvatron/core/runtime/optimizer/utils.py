import torch
import os
import json

from galvatron.core.runtime.optimizer.clip_grads import get_grad_norm_fp32, clip_grad_by_total_norm_fp32
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
from galvatron.core.runtime.parallel_state import fsdp2_enabled
from galvatron.core.runtime.optimizer.param_scheduler import get_optimizer_param_scheduler
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = ()
try:
    from apex.optimizers import FusedAdam as Adam
except ImportError:
    from torch.optim import AdamW as Adam


def _clip_grad_norm_fsdp2(model, max_norm, norm_type=2):
    parameters = []
    grads_for_norm = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, FSDPModule) and hasattr(module, "scaling_groups"):
                scale = 1 / (
                    torch.distributed.get_world_size(module.scaling_groups[0])
                    / torch.distributed.get_world_size(module.scaling_groups[1])
                )
                for param in module.parameters(recurse=False):
                    if param.grad is not None:
                        param.grad.mul_(scale)

    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad
        if DTensor != () and isinstance(grad, DTensor):
            grad = grad.to_local()
        parameters.append(param)
        grads_for_norm.append(grad.detach())

    if not grads_for_norm:
        return 0.0

    total_norm = get_grad_norm_fp32(grads_for_norm, norm_type)
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        with torch.no_grad():
            for param in parameters:
                if param.grad is not None:
                    param.grad.mul_(clip_coeff)
    return total_norm


def clip_grad_norm(model, max_norm, norm_type=2):
    if fsdp2_enabled():
        return _clip_grad_norm_fsdp2(model, max_norm, norm_type)

    parameters = []
    grads_for_norm = []
    with torch.no_grad():
        for name, module in model.named_modules():
            # TODO: find a better way to keep the correctness
            if isinstance(module, FSDP) and hasattr(module, "scaling_groups"):
                if module._handle.flat_param.grad is not None:
                    module._handle.flat_param.grad *= 1 / (
                        torch.distributed.get_world_size(module.scaling_groups[0])
                        / torch.distributed.get_world_size(module.scaling_groups[1])
                    )
    
    for name, params in model.named_parameters():
        if params.grad is None:
            continue
        parameters.append(params)
        grads_for_norm.append(params.grad)

    # Profiling / forward-only style runs may legitimately have no gradients.
    if not grads_for_norm:
        return 0.0

    total_norm = get_grad_norm_fp32(grads_for_norm, norm_type)
    if max_norm > 0:
        clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm)

    return total_norm


def get_optimizer_and_param_scheduler(model, args):

    train_args = args.train
    optimizer_cls = Adam
    if fsdp2_enabled() and getattr(Adam, "__module__", "").startswith("apex"):
        from torch.optim import AdamW
        optimizer_cls = AdamW
    optimizer = optimizer_cls(
        model.parameters(),
        lr=train_args.lr,
        weight_decay=train_args.weight_decay,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        eps=train_args.adam_eps,
    )

    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    ckpt_args = args.ckpt
    if ckpt_args.distributed_checkpoint:
        rank = torch.distributed.get_rank()
        if rank == 0:
            print("Begin to load optimizer and param scheduler")
        opt_path = os.path.join(ckpt_args.load, f"iter_{ckpt_args.load_iteration}", "optimizer", f"{rank}.pt")
        optimizer.load_state_dict(torch.load(opt_path, weights_only=False))
        opt_param_scheduler.load_state_dict(
            json.load(open(os.path.join(ckpt_args.load, f"iter_{ckpt_args.load_iteration}", "opt_param_scheduler.json")))
        )
        torch.distributed.barrier()
        if rank == 0:
            print("Finish loading optimizer and param scheduler")

    return optimizer, opt_param_scheduler