import os

import torch
from torch import nn
from tqdm import tqdm
from transformers import MixtralConfig, MixtralForCausalLM

from galvatron.core import (
    RuntimeProfiler,
    clip_grad_norm,
    get_optimizer_and_param_scheduler,
    initialize_galvatron,
    set_megatron_args_for_dataset,
)
from galvatron.models.moe.arguments import model_args
from galvatron.models.moe.dataloader import (
    DataLoaderForMoE,
    get_batch,
    get_train_valid_test_data_iterators,
    loss_func,
)
from galvatron.models.moe.MoEModel_checkpoint import save_moe_module
from galvatron.models.moe.MoEModel_hybrid_parallel import get_moe_config, get_runtime_profiler, moe_model_hp
from galvatron.models.moe.meta_configs import model_layer_configs, model_name
from galvatron.utils import distributed_dataloader, print_loss, set_seed
from megatron.training.arguments import _print_args


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_moe_config(args)
    model = moe_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")

    set_megatron_args_for_dataset(
        args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], model.dp_groups_whole[0]
    )
    if local_rank == 0:
        _print_args("arguments", args)

    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()

    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config, start_iter=0, end_iter=30)

    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    from torch.profiler import ProfilerActivity, record_function
    profile_iter = int(os.getenv("PROFILE_ITER", "14"))
    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=profile_iter - 4,
            warmup=4,
            active=1,
            repeat=1),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
        record_shapes=True,
        # with_stack=True,
        with_modules=True)
    prof.start()
    for iter in range(args.iteration, args.train_iters):
        tokens, kwargs, loss_func = get_batch(train_data_iterator)
        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        input_ids = tokens
        batch = [input_ids]

        with record_function("model_forward_backward"):
            loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

        profiler.profile_memory(iter, "After Backward")

        with record_function("gradient_clipping"):
            total_norm = clip_grad_norm(model, args.clip_grad)
        
        with record_function("optimizer_step"):
            optimizer.step()
            opt_param_scheduler.step(increment=args.global_batch_size)

        profiler.profile_memory(iter, "After optimizer_step")

        with record_function("zero_grad"):
            optimizer.zero_grad()

        profiler.post_profile_memory(iter)
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        profiler.profile_time_end(iter, loss, learning_rate, total_norm)

        torch.distributed.barrier()

        if args.save != None and (iter + 1) % args.save_interval == 0:
            save_moe_module(args.save, model, optimizer, opt_param_scheduler, iter + 1, args)
        
        prof.step()
        if iter == profile_iter:
            with open(f"training_log/breakdown/{os.getenv('METHOD')}_breakdown_{os.getenv('MODEL_SIZE')}_{os.getenv('DATA')}_batch{os.getenv('BATCH_SIZE')}_seq{os.getenv('SEQUENCE_LENGTH')}_aux{os.getenv('AUX')}_{torch.distributed.get_rank()}.txt", "w") as f:
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=200))
            prof.stop()
            exit(0)

if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
