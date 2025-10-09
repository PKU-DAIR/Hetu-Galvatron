import torch
from torch import nn
# from torch.optim import Adam

from transformers import LlamaConfig, LlamaForCausalLM
from tqdm import tqdm
import os
from galvatron.utils import set_seed, distributed_dataloader, print_loss
from galvatron.core import initialize_galvatron, GalvatronProfiler
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_hybrid_parallel_configs, construct_hybrid_parallel_model
from galvatron.models.llama_hf.dataloader import DataLoaderForLlama, get_batch, get_train_valid_test_data_iterators, loss_func
from galvatron.models.llama_hf.meta_configs import config_from_meta, set_model_config, model_name, model_layer_configs
from galvatron.models.llama_hf.arguments import model_args
from galvatron.core.initialize import init_empty_weights
from galvatron.core.utils import set_megatron_args_for_dataset, clip_grad_norm
from megatron.training.arguments import _print_args
from megatron.training.training import get_optimizer_param_scheduler
from apex.optimizers import FusedAdam as Adam

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, False)
    args.make_vocab_size_divisible_by = 1
    if local_rank == 0:
        print(config)
    
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if local_rank == 0:
        print("Creating Model...")
    if args.initialize_on_meta:
        with init_empty_weights():
            llama_model = LlamaForCausalLM(config)
    else:
        llama_model = LlamaForCausalLM(config)
    
    model = construct_hybrid_parallel_model(
        model=llama_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )

    if local_rank == 0:
        print("Creating Dataset...")
        
    set_megatron_args_for_dataset(args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], model.dp_groups_whole[0])
    if local_rank == 0:
        _print_args("arguments", args)

    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    
    path = os.path.dirname(os.path.abspath(__file__))
    profiler = GalvatronProfiler(args)
    profiler.set_profiler_dist(path, model_layer_configs(config), model_name(config),start_iter=0,end_iter=5000)
    
    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    for iter in range(args.train_iters):
        tokens, kwargs, loss_func = get_batch(train_data_iterator)
        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        input_ids = tokens
        batch = [input_ids]
        
        loss = model.forward_backward(batch, iter, profiler, 
                                      loss_func=loss_func,
                                      **kwargs)
        
        profiler.profile_memory(iter, "After Backward")
        
        # for name, weight in model.named_parameters():
        #     if torch.cuda.current_device() == 0:
        #         print(f"final grad {name},{weight.grad}")
        total_norm = clip_grad_norm(model, args.clip_grad)
        # total_norm = 0.0
        optimizer.step()
        opt_param_scheduler.step(increment=args.global_batch_size)
        
        profiler.profile_memory(iter, "After optimizer_step")
        
        optimizer.zero_grad()
        
        # print_loss(args, loss, ep, iter)

        profiler.post_profile_memory(iter)
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        profiler.profile_time_end(iter, loss, learning_rate, total_norm)
        
        torch.distributed.barrier()

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train_dist')
    set_seed()
    train(args)
