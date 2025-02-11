import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os
from galvatron.utils import set_seed, distributed_dataloader, print_loss
from galvatron.core import initialize_galvatron, GalvatronProfiler
from galvatron.models.bert_hf.BertModel_hybrid_parallel import bert_model_hp, get_bert_config
from galvatron.models.bert_hf.dataloader import DataLoaderForBert, get_batch, get_train_valid_test_data_iterators
from galvatron.models.bert_hf.meta_configs import model_name, model_layer_configs
from galvatron.models.bert_hf.arguments import model_args
from galvatron.core.initialize import init_empty_weights
from galvatron.core.utils import set_megatron_args_for_dataset
from megatron.training.arguments import _print_args

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_bert_config(args)
    model = bert_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")
    
    set_megatron_args_for_dataset(args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], model.dp_groups_whole[0])
    if local_rank == 0:
        _print_args("arguments", args)

    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = GalvatronProfiler(args)
    profiler.set_profiler_dist(path, model_layer_configs(config), model_name(config), start_iter=0)
    
    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")
    for iter in range(args.train_iters):
        input_ids, kwargs, loss_func = get_batch(train_data_iterator)
        
        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        batch = [input_ids]
        
        loss = model.forward_backward(batch, iter, profiler, 
                                    loss_func=loss_func,
                                    **kwargs)
        
        profiler.profile_memory(iter, "After Backward")
        
        optimizer.step()
        
        profiler.profile_memory(iter, "After optimizer_step")
        
        optimizer.zero_grad()
        
        profiler.post_profile_memory(iter)
        profiler.profile_time_end(iter, loss)

        torch.distributed.barrier()

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train_dist')
    set_seed()
    train(args)