import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
from tqdm import tqdm
import time
import os
import sys
from typing import Tuple, List, Any
import argparse

import galvatron
from galvatron.core.pipeline import PipelineParallel, PipeSequential
from galvatron.core.comm_groups import gen_comm_groups
from galvatron.utils import read_json_config, write_json_config


def single_all_to_all(input, group):
    seq_world_size = dist.get_world_size(group)
    input_t = input.reshape(seq_world_size, -1)
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    return output

def set_seed(rank):
    seed = 123 + rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    set_seed(rank)
    world_size = torch.distributed.get_world_size()
    node_num = world_size / args.nproc_per_node
    # assert(args.nproc_per_node == 8)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    pp_deg = args.pp_deg
    args.num_layers = 24
    # args.local_batch_size = 32
    train_batch_size_input = args.local_batch_size
    if rank == 0:
        print('local_bsz = %d'%train_batch_size_input)

    tp_size = args.global_tp_deg

    all_tp_sizes = [args.global_tp_deg] * 24
    all_sp_sizes = [1] * 24
    tp_consecutive_flags = [args.global_tp_consec] * 24
    pp_group, tp_groups, _, _, _, _, _, _, _, _ = gen_comm_groups(all_tp_sizes, all_sp_sizes, pp_deg, tp_consecutive_flags)

    # Calculate theoretical communication message size
    tp_size = args.global_tp_deg
    pp_size = args.pp_deg
    dp_size = world_size // pp_deg // tp_size
    bs = args.local_batch_size
    # per size: 1MB * bs when profile time == 1
    all2all_message_size_per_layer = (bs*512*1024*2/1024/1024)
    all2allmessage_size_total = all2all_message_size_per_layer * 24 // pp_deg
    if local_rank == 0:
        print('Strategy: %d_%d_%d'%(pp_size,tp_size,args.global_tp_consec))
        print('[all2all_message_size]: per_layer %d MB, total %d MB'%(all2all_message_size_per_layer,all2allmessage_size_total))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_list = []
    for _ in range(10):
        input = np.random.rand(*(bs, 512 // tp_size, 1024))
        input = torch.tensor(input, dtype=torch.bfloat16, device=device)
        output = single_all_to_all(input, tp_groups[0].group)
        
    for _ in range(100):
        input = np.random.rand(*(bs, 512 // tp_size, 1024))
        input = torch.tensor(input, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()
        start.record()
        output = single_all_to_all(input, tp_groups[0].group)
        end.record()
        torch.cuda.synchronize()
        time_list.append(start.elapsed_time(end))

    
    if args.profile_time == 0:
        assert False
    else:
        per_comm_time = sum(time_list) / len(time_list)
        per_comm_time = torch.tensor([per_comm_time]).to(device)
        torch.distributed.all_reduce(per_comm_time, group=tp_groups[0].group, op=torch.distributed.ReduceOp.SUM)
        per_comm_time = per_comm_time.cpu().numpy()[0] / tp_groups[0].size
        if rank == 0:
            print(sum(time_list), len(time_list))
            print('**********')
            print('comm_time_%dMB_%d_%d:'%(args.local_batch_size, pp_size, tp_size), per_comm_time)
            print('**********')
            path = os.path.dirname(os.path.abspath(__file__))
            env_config_path = os.path.join(path, './hardware_configs/sp_time_%dnodes_%dgpus_per_node.json'%(node_num,args.nproc_per_node))
            config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
            key = 'all2all_size_%d_%dMB_time'%(tp_size,args.local_batch_size)
            config[key] = per_comm_time
            write_json_config(config, env_config_path)
            print('Already written all2all bandwidth into env config file %s!'%(env_config_path))


    os.system("rm -rf profile_data_rank%d"%rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16,32,64,128,256],
    )
    parser.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8,16,32,64,128,256],
    )
    parser.add_argument(
        "--local_batch_size", type=int, default=32, help="local training batch size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=24, help="Number of layers"
    )
    parser.add_argument("--local-rank" ,type=int,default=-1)
    parser.add_argument(
        "--nproc_per_node", type=int, default=-1, help="Nproc per node",
    )
    parser.add_argument(
        "--profile_time", type=int, default=0, help="Profile time",
    )
    args = parser.parse_args()
    from megatron.training.global_vars import set_args
    args.sequence_parallel = False
    args.shape_order = "SBH"
    args.use_ulysses = False
    args.async_grad_reduce = True
    set_args(args)
    train(args)
