import torch
from torch import nn
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
from tqdm import tqdm
import time
import os
import sys
from typing import Tuple, List
import argparse

import galvatron
from galvatron.core.pipeline import PipelineParallel, PipeSequential
from galvatron.core.comm_groups import gen_comm_groups
from galvatron.utils import read_json_config, write_json_config

def init_method_constant(val):
    def init_(tensor):
        return torch.nn.init.constant_(tensor, val)
    return init_

class pre_sync_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, hidden_states):
        return hidden_states

class pre_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        return hidden_states

def _reduce(input_, group):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group)==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_.contiguous(), group=group)

    return input_

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    
    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None

def reduce_from_tensor_model_parallel_region_group(input_, group):
    return _ReduceFromModelParallelRegion.apply(input_, group)

def copy_to_tensor_model_parallel_region_group(input_, group):
    return _CopyToModelParallelRegion.apply(input_, group)

class allreduce_block(nn.Module):
    def __init__(self, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(1024, 1024)

    def forward(self, hidden_states):
        hidden_states = copy_to_tensor_model_parallel_region_group(hidden_states, self.tp_group.group)
        hidden_states = reduce_from_tensor_model_parallel_region_group(hidden_states, self.tp_group.group)
        hidden_states = hidden_states.requires_grad_(True)
        return hidden_states

class DataLoaderRandom(Dataset):
    def __init__(self, local_bsz):
        # world_size = torch.distributed.get_world_size()
        self.dataset_size = local_bsz*11
        # self.input = np.random.randint(0, 100, size=(self.dataset_size, 512, 1024))
        self.input = np.random.rand(*(self.dataset_size, 512, 1024))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input = torch.FloatTensor(self.input[idx])
        return input

def fake_loss_func(labels, outputs):
    output = outputs[0]
    loss = output.sum()
    loss = loss.requires_grad_(True)
    return loss

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
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize torch.profiler before initializing CUDA context to avoid the bug
    # Refer to https://github.com/pytorch/pytorch/issues/60158
    with torch.profiler.profile() as p:
        pass

    pp_deg = args.pp_deg
    args.num_layers = 48
    args.local_batch_size = 32
    args.global_tp_deg = world_size // pp_deg
    train_batch_size_input = args.local_batch_size
    if rank == 0:
        print('local_bsz = %d'%train_batch_size_input)

    dataset = DataLoaderRandom(train_batch_size_input)
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input)

    all_tp_sizes = [args.global_tp_deg] * 48
    all_sp_sizes = [1] * 48
    tp_consecutive_flags = [args.global_tp_consec] * 48
    pp_group, tp_groups, _, _, _, _, _, _, _, _ = gen_comm_groups(all_tp_sizes, all_sp_sizes, pp_deg, tp_consecutive_flags)

    model = PipeSequential()
    model.add_module('pre_sync_module', pre_sync_module())
    model.add_module('pre_mlp', pre_mlp())
    for i in range(len(all_tp_sizes)):
        module = allreduce_block(tp_group=tp_groups[i])
        model.add_module('mlp_%d'%i, module)

    avg_num_layers = args.num_layers // args.pp_deg
    pp_ranks = [0, 0]
    for i in range(args.pp_deg):
        pp_ranks += [i] * avg_num_layers
    
    layer_output_tensor_shapes = [[[-1, 512, 1024]]] * len(pp_ranks)
    model = model.to(device)
    model = PipelineParallel(
                model = model, 
                model_ranks = pp_ranks, 
                layer_output_tensor_shapes = layer_output_tensor_shapes, 
                chunks=1, 
                process_group = pp_group.ranks, 
                nproc_per_node=8,
                info=False)

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    # Calculate theoretical communication message size
    pp_size = args.pp_deg
    # assert tp_size*pp_size*dp_size == 8
    p2p_message_size = train_batch_size_input*512*1024*4/1024/1024
    if local_rank == 0:
        print('Strategy: pp_deg = %d'%(pp_size))
        print('[p2p_message_size]: total %d MB'%(p2p_message_size))

    def trace_handler(prof):
        if rank == inter_node_send_rank:
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
            time.sleep(0.5*model.pp_global_ranks[0])
            print('Results of p2p from rank %d to rank %d:'%(inter_node_send_rank, inter_node_recv_rank))
            print(table)
            table = table.split('\n')
            def split_line(line):
                line = line.split('  ')
                ls = []
                for s in line:
                    if len(s):
                        ls.append(s.strip())
                return ls
            def str2time(s):
                if 'ms' in s:
                    return float(s[:-2])
                elif 'us' in s:
                    return float(s[:-2])*1e-3
                else:
                    return float(s[:-1])*1e3
            for line in table:
                if 'Name' in line:
                    title = split_line(line)
                if 'ncclKernel_SendRecv' in line:
                    result = split_line(line)
            for i in range(len(title)):
                # print('%s: %s'%(title[i],result[i]))
                if 'CUDA total' in title[i]:
                    cuda_total_idx = i
            p2p_time = str2time(result[cuda_total_idx])/10
            comm_coe = p2p_time / p2p_message_size
            print('**********')
            print('p2p_coe_pp_deg_%d (ms/MB):'%(pp_size), comm_coe)
            print('**********')

            comm_coe = torch.tensor([comm_coe]).to(device)
            torch.distributed.all_reduce(comm_coe, group=tp_groups[0].group, op=torch.distributed.ReduceOp.SUM)
            comm_coe = comm_coe.cpu().numpy()[0] / tp_groups[0].size
            if 0 in model.pp_global_ranks:
                time.sleep(1)
                print('********************')
                print('Final result:')
                print('p2p_coe_pp_deg_%d (ms/MB):'%(pp_size), comm_coe)
                print('********************')
                comm_type = 'p2p'
                gpu_num = world_size
                env_config_path = '../../env_configs/%s_bandwidth_dist_%d_gpus.json'%(comm_type, gpu_num)
                config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
                key = 'pp_deg_%d'%(pp_size)
                config[key] = comm_coe
                write_json_config(config, env_config_path)
                print('Already written comm_coe_%s into env config file %s!'%(key, env_config_path))

    inter_node_send_rank = model.pp_global_ranks[model.group_size // 2 - 1]
    inter_node_recv_rank = model.pp_global_ranks[model.group_size // 2]
    if rank == inter_node_send_rank:
        print('p2p comm from rank %d to rank %d.'%(inter_node_send_rank, inter_node_recv_rank))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                schedule=torch.profiler.schedule(wait=0,warmup=1,active=10),
                                on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_data_rank%d"%rank)) as p:
        for i, input in enumerate(tqdm(trainloader)):
            input = input.to(device)
            if rank == inter_node_send_rank:
                model.send_forward(input, tensor_shape=[train_batch_size_input, 512, 1024])
            if rank == inter_node_recv_rank:
                out = model.recv_forward(tensor_shape=[train_batch_size_input, 512, 1024])
            p.step()
    
    if rank == inter_node_send_rank:
        rawdir = "./profile_data_rank%d"%rank
        dir_list = os.listdir(rawdir)
        for dir in dir_list:
            if os.path.exists(rawdir+"/"+dir+"/ASCEND_PROFILER_OUTPUT"):
                rawdir = os.path.join(rawdir,dir)
                rawdir = os.path.join(rawdir,"ASCEND_PROFILER_OUTPUT")
        dir_list = os.listdir(rawdir)
        
        for file in dir_list:
            if file.endswith("details.csv"):
                file_dir = os.path.join(rawdir,file)
                break
        with open(file_dir) as f:
            comm_num = 0
            comm_time_list = []
            for line in f:
                line = line.split(',')
                if 'SendRecv' in line[0]:
                    comm_num += 1
                    comm_time_list.append(float(line[4]))
            comm_cnt = len(comm_time_list)
            comm_time = sum(sorted(comm_time_list)[comm_cnt//4:comm_cnt//4*3]) * 2
            comm_time *= 1e-3
            print(comm_time, comm_num)
        
        p2p_time = comm_time / 10
        comm_coe = p2p_message_size / p2p_time
        comm_coe = torch.tensor([comm_coe]).to(device)
        
        torch.distributed.all_reduce(comm_coe, group=tp_groups[0].group, op=torch.distributed.ReduceOp.SUM)
        comm_coe = comm_coe.cpu().numpy()[0] / tp_groups[0].size
        if 0 in model.pp_global_ranks:
            print('**********')
            print('p2p_coe_pp_deg_%d (ms/MB):'%(pp_size), comm_coe)
            print('**********')
            path = os.path.dirname(os.path.abspath(__file__))
            env_config_path = os.path.join(path, './hardware_configs/p2p_bandwidth_%dnodes_%dgpus_per_node.json'%(node_num,args.nproc_per_node))
            config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
            key = 'pp_size_%d'%(pp_size)
            config[key] = comm_coe
            write_json_config(config, env_config_path)
            print('Already written p2p bandwidth into env config file %s!'%(env_config_path))

    torch.distributed.barrier()
    os.system("rm -rf profile_data_rank%d"%rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree."
    )
    parser.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree."
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
    args = parser.parse_args()
    from megatron.training.global_vars import set_args
    args.sequence_parallel = False
    args.shape_order = "SBH"
    args.use_ulysses = False
    args.async_grad_reduce = True
    set_args(args)
    train(args)
