import torch
import numpy as np
import random
import math
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def distributed_dataloader(dataset, global_bsz, shuffle = True, args = None):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    pp_deg = args.pp_deg if args is not None and 'pp_deg' in args else 1
    data_num_replicas = world_size // pp_deg
    train_batch_size_input = global_bsz // data_num_replicas
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=shuffle,num_replicas=data_num_replicas,rank=rank%data_num_replicas))
    return trainloader

def distributed_dataloader_group(dataset, global_bsz, shuffle = True, devices = None):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    devices = sorted(list(set(devices))) if devices is not None else list(range(world_size))
    data_num_replicas, group_rank = len(devices), devices.index(rank) if rank in devices else 0
    if global_bsz % data_num_replicas == 0:
        train_batch_size_input = global_bsz // data_num_replicas
        trainloader = DataLoader(dataset=dataset,
                                batch_size=train_batch_size_input,
                                sampler=DistributedSampler(dataset,shuffle=shuffle,num_replicas=data_num_replicas,rank=group_rank))
    else:
        # sampler = CustomDistributedSampler(dataset, shuffle=shuffle, num_replicas=data_num_replicas, rank=group_rank, global_batch_size=global_bsz)
        # trainloader = DataLoader(dataset, sampler=sampler, batch_size=sampler.max_samples)
        train_batch_size_input = (global_bsz // data_num_replicas) + 1
        trainloader = DataLoader(dataset=dataset,
                                batch_size=train_batch_size_input,
                                sampler=DistributedSampler(dataset,shuffle=shuffle,num_replicas=data_num_replicas,rank=group_rank))
    return trainloader

class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, global_batch_size=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.global_batch_size = global_batch_size
        self.total_size = math.ceil(len(dataset) / num_replicas) * num_replicas
        self.seed = seed
        self.generator = torch.Generator()
        
        # Calculate the number of samples for each replica
        base_samples_per_replica = self.global_batch_size // self.num_replicas
        rest = self.global_batch_size % self.num_replicas

        # Determine the start and end indices for each replica
        self.samples_per_replica = base_samples_per_replica + (1 if self.rank < rest else 0)
        self.max_samples = base_samples_per_replica + (1 if rest > 0 else 0)
        
    def __iter__(self):
        # Set the generator for this epoch
        if self.shuffle:
            self.generator.manual_seed(self.seed + self.epoch)

        # Generate indices
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        
        # Adjust indices for the total size
        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank * self.total_size // self.num_replicas:(self.rank + 1) * self.total_size // self.num_replicas]

        # Return the indices for this specific replica
        indices = indices[:self.samples_per_replica]
        return iter(indices)

def print_loss(args, loss, ep, iter):
    if args.check_loss or args.profile:
        if loss is None:
            return
        if isinstance(loss, (list, tuple)): # Average loss of each microbatch
            if len(loss) == 0:
                return
            if isinstance(loss[0], torch.Tensor):
                loss = np.mean([l.item() for l in loss])
            else:
                loss = np.mean(loss)
        else:
            loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        print('[Epoch %d] (Iteration %d): Loss = %.3f'% (ep,iter,loss))