import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def set_seed(seed = 1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def distributed_dataloader(dataset, global_bsz, shuffle = True, args = None, group = None, collate_fn=None):
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    # pp_deg = args.pp_deg if args is not None and 'pp_deg' in args else 1
    # data_num_replicas = world_size // pp_deg
    train_batch_size_input = global_bsz // world_size
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=shuffle,num_replicas=world_size,rank=rank),
                            collate_fn=collate_fn)
    return trainloader

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
        if ep == -1:
            print('(Iteration %d): Loss = %.3f'% (iter,loss))
        else:
            print('[Epoch %d] (Iteration %d): Loss = %.3f'% (ep,iter,loss))