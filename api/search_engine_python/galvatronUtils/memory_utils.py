import torch

def print_peak_memory(prefix, device, type='allocated'):
    if type == 'allocated':
        print(prefix, '[Allocated]')
        max_mem = torch.cuda.max_memory_allocated(device)/2**20
        cur_mem = torch.cuda.memory_allocated(device)/2**20
        print("\tMax memory: %.2f MB\tCurrent memory : %.2f MB"%(max_mem, cur_mem))
    elif type == 'reserved':
        print(prefix, '[Reserved]')
        max_mem = torch.cuda.max_memory_reserved(device)/2**20
        cur_mem = torch.cuda.memory_reserved(device)/2**20
        print("\tMax memory: %.2f MB\tCurrent memory : %.2f MB"%(max_mem, cur_mem))
    return max_mem, cur_mem

def print_param_num(model):
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
