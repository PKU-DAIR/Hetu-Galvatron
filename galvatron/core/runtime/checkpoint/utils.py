import torch
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor


def copy_to_weight(submodule: nn.Module, tp_aware_weight: torch.Tensor):
    from galvatron.core.runtime.parallel_state import fsdp2_enabled

    if fsdp2_enabled():
        local_weight = distribute_tensor(
            tp_aware_weight,
            device_mesh=submodule.weight.device_mesh,
            placements=submodule.weight.placements,
        )
        submodule.weight.copy_(local_weight)
    else:
        submodule.weight.copy_(tp_aware_weight)


def copy_to_bias(submodule: nn.Module, tp_aware_bias: torch.Tensor):
    from galvatron.core.runtime.parallel_state import fsdp2_enabled
    
    if fsdp2_enabled():
        local_bias = distribute_tensor(
            tp_aware_bias,
            device_mesh=submodule.bias.device_mesh,
            placements=submodule.bias.placements,
        )
        submodule.bias.copy_(local_bias)
    else:
        submodule.bias.copy_(tp_aware_bias)