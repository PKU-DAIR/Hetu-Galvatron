import json
import os
import operator
import torch
from functools import partial, reduce
from packaging.version import Version as PkgVersion
from importlib.metadata import version
import logging
from typing import Any, Dict

import torch.distributed
from galvatron.core.runtime import parallel_state
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs

try:
    _torch_version = PkgVersion(torch.__version__)
except Exception:
    # This is a WAR for building docs, where torch is not actually imported
    _torch_version = PkgVersion("0.0.0")

_te_version = None

# utility functions, support on nested attributes for getattr, setattr, and setattr
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
# https://stackoverflow.com/questions/24779483/hasattr-for-nested-attributes
def rgetattr(obj, attr):
    if attr == "":
        return obj

    def _getattr_fsdp(obj, attr):
        if isinstance(obj, FSDP):
            return getattr(obj._fsdp_wrapped_module, attr)
        else:
            return getattr(obj, attr)

    return reduce(_getattr_fsdp, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rhasattr(obj, attr):
    try:
        rgetattr(obj, attr)
        return True
    except AttributeError:
        return False


def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
    """If torch distributed is initialized, log only on rank

    Args:
        logger (logging.Logger): The logger to write the logs

        args (Tuple[Any]): All logging.Logger.log positional arguments

        rank (int, optional): The rank to write on. Defaults to 0.

        kwargs (Dict[str, Any]): All logging.Logger.log keyword arguments
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            logger.log(*args, **kwargs)
    else:
        logger.log(*args, **kwargs)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        """
        Returns (potentially) a sub-tensor from the self.buffer for the given shape.
        """
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def get_torch_version():
    """Get pytorch version from __version__; if not available use pip's. Use caching."""

    def get_torch_version_str():
        import torch

        if hasattr(torch, '__version__'):
            return str(torch.__version__)
        else:
            return version("torch")

    global _torch_version
    if _torch_version is None:
        _torch_version = PkgVersion(get_torch_version_str())
    return _torch_version


def is_torch_min_version(version, check_equality=True):
    """Check if minimum version of `torch` is installed."""
    if check_equality:
        return get_torch_version() >= PkgVersion(version)
    return get_torch_version() > PkgVersion(version)


def get_te_version():
    """Get TE version from __version__; if not available use pip's. Use caching."""

    def get_te_version_str():
        import transformer_engine as te

        if hasattr(te, '__version__'):
            return str(te.__version__)
        else:
            return version("transformer-engine")

    global _te_version
    if _te_version is None:
        _te_version = PkgVersion(get_te_version_str())
    return _te_version


def is_te_min_version(version, check_equality=True):
    """Check if minimum version of `transformer-engine` is installed."""
    if check_equality:
        return get_te_version() >= PkgVersion(version)
    return get_te_version() > PkgVersion(version)


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def set_megatron_args_for_dataset(args:GalvatronRuntimeArgs):
    torch.distributed.barrier()

    vocab_dp_comm_group = parallel_state.get_vocab_dp_comm_group()
    world_size = args.world_size
    assert world_size // args.parallel.pp_deg // args.parallel.vocab_tp // args.parallel.vocab_cp == len(vocab_dp_comm_group.ranks)
    
    if args.ckpt.load_iteration != 0:
        assert args.ckpt.distributed_checkpoint == True, "Checkpoint iteration > 0 requires distributed checkpoint"
        args.train.iteration = args.ckpt.load_iteration
    else:
        args.train.iteration = 0

    args.train.micro_batch_size = args.train.global_batch_size // len(vocab_dp_comm_group.ranks)


def get_layernorm_offset(model, layernorm_name=[]):
    total_ln_offset = []
    total_ln_size = []
    for module in model:
        ln_offset = []
        ln_size = []
        offset = 0
        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            is_ln = False
            for ln_name in layernorm_name:
                if ln_name in submodule_name:
                    is_ln = True
                    break
            for param_name, param in _named_parameters_with_duplicates(submodule, recurse=False):
                if is_ln: #  or getattr(param, "sequence_parallel", False):
                    ln_offset.append(offset)
                    ln_size.append(param.numel())
                offset += param.numel()
        total_ln_offset.append(ln_offset)
        total_ln_size.append(ln_size)

    return total_ln_offset, total_ln_size


def get_batch_on_this_tp_rank(data_iterator):
    # Import here to avoid circular import at module load time.
    from galvatron.core.runtime.parallel_state import get_args
    args = get_args()

    def _broadcast(item):
       if item is not None:
           torch.distributed.broadcast(item, parallel_state.get_vocab_tp_sp_src_rank(), group=parallel_state.get_vocab_tp_sp_comm_group().group)

    if parallel_state.get_vocab_tp_sp_rank() == 0:

       if data_iterator is not None:
           data = next(data_iterator)
       else:
           data = None

       batch = {
           'tokens': data["tokens"].cuda(non_blocking = True),
           'labels': data["labels"].cuda(non_blocking = True),
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
           'position_ids': data["position_ids"].cuda(non_blocking = True)
       }

       if args.parallel.pp_deg == 1:
           _broadcast(batch['tokens'])
           _broadcast(batch['labels'])
           _broadcast(batch['loss_mask'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

       elif parallel_state.is_pipeline_first_stage():
           _broadcast(batch['tokens'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

       elif parallel_state.is_pipeline_last_stage():
           # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
           # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
           # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
        #    if args.mtp_num_layers is not None:
        #         _broadcast(batch['tokens'])
        #         _broadcast(batch['position_ids'])
           _broadcast(batch['labels'])
           _broadcast(batch['loss_mask'])
           _broadcast(batch['attention_mask'])

    else:

       tokens=torch.empty((args.train.micro_batch_size,args.train.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       labels=torch.empty((args.train.micro_batch_size,args.train.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       loss_mask=torch.empty((args.train.micro_batch_size,args.train.seq_length), dtype = torch.float32 , device = torch.cuda.current_device())
       if args.data.create_attention_mask_in_dataloader:
           attention_mask=torch.empty(
                (args.train.micro_batch_size,1,args.train.seq_length,args.train.seq_length), dtype = torch.bool , device = torch.cuda.current_device()
            )
       else:
           attention_mask=None
       position_ids=torch.empty((args.train.micro_batch_size, args.train.seq_length), dtype=torch.int64, device=torch.cuda.current_device())

       if args.parallel.pp_deg == 1:
           _broadcast(tokens)
           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)
           _broadcast(position_ids)

       elif parallel_state.is_pipeline_first_stage():
           labels=None
           loss_mask=None

           _broadcast(tokens)
           _broadcast(attention_mask)
           _broadcast(position_ids)

       elif parallel_state.is_pipeline_last_stage():
           # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
           # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
           # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
        #    if args.mtp_num_layers is not None:
        #         _broadcast(tokens)
        #         _broadcast(position_ids)
        #    else:
           tokens=None
           position_ids=None

           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)

       batch = {
           'tokens': tokens,
           'labels': labels,
           'loss_mask': loss_mask,
           'attention_mask': attention_mask,
           'position_ids': position_ids
       }

    return batch


def get_batch_on_this_cp_rank(batch: Dict[str, Any]):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    cp_size = parallel_state.get_vocab_cp_world_size()
    if cp_size > 1:
        cp_rank = parallel_state.get_vocab_cp_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 2
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor(
                    [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
                ).cuda(non_blocking=True)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                batch[key] = val

    return batch


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    vocab_dp_comm_group = parallel_state.get_vocab_dp_comm_group()

    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=vocab_dp_comm_group.group)
    averaged_losses = averaged_losses / parallel_state.get_parallel_world_size(vocab_dp_comm_group.group)

    return averaged_losses

def average_losses_across_context_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_context_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_context_parallel_group())

    return averaged_losses