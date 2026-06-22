import os
from typing import List

from galvatron.core.runtime.utils.utils import GlobalMemoryBuffer
from galvatron.core.runtime.datasets.megatron.tokenizer import build_tokenizer
import torch
import torch.distributed
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
from galvatron.core.runtime.comm_groups import CommGroup

# --- Helper Functions ---
def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


# --- Parallel World Size and Rank ---
def get_parallel_world_size(group:torch.distributed.ProcessGroup):
    return torch.distributed.get_world_size(group=group)


def get_parallel_rank(group:torch.distributed.ProcessGroup):
    return torch.distributed.get_rank(group=group)


# --- Global Memory Buffer ---
_GLOBAL_MEMORY_BUFFER:GlobalMemoryBuffer = None

def set_global_memory_buffer():
    """Initialize global buffer."""
    global _GLOBAL_MEMORY_BUFFER
    _ensure_var_is_not_initialized(_GLOBAL_MEMORY_BUFFER, 'global memory buffer')
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


# --- Global Args ---
_GLOBAL_ARGS:GalvatronRuntimeArgs = None

def set_args(args:GalvatronRuntimeArgs):
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _GLOBAL_ARGS = args


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


# --- Global Tokenizer ---
_GLOBAL_TOKENIZER = None

def _build_tokenizer(args:GalvatronRuntimeArgs):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


# --- Global Tensorboard Writer ---
_GLOBAL_TENSORBOARD_WRITER = None

def _set_tensorboard_writer(args:GalvatronRuntimeArgs):
    """Set tensorboard writer. *args* is the full GalvatronRuntimeArgs."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, 'tensorboard writer')
    log_cfg = args.logging
    if getattr(log_cfg, 'tensorboard_dir', None) and \
       args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=log_cfg.tensorboard_dir,
                max_queue=log_cfg.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


# --- Global Wandb Writer ---
_GLOBAL_WANDB_WRITER = None

def _set_wandb_writer(args:GalvatronRuntimeArgs):
    """Set wandb writer. *args* is the full GalvatronRuntimeArgs."""
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER, 'wandb writer')
    log_cfg = args.logging
    if getattr(log_cfg, 'wandb_project', '') and args.rank == (args.world_size - 1):
        if log_cfg.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb
        if log_cfg.wandb_save_dir:
            save_dir = log_cfg.wandb_save_dir
        else:
            save_dir = os.path.join(args.ckpt.save, 'wandb')
        wandb_kwargs = {
            'dir': save_dir,
            'name': log_cfg.wandb_exp_name,
            'project': log_cfg.wandb_project,
            'config': args.model_dump()}
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb


# --- Total Global Variables ---
def set_global_variables(args:GalvatronRuntimeArgs):
    """Set global variables."""
    set_args(args)
    _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_wandb_writer(args)


# --- pipeline related variables ---
_GLOBAL_PP_COMM_GROUP:CommGroup = None

def set_pp_comm_group(comm_group:CommGroup):
    global _GLOBAL_PP_COMM_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_PP_COMM_GROUP, 'pipeline parallel comm group')
    _GLOBAL_PP_COMM_GROUP = comm_group


def get_pp_comm_group():
    global _GLOBAL_PP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_PP_COMM_GROUP, 'pipeline parallel comm group')
    return _GLOBAL_PP_COMM_GROUP


def get_pp_world_size():
    global _GLOBAL_PP_COMM_GROUP
    assert _GLOBAL_PP_COMM_GROUP is not None, 'pipeline parallel group is not initialized'
    return get_parallel_world_size(_GLOBAL_PP_COMM_GROUP.group)


def get_pp_rank():
    global _GLOBAL_PP_COMM_GROUP
    assert _GLOBAL_PP_COMM_GROUP is not None, 'pipeline parallel group is not initialized'
    return get_parallel_rank(_GLOBAL_PP_COMM_GROUP.group)


def is_pipeline_first_stage():
    return get_pp_rank() == 0


def is_pipeline_last_stage():
    return get_pp_rank() == get_pp_world_size() - 1


# TODO: Add vpp support
def get_virtual_pipeline_model_parallel_rank():
    return None


# --- vocab related variables ---
_GLOBAL_VOCAB_TP_SP_COMM_GROUP:CommGroup = None
_GLOBAL_VOCAB_CP_COMM_GROUP:CommGroup = None
_GLOBAL_VOCAB_DP_COMM_GROUP:CommGroup = None
_GLOBAL_VOCAB_TP_SP_SRC_RANK:int = None # TODO: Further verify the role and correctness
_GLOBAL_VOCAB_TP_SP_CP_GROUP:torch.distributed.ProcessGroup = None

def set_vocab_tp_sp_comm_group(comm_group:CommGroup):
    global _GLOBAL_VOCAB_TP_SP_COMM_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_VOCAB_TP_SP_COMM_GROUP, 'vocab tp sp comm group')
    _GLOBAL_VOCAB_TP_SP_COMM_GROUP = comm_group


def set_vocab_cp_comm_group(comm_group:CommGroup):
    global _GLOBAL_VOCAB_CP_COMM_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_VOCAB_CP_COMM_GROUP, 'vocab cp comm group')
    _GLOBAL_VOCAB_CP_COMM_GROUP = comm_group


def set_vocab_dp_comm_group(comm_group:CommGroup):
    global _GLOBAL_VOCAB_DP_COMM_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_VOCAB_DP_COMM_GROUP, 'vocab dp comm group')
    _GLOBAL_VOCAB_DP_COMM_GROUP = comm_group


def set_vocab_tp_sp_src_rank(rank:int):
    global _GLOBAL_VOCAB_TP_SP_SRC_RANK
    _ensure_var_is_not_initialized(_GLOBAL_VOCAB_TP_SP_SRC_RANK, 'vocab tp sp src rank')
    _GLOBAL_VOCAB_TP_SP_SRC_RANK = rank


def get_vocab_tp_sp_comm_group():
    global _GLOBAL_VOCAB_TP_SP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_VOCAB_TP_SP_COMM_GROUP, 'vocab tp sp comm group')
    return _GLOBAL_VOCAB_TP_SP_COMM_GROUP


def get_vocab_cp_comm_group():
    global _GLOBAL_VOCAB_CP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_VOCAB_CP_COMM_GROUP, 'vocab cp comm group')
    return _GLOBAL_VOCAB_CP_COMM_GROUP


def get_vocab_dp_comm_group():
    global _GLOBAL_VOCAB_DP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_VOCAB_DP_COMM_GROUP, 'vocab dp comm group')
    return _GLOBAL_VOCAB_DP_COMM_GROUP


def get_vocab_tp_sp_src_rank():
    global _GLOBAL_VOCAB_TP_SP_SRC_RANK
    _ensure_var_is_initialized(_GLOBAL_VOCAB_TP_SP_SRC_RANK, 'vocab tp sp src rank')
    return _GLOBAL_VOCAB_TP_SP_SRC_RANK


def get_vocab_tp_sp_world_size():
    global _GLOBAL_VOCAB_TP_SP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_VOCAB_TP_SP_COMM_GROUP, 'vocab tp sp comm group')
    return get_parallel_world_size(_GLOBAL_VOCAB_TP_SP_COMM_GROUP.group)


def get_vocab_tp_sp_rank():
    global _GLOBAL_VOCAB_TP_SP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_VOCAB_TP_SP_COMM_GROUP, 'vocab tp sp comm group')
    return get_parallel_rank(_GLOBAL_VOCAB_TP_SP_COMM_GROUP.group)


def get_vocab_dp_world_size():
    global _GLOBAL_VOCAB_DP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_VOCAB_DP_COMM_GROUP, 'vocab dp comm group')
    return get_parallel_world_size(_GLOBAL_VOCAB_DP_COMM_GROUP.group)


def get_vocab_dp_rank():
    global _GLOBAL_VOCAB_DP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_VOCAB_DP_COMM_GROUP, 'vocab dp comm group')
    return get_parallel_rank(_GLOBAL_VOCAB_DP_COMM_GROUP.group)


def get_vocab_cp_world_size():
    global _GLOBAL_VOCAB_CP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_VOCAB_CP_COMM_GROUP, 'vocab cp comm group')
    return get_parallel_world_size(_GLOBAL_VOCAB_CP_COMM_GROUP.group)


def get_vocab_cp_rank():
    global _GLOBAL_VOCAB_CP_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_VOCAB_CP_COMM_GROUP, 'vocab cp comm group')
    return get_parallel_rank(_GLOBAL_VOCAB_CP_COMM_GROUP.group)


def _set_vocab_tp_sp_cp_group():
    global _GLOBAL_VOCAB_TP_SP_COMM_GROUP
    global _GLOBAL_VOCAB_CP_COMM_GROUP
    global _GLOBAL_VOCAB_TP_SP_CP_GROUP

    _ensure_var_is_initialized(_GLOBAL_VOCAB_TP_SP_COMM_GROUP, 'vocab tp sp comm group')
    _ensure_var_is_initialized(_GLOBAL_VOCAB_CP_COMM_GROUP, 'vocab cp comm group')
    _ensure_var_is_not_initialized(_GLOBAL_VOCAB_TP_SP_CP_GROUP, 'vocab tp sp cp comm group')
    
    tp_sp_ranks = _GLOBAL_VOCAB_TP_SP_COMM_GROUP.ranks
    cp_ranks = _GLOBAL_VOCAB_CP_COMM_GROUP.ranks
    ranks = sorted(list(set(tp_sp_ranks + cp_ranks)))
    _GLOBAL_VOCAB_TP_SP_CP_GROUP = torch.distributed.new_group(ranks=ranks, backend='nccl')

def get_vocab_tp_sp_cp_group():
    global _GLOBAL_VOCAB_TP_SP_CP_GROUP
    if _GLOBAL_VOCAB_TP_SP_CP_GROUP is None:
        _set_vocab_tp_sp_cp_group()
    return _GLOBAL_VOCAB_TP_SP_CP_GROUP

def get_vocab_tp_sp_cp_world_size():
    global _GLOBAL_VOCAB_TP_SP_CP_GROUP
    if _GLOBAL_VOCAB_TP_SP_CP_GROUP is None:
        _set_vocab_tp_sp_cp_group()
    return get_parallel_world_size(_GLOBAL_VOCAB_TP_SP_CP_GROUP)


def get_vocab_tp_sp_cp_rank():
    global _GLOBAL_VOCAB_TP_SP_CP_GROUP
    if _GLOBAL_VOCAB_TP_SP_CP_GROUP is None:
        _set_vocab_tp_sp_cp_group()
    return get_parallel_rank(_GLOBAL_VOCAB_TP_SP_CP_GROUP)


# --- transformer layer related variables ---
_GLOBAL_TP_WHOLE_COMM_GROUP:List[CommGroup] = None
_GLOBAL_SP_WHOLE_COMM_GROUP:List[CommGroup] = None
_GLOBAL_DP_WHOLE_COMM_GROUP:List[CommGroup] = None
_GLOBAL_CP_WHOLE_COMM_GROUP:List[CommGroup] = None
_GLOBAL_SDP_WHOLE_COMM_GROUP:List[CommGroup] = None

def set_tp_whole_comm_group(whole_comm_group:List[CommGroup]):
    global _GLOBAL_TP_WHOLE_COMM_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_TP_WHOLE_COMM_GROUP, 'tp_whole_comm_group')
    _GLOBAL_TP_WHOLE_COMM_GROUP = whole_comm_group


def set_sp_whole_comm_group(whole_comm_group:List[CommGroup]):
    global _GLOBAL_SP_WHOLE_COMM_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_SP_WHOLE_COMM_GROUP, 'sp_whole_comm_group')
    _GLOBAL_SP_WHOLE_COMM_GROUP = whole_comm_group


def set_dp_whole_comm_group(whole_comm_group:List[CommGroup]):
    global _GLOBAL_DP_WHOLE_COMM_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_DP_WHOLE_COMM_GROUP, 'dp_whole_comm_group')
    _GLOBAL_DP_WHOLE_COMM_GROUP = whole_comm_group


def set_cp_whole_comm_group(whole_comm_group:List[CommGroup]):
    global _GLOBAL_CP_WHOLE_COMM_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_CP_WHOLE_COMM_GROUP, 'cp_whole_comm_group')
    _GLOBAL_CP_WHOLE_COMM_GROUP = whole_comm_group


def set_sdp_whole_comm_group(whole_comm_group:List[CommGroup]):
    global _GLOBAL_SDP_WHOLE_COMM_GROUP
    _ensure_var_is_not_initialized(_GLOBAL_SDP_WHOLE_COMM_GROUP, 'sdp_whole_comm_group')
    _GLOBAL_SDP_WHOLE_COMM_GROUP = whole_comm_group


def get_tp_whole_comm_group():
    global _GLOBAL_TP_WHOLE_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_TP_WHOLE_COMM_GROUP, 'tp_whole_comm_group')
    return _GLOBAL_TP_WHOLE_COMM_GROUP


def get_sp_whole_comm_group():
    global _GLOBAL_SP_WHOLE_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_SP_WHOLE_COMM_GROUP, 'sp_whole_comm_group')
    return _GLOBAL_SP_WHOLE_COMM_GROUP


def get_dp_whole_comm_group():
    global _GLOBAL_DP_WHOLE_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_DP_WHOLE_COMM_GROUP, 'dp_whole_comm_group')
    return _GLOBAL_DP_WHOLE_COMM_GROUP


def get_cp_whole_comm_group():
    global _GLOBAL_CP_WHOLE_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_CP_WHOLE_COMM_GROUP, 'cp_whole_comm_group')
    return _GLOBAL_CP_WHOLE_COMM_GROUP


def get_sdp_whole_comm_group():
    global _GLOBAL_SDP_WHOLE_COMM_GROUP
    _ensure_var_is_initialized(_GLOBAL_SDP_WHOLE_COMM_GROUP, 'sdp_whole_comm_group')
    return _GLOBAL_SDP_WHOLE_COMM_GROUP


# --- MoE Related Variables ---
_MOE_LAYER_WISE_LOGGING_TRACKER = {}

def get_moe_layer_wise_logging_tracker():
    global _MOE_LAYER_WISE_LOGGING_TRACKER
    return _MOE_LAYER_WISE_LOGGING_TRACKER


# --- FSDP2 Related Variables ---
def fsdp2_enabled() -> bool:
    args = get_args()
    return args.parallel.fsdp_version == "fsdp2"