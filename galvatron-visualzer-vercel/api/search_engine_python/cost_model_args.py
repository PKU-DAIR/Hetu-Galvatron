from dataclasses import dataclass, field
from typing import Optional, Callable, Union
import numpy as np


@dataclass
class ModelArgs:
    parameter_size: int = 48
    seq_length: int = 1024
    hidden_size: int = 4096
    layer_num:int = 16
    
@dataclass
class TrainArgs:
    mixed_precision: bool = False
    checkpoint: bool = False
    async_grad_reduce: bool = True
    pytorch_context_mem: int = 1024
    
@dataclass
class ParallelArgs:
    use_zero2_for_dp: bool = False
    disable_vtp: bool = False
    sequence_parallel: bool = False
    sp_space:str = 'sp+tp'
    
    pipeline_type: str = 'gpipe'
    optimal_chunk_func: Optional[Callable] = None
    chunks: Optional[int] = None
    
@dataclass
class ProfileModelArgs:
    tp_activation_per_bsz_dict:dict = field(default_factory=lambda: {1:85, 2:47, 4:28, 8:18.5})
    other_memory_pp_off:dict = field(default_factory=lambda: {'model_states': 640, 'activation': 320})
    other_memory_pp_on:dict = field(default_factory=lambda: {'first_stage':{'model_states': 640, 'activation': 320}, 'last_stage':{'model_states': 640, 'activation': 320}})
    forward_computation_time: Optional[Union[float, np.ndarray]] = 35 / 24
    other_time_profiled: Optional[Union[float, np.ndarray]] = 0
    
@dataclass
class ProfileHardwareArgs:
    bct_fct_coe: float = 2
    extra_overhead: float = 0
    comm_coe_dict: dict = field(default_factory=lambda: {'8': 0.0062326653993580354, '4_0': 0.006042551648710218, '4_1': 0.006087464692704782, '2_0': 0.006496332820123041, '2_1': 0.006424794567193714, '1': 0})
    dp_overlap_coe: float = 1.3
    bct_overlap_coe: float = 1.3
    p2p_comm_coe_dict: dict = field(default_factory=lambda: {2: 0.006787944610371979, 4: 0.0074923765069042254, 8: 0.00920674670398468})
    allreduce_dict: dict = field(default_factory=lambda: {})
    all2all_dict: dict = field(default_factory=lambda: {})
    costmodel_coe: float = 1.0
