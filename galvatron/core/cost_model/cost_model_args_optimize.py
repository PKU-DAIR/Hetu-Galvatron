from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum

@dataclass
class TrainArgsOptimize:
    """Basic training arguments"""
    seq_length: int = 1024
    hidden_size: int = 4096
    
    """Optimization related arguments"""
    mixed_precision: bool = False
    
    """Data parallel related arguments"""
    use_zero2_for_dp: bool = False
    async_grad_reduce: bool = True
    
    """Tensor/Sequence parallel related arguments"""
    disable_vtp: bool = False
    sequence_parallel: bool = False
    
    """Pipeline parallel related arguments"""
    pipeline_type: str = 'gpipe'
    
    
@dataclass
class ProfileModelArgsOptimize:
    """Profiled computation related arguments"""
    forward_computation_time: Optional[Union[float, list]] = 35 / 24
    other_time_profiled: Optional[Union[float, list]] = 0
    
    """Profiled memory related arguments"""
    parameter_memory: float = 2025.0928
    tp_activation_per_bsz_dict: dict = field(default_factory=lambda: {1:85, 2:47, 4:28, 8:18.5})
    other_memory_pp_off: dict = field(default_factory=lambda: {'model_states': 640, 'activation': 320})
    other_memory_pp_on: dict = field(default_factory=lambda: {'model_states': 640, 'activation': 320})

@dataclass
class ProfileHardwareArgsOptimize:
    """Communication Coefficients related arguments"""
    bct_fct_coe: float = 2
    overlap_slowdown_coe: float = 1.3

    """Communication related arguments"""
    comm_coe_dict: dict = field(default_factory=lambda: {'8': 0.0062326653993580354, '4_0': 0.006042551648710218, '4_1': 0.006087464692704782, '2_0': 0.006496332820123041, '2_1': 0.006424794567193714, '1': 0})
    p2p_comm_coe_dict: dict = field(default_factory=lambda: {2: 0.006787944610371979, 4: 0.0074923765069042254, 8: 0.00920674670398468})
    allreduce_dict: dict = field(default_factory=lambda: {})
    all2all_dict: dict = field(default_factory=lambda: {})
    
@dataclass
class UtilsArgsOptimize:
    """Utility related arguments"""
    extra_overhead: float = 0
    costmodel_coe: float = 1.0
    dummy_layer_num: int = 24
    pytorch_context_mem: int = 1024

class EstimateTPTimeType(Enum):
    FIXED = 1 
    FIT = 2        

@dataclass
class VersionOptionArgsOptimize:
    """Version and Iteration related arguments"""
    estimate_tp_time_type: EstimateTPTimeType = EstimateTPTimeType.FIXED
    zero_with_slight_noise: bool = True