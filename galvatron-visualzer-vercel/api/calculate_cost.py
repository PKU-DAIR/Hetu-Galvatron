import os
import sys
import typing
from dataclasses import dataclass, field
from search_engine_python.cost_model import TimeCostModel, MemoryCostModel, OtherTimeCostModel
from search_engine_python.cost_model_args import ModelArgs, TrainArgs, ParallelArgs, ProfileModelArgs, ProfileHardwareArgs
from typing import Optional, Dict, Any, List, Union

# 动态添加 search_engine_python 目录到 sys.path
current_dir = os.path.dirname(__file__)
search_engine_dir = os.path.join(current_dir, 'search_engine_python')
sys.path.append(search_engine_dir)
sys.path.append(os.path.join(search_engine_dir, '..')) # 添加 pages/api 目录，以便 search_engine_python 内部的相对导入（如果需要）

# 定义输入配置类型
@dataclass
class ModelConfig:
    """model config"""
    pp_size: int = 1
    tp_size: int = 8
    dp_size: int = 1
    global_batch_size: int = 8
    hidden_dim: int = 512
    num_layers: int = 12
    seq_length: int = 128
    vocab_size: int = 30522
    sequence_parallel: bool = True
    mixed_precision: bool = True
    micro_batch_size: int = 1
    attention_heads: int = 8
    ff_dim: int = 2048
    zero_stage: int = 0
    chunks: int = 8
    total_gpus: int = 8
    checkpoint: bool = False
    use_ulysses: bool = False
    stage_idx: int = 0

# 定义memory_config的类型, 对应search_engine.py中的self.memory_config, 都是从系统profiler中得到的原始数据
@dataclass
class RawMemoryProfilingConfig:
    model_config: Dict[str, Any]
    # 以下字段根据 sequence_parallel 的值可能存在不同的键
    layertype_0: Optional[Dict[str, Any]] = None
    layertype_0_sp: Optional[Dict[str, Any]] = None
    other_memory_pp_off: Optional[Dict[str, Any]] = None
    other_memory_pp_off_sp: Optional[Dict[str, Any]] = None
    other_memory_pp_on_first: Optional[Dict[str, Any]] = None
    other_memory_pp_on_first_sp: Optional[Dict[str, Any]] = None
    other_memory_pp_on_last: Optional[Dict[str, Any]] = None
    other_memory_pp_on_last_sp: Optional[Dict[str, Any]] = None

# TODO：定义time_config的类型, 对应search_engine.py中的self.time_config, 都是从系统profiler中得到的原始数据
@dataclass
class RawTimeProfilingConfig:
    # time profiling 结果是动态的 暂时不使用强自定义类型
    pass
   

# 定义输出结果类型
@dataclass
class MemoryCostResult:
    """对应 MemoryCostModel.js 中的 getMemoryCost() 返回结果"""
    num_layers: int
    stage_idx: int
    model_states: float
    parameter: float
    gradient: float
    grad_accumulate: float
    optimizer: float
    activation: float
    per_layer_parameter: float
    per_layer_activation: float
    other_memory_model_states: float
    other_memory_parameter: float
    other_memory_gradient: float
    other_memory_grad_accumulate: float
    other_memory_optimizer: float
    other_memory_activation: float
    total: float

@dataclass
class TimeCostResult:
    """对应 TimeCostModel 的计算结果"""
    forward_time: float          # 前向传播时间 (ms)
    backward_time: float         # 反向传播时间 (ms)
    dp_communication_time: float # 数据并行通信时间 (ms)
    tp_communication_time: float # 张量并行通信时间 (ms)
    pp_communication_time: float # 流水线并行通信时间 (ms)
    iteration_time: float        # 每次迭代总时间 (ms)
    samples_per_second: float    # 每秒处理的样本数（吞吐量）

@dataclass
class DeviceConfig:
    """Device configuration for time estimation"""
    device_type: str = "4090"
    device_factor: float = 0.8
    device_count: int = 8
    comm_efficiency: float = 0.85

def calculate_time(
    config: ModelConfig, 
    forward_computation_time: float = 10.0,
    bct_fct_coe: float = 2.0,
    dp_overlap_coe: float = 1.0,
    bct_overlap_coe: float = 1.0,
    allreduce_bandwidth: float = 100.0,
    p2p_bandwidth: float = 300.0,
    sp_space: str = 'tp+sp',
    async_grad_reduce: bool = False,
    device_count: int = 8
) -> TimeCostResult:
    """
    计算训练时间消耗
    参数:
        config: 模型配置参数
        forward_computation_time: 前向计算时间 (ms)
        bct_fct_coe: 后向/前向计算比例
        dp_overlap_coe: DP重叠系数
        bct_overlap_coe: BCT重叠系数
        allreduce_bandwidth: AllReduce带宽 (GB/s)
        p2p_bandwidth: P2P带宽 (GB/s)
        sp_space: 序列并行空间
        async_grad_reduce: 异步梯度归约
        device_count: 设备数量
        
    返回:
        时间消耗结果
    """
    
    print(f"Time calculation with hardware params:")
    print(f"  Forward time: {forward_computation_time} ms")
    print(f"  BCT/FCT ratio: {bct_fct_coe}")
    print(f"  DP overlap: {dp_overlap_coe}")
    print(f"  BCT overlap: {bct_overlap_coe}")
    print(f"  AllReduce BW: {allreduce_bandwidth} GB/s")
    print(f"  P2P BW: {p2p_bandwidth} GB/s")
    print(f"  Device count: {device_count}")
    
    # Use the provided forward computation time directly
    forward_time_per_layer = forward_computation_time
    
    # Ensure forward_time_per_layer is not None
    if forward_time_per_layer is None:
        forward_time_per_layer = 10.0  # Default fallback
    
    # Calculate computation times
    forward_time = forward_time_per_layer * config.num_layers
    backward_time = forward_time * bct_fct_coe  # Use provided BCT/FCT ratio
    
    # Communication costs using provided bandwidth parameters
    dp_communication_time = 0
    if config.dp_size > 1:
        # DP communication estimate using AllReduce bandwidth
        param_size_gb = config.hidden_dim * config.hidden_dim * 4 * config.num_layers / (1024 * 1024 * 1024)
        dp_communication_time = param_size_gb / allreduce_bandwidth * 1000 * dp_overlap_coe  # Convert to ms
    
    tp_communication_time = 0
    if config.tp_size > 1:
        # TP communication estimate using P2P bandwidth
        activation_size_gb = config.seq_length * config.hidden_dim * config.micro_batch_size * 4 / (1024 * 1024 * 1024)
        tp_communication_time = activation_size_gb / p2p_bandwidth * 1000 * config.num_layers  # Convert to ms
    
    pp_communication_time = 0
    if config.pp_size > 1:
        # PP communication estimate using P2P bandwidth
        activation_size_gb = config.seq_length * config.hidden_dim * config.micro_batch_size * 4 / (1024 * 1024 * 1024)
        pp_communication_time = activation_size_gb / p2p_bandwidth * 1000  # Convert to ms
    
    # Total iteration time with overlap modeling
    total_comp_time = forward_time + backward_time
    total_comm_time = dp_communication_time + tp_communication_time + pp_communication_time
    
    # Apply overlap coefficients
    overlap_factor = min(dp_overlap_coe, bct_overlap_coe) * 0.5  # Conservative overlap
    
    # Iteration time with overlap
    iteration_time = max(total_comp_time, total_comm_time) + min(total_comp_time, total_comm_time) * (1 - overlap_factor)
    
    # Throughput calculation (samples/sec)
    samples_per_second = config.global_batch_size / (iteration_time / 1000)
    
    # Print diagnostic information
    print(f"Time calculation results:")
    print(f"  Device count: {device_count}")
    print(f"  Forward time: {forward_time} ms")
    print(f"  Backward time: {backward_time} ms")
    print(f"  DP communication: {dp_communication_time} ms")
    print(f"  TP communication: {tp_communication_time} ms")
    print(f"  PP communication: {pp_communication_time} ms")
    print(f"  Total iteration time: {iteration_time} ms")
    print(f"  Throughput: {samples_per_second} samples/sec")
    
    return TimeCostResult(
        forward_time=forward_time,
        backward_time=backward_time,
        dp_communication_time=dp_communication_time,
        tp_communication_time=tp_communication_time,
        pp_communication_time=pp_communication_time,
        iteration_time=iteration_time,
        samples_per_second=samples_per_second
    )

# not used yet. stick with the currnet js calcualation backend 
def calculate_memory(config: ModelConfig, raw_memory_config: Optional[Dict[str, Any]] = None) -> MemoryCostResult:
    """
    计算内存消耗
    参数:
        config: 配置参数
        raw_memory_config: 原始关于memory_profiling的配置数据
        
    返回:
        内存消耗结果，对应 MemoryCostModel.js 中的 getMemoryCost() 返回值
    """
    
    if raw_memory_config is None:
        raise ValueError("Missing required raw configuration data. Please upload a Galvatron config file.")
    
    # Prepare arguments for MemoryCostModel
    # Create model_args
    model_args = ModelArgs(
        parameter_size=0,  # Will be set after parsing raw config
        hidden_size=config.hidden_dim,
        seq_length=config.seq_length,
        layer_num=config.num_layers,
    )
    
    # Create train_args
    train_args = TrainArgs(
        mixed_precision=config.mixed_precision,
        async_grad_reduce=False,
        pytorch_context_mem=0,
    )
    
    # Create parallel_args
    parallel_args = ParallelArgs(
        use_zero2_for_dp=(config.zero_stage == 2),
        disable_vtp=False,
        sequence_parallel=config.sequence_parallel,
        sp_space='tp+sp' if config.sequence_parallel else 'tp',
        pipeline_type='gpipe',
        optimal_chunk_func=None,
        chunks=config.chunks,
    )
    
    # Determine layer type based on sequence_parallel
    layer_type = "layertype_0_sp" if config.sequence_parallel else "layertype_0"
    other_layer_type_pp_off = "other_memory_pp_off_sp" if config.sequence_parallel else "other_memory_pp_off"
    other_layer_type_pp_on_first = "other_memory_pp_on_first_sp" if config.sequence_parallel else "other_memory_pp_on_first"
    other_layer_type_pp_on_last = "other_memory_pp_on_last_sp" if config.sequence_parallel else "other_memory_pp_on_last"
    
    # Check if required configs exist in raw_memory_config
    if layer_type not in raw_memory_config:
        raise ValueError(f"{layer_type} not found in raw_memory_config")
    if other_layer_type_pp_off not in raw_memory_config:
        raise ValueError(f"{other_layer_type_pp_off} not found in raw_memory_config")
    if other_layer_type_pp_on_first not in raw_memory_config:
        raise ValueError(f"{other_layer_type_pp_on_first} not found in raw_memory_config")
    if other_layer_type_pp_on_last not in raw_memory_config:
        raise ValueError(f"{other_layer_type_pp_on_last} not found in raw_memory_config")
    
    # Get sequence lengths from raw config
    seq_lengths = list(map(int, raw_memory_config[layer_type].keys()))
    max_seq_length = max(seq_lengths)
    
    # Prepare profile model args
    layer_config = raw_memory_config[layer_type][str(max_seq_length)]
    parameter_size = layer_config["parameter_size"]
    tp_activation_per_bsz_dict = layer_config["tp_activation_per_bsz_dict"].copy()
    
    # Scale activation sizes based on sequence length ratio
    for tp in tp_activation_per_bsz_dict:
        tp_activation_per_bsz_dict[tp] = tp_activation_per_bsz_dict[tp] * config.seq_length / max_seq_length
    
    # Get other memory configurations
    other_memory_pp_off = raw_memory_config[other_layer_type_pp_off][str(max_seq_length)]
    other_memory_pp_on_first = raw_memory_config[other_layer_type_pp_on_first][str(max_seq_length)]
    other_memory_pp_on_last = raw_memory_config[other_layer_type_pp_on_last][str(max_seq_length)]
    
    # Scale parameter sizes for other memory (model_states is 4x parameter size)
    for tp in other_memory_pp_off["model_states"]:
        other_memory_pp_off["model_states"][tp] = other_memory_pp_off["model_states"][tp] * config.seq_length / max_seq_length
    
    for tp in other_memory_pp_on_first["model_states"]:
        other_memory_pp_on_first["model_states"][tp] = other_memory_pp_on_first["model_states"][tp] * config.seq_length / max_seq_length
    
    for tp in other_memory_pp_on_last["model_states"]:
        other_memory_pp_on_last["model_states"][tp] = other_memory_pp_on_last["model_states"][tp] * config.seq_length / max_seq_length
    
    # Scale activation sizes for other memory
    for tp in other_memory_pp_off["activation"]:
        other_memory_pp_off["activation"][tp] = other_memory_pp_off["activation"][tp] * config.seq_length / max_seq_length * 2/3
    
    for tp in other_memory_pp_on_first["activation"]:
        other_memory_pp_on_first["activation"][tp] = other_memory_pp_on_first["activation"][tp] * config.seq_length / max_seq_length
    
    for tp in other_memory_pp_on_last["activation"]:
        other_memory_pp_on_last["activation"][tp] = other_memory_pp_on_last["activation"][tp] * config.seq_length / max_seq_length * 1/2
    
    # Set parameter_size in model_args
    model_args.parameter_size = parameter_size
    
    # Create profile_model_args
    profile_model_args = ProfileModelArgs(
        tp_activation_per_bsz_dict=tp_activation_per_bsz_dict,
        other_memory_pp_off=other_memory_pp_off,
        other_memory_pp_on={
            'first_stage': other_memory_pp_on_first,
            'last_stage': other_memory_pp_on_last
        },
        forward_computation_time=None,  # Not needed for memory calculation
        other_time_profiled=None,       # Not needed for memory calculation
    )
    
    # Create strategy array in format [pp_size, tp_size, dp_size, dict]
    strategy = [
        config.pp_size,
        config.tp_size,
        config.dp_size,
        {'sp': 1 if config.sequence_parallel else 0, 'cpt': 1 if config.checkpoint else 0, 'fsdp': 0}
    ]
    
    # Create MemoryCostModel instance
    mem_model = MemoryCostModel(
        strategy,
        global_batch_size=config.global_batch_size,
        mbsz=config.micro_batch_size,
        min_tp=1,  # Default value
        max_tp=config.tp_size,  # Use tp_size as max_tp
        stage_idx=config.stage_idx,
        vsp=0,  # Default value
        embed_sdp=False,  # Default value
        model_args=model_args,
        train_args=train_args,
        parallel_args=parallel_args,
        profile_model_args=profile_model_args
    )
    
    # Get memory cost results
    mem_cost = mem_model.get_memory_cost()
    
    # For calculating memory components
    mixed_precision = config.mixed_precision
    if config.use_ulysses:
        sdp_size = config.tp_size * config.dp_size
    else:
        sdp_size = config.dp_size
    
    # Helper functions for ZeRO ratios
    def zero1_ratio(d):
        if mixed_precision:
            return 6/8 * (1/d) + 2/8
        return 2/4 * (1/d) + 2/4
    
    def zero2_ratio(d):
        if mixed_precision:
            if config.chunks > 1:
                return 9/10 * (1/d) + 1/10
            return 7/8 * (1/d) + 1/8
        return 3/4 * (1/d) + 1/4
    
    def zero3_ratio(d):
        return 1/d
        
    # Calculate total parameter, activation, and model states
    model_states_size = mem_cost['model_states']
    activation_size = mem_cost['activation']
    parameter_size = mem_cost['parameter']
    
    # Calculate number of layers per stage
    num_layers = config.num_layers // config.pp_size if config.pp_size > 0 else config.num_layers
    
    # Calculate memory components based on ZeRO stage
    if not mixed_precision:
        param_mem = parameter_size
        grad_mem = parameter_size
        optimizer_mem = 2 * parameter_size
        grad_accumulate_mem = 0
    else:
        param_mem = parameter_size / 2
        grad_mem = parameter_size / 2
        optimizer_mem = 3 * parameter_size
        grad_accumulate_mem = config.chunks == 1 and 0 or parameter_size
    
    # Apply ZeRO optimizations
    if config.zero_stage >= 1:
        optimizer_mem /= sdp_size
    if config.zero_stage >= 2:
        grad_mem /= sdp_size
        grad_accumulate_mem /= sdp_size
    if config.zero_stage >= 3:
        param_mem /= sdp_size
    
    # Handle chunks > 1 and zero_stage <= 1
    if config.chunks > 1 and config.zero_stage <= 1:
        grad_mem = 0
    
    # Get other memory costs
    other_memory_cost = mem_cost['other'][1] if 1 in mem_cost['other'] else [0]
    other_memory_model_states = other_memory_cost[config.stage_idx]
    
    # Calculate other memory components
    if not mixed_precision:
        other_param_mem = other_memory_model_states / 4
        other_grad_mem = other_memory_model_states / 4
        other_optimizer_mem = other_memory_model_states / 2
        other_grad_accumulate_mem = 0
    else:
        other_param_mem = other_memory_model_states / 8
        other_grad_mem = other_memory_model_states / 8
        other_optimizer_mem = other_memory_model_states * 3/4
        other_grad_accumulate_mem = config.chunks == 1 and 0 or other_memory_model_states / 4
    
    # Apply ZeRO optimizations to other memory
    if config.zero_stage >= 1:
        other_optimizer_mem /= sdp_size
    if config.zero_stage >= 2:
        other_grad_mem /= sdp_size
        other_grad_accumulate_mem /= sdp_size
    if config.zero_stage >= 3:
        other_param_mem /= sdp_size
    
    # Handle chunks > 1 and zero_stage <= 1 for other memory
    if config.chunks > 1 and config.zero_stage <= 1:
        other_grad_mem = 0
    
    # Calculate other activation memory
    other_memory_activation = 0
    
    # Calculate total memory
    total_model_states = num_layers * model_states_size + other_memory_model_states
    total_activation = num_layers * activation_size + other_memory_activation
    
    # Adjust activation if pipeline parallelism is used
    if config.pp_size > 1:
        # Adjust based on stage_idx - similar to the "act_1f1b_ratio" in JS implementation
        total_activation = (config.pp_size - config.stage_idx - 1) * num_layers * activation_size + total_activation
    
    total_param_mem = num_layers * param_mem + other_param_mem
    total_grad_mem = num_layers * grad_mem + other_grad_mem
    total_optimizer_mem = num_layers * optimizer_mem + other_optimizer_mem
    total_grad_accumulate_mem = num_layers * grad_accumulate_mem + other_grad_accumulate_mem
    
    total_mem = total_model_states + total_activation
    
    # Return memory cost result
    return MemoryCostResult(
        num_layers=num_layers,
        stage_idx=config.stage_idx,
        model_states=total_model_states,
        parameter=total_param_mem,
        gradient=total_grad_mem,
        grad_accumulate=total_grad_accumulate_mem,
        optimizer=total_optimizer_mem,
        activation=total_activation,
        per_layer_parameter=param_mem,
        per_layer_activation=activation_size,
        other_memory_model_states=other_memory_model_states,
        other_memory_parameter=other_param_mem,
        other_memory_gradient=other_grad_mem,
        other_memory_grad_accumulate=other_grad_accumulate_mem,
        other_memory_optimizer=other_optimizer_mem,
        other_memory_activation=other_memory_activation,
        total=total_mem
    )

