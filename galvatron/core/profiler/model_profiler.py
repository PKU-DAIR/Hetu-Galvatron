import copy
import os
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from galvatron.utils.config_utils import array2str, num2str, read_json_config, str2array, write_json_config

from .base_profiler import BaseProfiler
from galvatron.core.profiler.args_schema import GalvatronModelProfilerArgs


class ModelProfiler(BaseProfiler):
    """Model profiler for analyzing model performance characteristics including computation and memory usage"""

    def __init__(self, args: GalvatronModelProfilerArgs):
        """Initialize model profiler

        Args:
            args: Arguments containing profiling configuration including:
                - profile_mode: Profiling mode ('static', 'batch', or 'sequence')
                - profile_type: Type of profiling ('computation' or 'memory')
                - profile_fixed_batch_size: Batch size for static profiling
                - profile_min/max_batch_size: Range for batch size profiling
                - profile_min/max_seq_length: Range for sequence length profiling
                - profile_batch/seq_length_step: Step size for profiling
        """
        super().__init__()
        self.args = args

        self.global_batch_size_list = None
        self.layernum_tuple_list = None
        self.seq_length_tuple_list = None
        self.basic_overrides_dict = None
        self.envs_dict = None

        self.num_layertype = 1 # TODO: modify this trick


    def set_profiler_launcher(self, path: str, model_name: Optional[str] = None,) -> None:
        """Set up profiler launcher configuration

        Args:
            path: Path to profiling scripts
            layernum_arg_names: Names of arguments specifying number of layers
            model_name: Name of the model being profiled
            seqlen_arg_names: Names of arguments specifying sequence lengths
            layernum_listed: Whether layer numbers are provided as a list
        """
        args = self.args

        self.set_work_dir(path)
        self.set_model_name(model_name)
        self.set_profile_unit(args.profile_unit)
        self.set_mixed_precision(args.profile_mixed_precision)

    # =============== Necessary initialization Functions ===============
    def get_global_batch_size_list(self) -> List[int]:
        if self.global_batch_size_list == None:
            args = self.args
            
            if args.profile_mode == 'static':
                assert args.profile_fixed_batch_size is not None, f"profile_fixed_batch_size is not set for static mode"
                self.global_batch_size_list = [args.profile_fixed_batch_size]
            elif args.profile_mode == 'batch':
                assert args.profile_min_batch_size is not None and args.profile_max_batch_size is not None and args.profile_batch_size_step is not None, f"profile_min_batch_size, profile_max_batch_size, and profile_batch_size_step are not set for batch mode"
                self.global_batch_size_list = list(range(args.profile_min_batch_size, args.profile_max_batch_size + 1, args.profile_batch_size_step))
            elif args.profile_mode == 'sequence':
                assert args.profile_fixed_batch_size is not None, f"profile_fixed_batch_size is not set for sequence mode"
                self.global_batch_size_list = [args.profile_fixed_batch_size]
            
        return self.global_batch_size_list

    def get_layernum_tuple_list(self) -> Union[List[Tuple[int]], List[Tuple[int, int]]]:
        if self.layernum_tuple_list is None:
            args = self.args
            assert args.profile_layernum_min is not None and args.profile_layernum_max is not None, f"profile_layernum_min and profile_layernum_max are not set"
            
            if self.num_layertype == 1: # decoder-only or encoder-only
                self.layernum_tuple_list = [
                    (args.profile_layernum_min, ),
                    (args.profile_layernum_max, )
                ]
            else: # encoder-decoder
                self.layernum_tuple_list = [
                    (args.profile_layernum_min, args.profile_layernum_min),
                    (args.profile_layernum_max, args.profile_layernum_min),
                    (args.profile_layernum_min, args.profile_layernum_max),
                ]

        return self.layernum_tuple_list

    def get_seq_length_tuple_list(self) -> Union[List[Tuple[int]], List[Tuple[int, int]]]:
        if self.seq_length_tuple_list is None:
            args = self.args

            if self.num_layertype == 1: # decoder-only or encoder-only
                if args.profile_mode == 'static' or args.profile_mode == 'batch':
                    assert args.profile_fixed_seq_length_list is not None, f"profile_fixed_seq_length_list is not set for static or batch mode"
                    assert len(args.profile_fixed_seq_length_list) == 1, f"profile_fixed_seq_length_list should have only one element for decoder-only or encoder-only model"
                    self.seq_length_tuple_list = [
                        (args.profile_fixed_seq_length_list[0],),
                    ]
                elif args.profile_mode == 'sequence':
                    if args.profile_type == 'computation':
                        assert args.profile_min_seq_length is not None and args.profile_max_seq_length is not None and args.profile_seq_length_step is not None, f"profile_min_seq_length, profile_max_seq_length, and profile_seq_length_step are not set for computation mode and sequence mode"
                        seq_length_all_case = list(range(args.profile_min_seq_length, args.profile_max_seq_length + 1, args.profile_seq_length_step))
                    elif args.profile_type == 'memory':
                        assert args.profile_min_seq_length is not None and args.profile_max_seq_length is not None, f"profile_min_seq_length and profile_max_seq_length are not set for memory mode and sequence mode"
                        # For memory profiling, sequence lengths must be powers of 2
                        assert (1 << (args.profile_min_seq_length.bit_length() - 1)) == args.profile_min_seq_length, "profile_min_seq_length must be a power of 2"
                        assert (1 << (args.profile_max_seq_length.bit_length() - 1)) == args.profile_max_seq_length, "profile_max_seq_length must be a power of 2"
                        # Include max power-of-two sequence length in memory sequence profiling.
                        seq_length_all_case = [
                            (1 << shift)
                            for shift in range(
                                args.profile_min_seq_length.bit_length() - 1,
                                args.profile_max_seq_length.bit_length(),
                            )
                        ]
                    self.seq_length_tuple_list = [
                        (seq_length, ) for seq_length in seq_length_all_case
                    ]
            else:
                if args.profile_mode == 'static' or args.profile_mode == 'batch':
                    assert args.profile_fixed_seq_length_list is not None, f"profile_fixed_seq_length_list is not set for static or batch mode"
                    assert len(args.profile_fixed_seq_length_list) == 2, f"profile_fixed_seq_length_list should have two elements for encoder-decoder model"
                    self.seq_length_tuple_list = [
                        (args.profile_fixed_seq_length_list[0], args.profile_fixed_seq_length_list[1])
                    ]
                elif args.profile_mode == 'sequence':
                    raise NotImplementedError("Sequence profiling is not supported for encoder-decoder model")

        return self.seq_length_tuple_list

    def get_basic_overrides_dict(self) -> Dict[str, Any]:
        if self.basic_overrides_dict is None:
            args = self.args
            if args.profile_type == 'computation':
                self.basic_overrides_dict = {
                    'runtime.parallel.pp_deg': 1,
                    'runtime.parallel.global_tp_deg': 1,
                    'runtime.parallel.global_cp_deg': 1,
                    'runtime.parallel.global_checkpoint': 0,
                    'runtime.parallel.vocab_tp': 1,
                    'runtime.parallel.vocab_cp': 1,
                    'runtime.parallel.default_dp_type': 'ddp',
                    'runtime.parallel.sdp':0,
                    'runtime.parallel.pipeline_type': 'gpipe',
                    'runtime.parallel.mixed_precision': args.profile_mixed_precision,

                    'runtime.train.chunks': 1,
                    'runtime.train.use_flash_attn': True,
                    'runtime.train.sequence_parallel': True,
                    
                    'runtime.profile.profile': 1,
                    'runtime.profile.profile_mode': args.profile_mode,
                    'runtime.profile.profile_unit': args.profile_unit,
                    'runtime.profile.profile_forward': 1,

                    'runtime.model.model_size': args.model_info.model_size,
                    'runtime.model.is_moe_model': args.model_info.is_moe_model,
                    'runtime.model.model_config_path': args.model_info.model_config_path,
                    'runtime.model.set_layernum_manually': 1,
                    'runtime.model.set_seqlen_manually': 1,
                }
            else:
                global_batch_size_list = self.get_global_batch_size_list()
                assert len(global_batch_size_list) == 1
                self.basic_overrides_dict = {
                    'runtime.parallel.default_dp_type': args.profile_dp_type,
                    'runtime.parallel.pipeline_type': 'gpipe',
                    'runtime.parallel.mixed_precision': args.profile_mixed_precision,

                    'runtime.train.global_batch_size': global_batch_size_list[0],
                    'runtime.train.chunks': 1,
                    'runtime.train.use_flash_attn': True,
                    'runtime.train.sequence_parallel': True,

                    'runtime.profile.profile': 1,
                    'runtime.profile.profile_mode': args.profile_mode,
                    'runtime.profile.profile_unit': args.profile_unit,
                    'runtime.profile.profile_forward': 0,
                    'runtime.profile.save_profiled_memory': 1,

                    'runtime.model.model_size': args.model_info.model_size,
                    'runtime.model.is_moe_model': args.model_info.is_moe_model,
                    'runtime.model.model_config_path': args.model_info.model_config_path,
                    'runtime.model.set_layernum_manually': 1,
                    'runtime.model.set_seqlen_manually': 1,
                }
        
        return self.basic_overrides_dict

    def get_envs_dict(self) -> Dict[str, Any]:
        if self.envs_dict is None:
            # TODO: Verify that all required fields are complete.
            self.envs_dict = {
                'CUDA_DEVICE_MAX_CONNECTIONS': 1,
            }

        return self.envs_dict

    def dict_to_str(self, d: dict, sep: str = "=") -> str:
        string = ""
        for key, value in d.items():
            string += f"{key}{sep}{value} "
        return string

    # =============== For Launching Profiling Scripts ===============
    def launch_profiling_scripts(self) -> None:
        """Launch profiling scripts for memory or computation profiling

        This method handles:
        1. Memory profiling with different tensor parallelism and pipeline parallelism settings
        2. Computation profiling with different batch sizes and sequence lengths

        Note:
            Memory profiling only supports sequence or static profile modes
        """
        args = self.args
        if args.profile_type == "memory":
            self._launch_memory_profiling()
        elif args.profile_type == "computation":
            self._launch_computation_profiling()

    def _launch_memory_profiling(self) -> None:
        assert self.num_layertype == 1, "Currently only support one layer type for memory profiling"
        assert self.args.profile_mode == "sequence" or self.args.profile_mode == "static", "Memory profiling only supports sequence or static profile mode"

        if self.args.profile_flow_control == "data_only":
            return
        
        args = self.args

        num_nodes = int(os.getenv('NUM_NODES', -1))
        num_gpus_per_node = int(os.getenv('NUM_GPUS_PER_NODE', -1))
        assert num_nodes != -1 and num_gpus_per_node != -1, "NUM_NODES and NUM_GPUS_PER_NODE are not set"
        world_size = num_nodes * num_gpus_per_node
        max_tp_deg = min(world_size, args.profile_max_tp_deg) if args.profile_mode == 'static' else 1

        layernum_tuple_list = self.get_layernum_tuple_list()     
        seq_length_tuple_list = self.get_seq_length_tuple_list()
        envs_dict = self.get_envs_dict()
        basic_overrides_dict = self.get_basic_overrides_dict()

        log_dir = os.path.join(self.work_dir, "logs/profile_memory")
        os.makedirs(log_dir, exist_ok=True)

        cmd_list = []

        runtime_launcher = os.getenv("RUNTIME_LAUNCHER", None)
        assert runtime_launcher is not None, "RUNTIME_LAUNCHER is not set"
        
        # case1: no pipeline parallelism, only tensor parallelism, no checkpoint
        for seq_length_tuple in seq_length_tuple_list:
            tp_deg = 1
            while tp_deg <= max_tp_deg:
                for enable_vocab_tp in [0, 1]:
                    if tp_deg == 1 and enable_vocab_tp == 1:
                        continue
                    for layernum_tuple in layernum_tuple_list:
                        extra_overrides_dict = {
                            'runtime.parallel.pp_deg': 1, # no pipeline parallelism
                            'runtime.parallel.global_tp_deg': tp_deg,
                            'runtime.parallel.global_checkpoint': 0, # no checkpoint
                            'runtime.parallel.vocab_tp': tp_deg if enable_vocab_tp == 1 else 1,
                            'runtime.model.num_layers': layernum_tuple[0], # decoder-only or encoder-only
                            'runtime.train.seq_length': seq_length_tuple[0], # decoder-only or encoder-only
                        }
                        extra_overrides_dict.update(basic_overrides_dict)
                        log_name = f'pp1_tp{tp_deg}_vocab{enable_vocab_tp}_ckpt0_layernum{layernum_tuple[0]}_seq{seq_length_tuple[0]}'
                        envs_string = self.dict_to_str(envs_dict, sep='=')
                        overrides_string = self.dict_to_str(extra_overrides_dict, sep='=')
                        cmd = f"{envs_string} {runtime_launcher} {self.args.runtime_yaml_template_path} {overrides_string} 2>&1 | tee {log_dir}/{log_name}.log"
                        cmd_list.append(cmd)
                tp_deg *= 2
        
        # case2: no pipeline parallelism, no tensor parallelism, only checkpoint
        for seq_length_tuple in seq_length_tuple_list:
            for layernum_tuple in layernum_tuple_list:
                extra_overrides_dict = {
                    'runtime.parallel.pp_deg': 1, # no pipeline parallelism
                    'runtime.parallel.global_tp_deg': 1, # no tensor parallelism
                    'runtime.parallel.global_checkpoint': 1, # only checkpoint
                    'runtime.parallel.vocab_tp': 1, # no vocabulary parallelism
                    'runtime.model.num_layers': layernum_tuple[0], # decoder-only or encoder-only
                    'runtime.train.seq_length': seq_length_tuple[0], # decoder-only or encoder-only
                }
                extra_overrides_dict.update(basic_overrides_dict)
                log_name = f'pp1_tp1_vocab1_ckpt1_layernum{layernum_tuple[0]}_seq{seq_length_tuple[0]}'
                envs_string = self.dict_to_str(envs_dict, sep='=')
                overrides_string = self.dict_to_str(extra_overrides_dict, sep='=')
                cmd = f"{envs_string} {runtime_launcher} {self.args.runtime_yaml_template_path} {overrides_string} 2>&1 | tee {log_dir}/{log_name}.log"
                cmd_list.append(cmd)

        # case3: pipeline parallelism, tensor parallelism, no checkpoint
        for seq_length_tuple in seq_length_tuple_list:
            for pp_deg in [2, 4]:
                layer_num = pp_deg # At this point, each stage has exactly one layer.
                tp_deg = 1
                while tp_deg <= max_tp_deg:
                    if pp_deg * tp_deg <= world_size:
                        for enable_vocab_tp in [0, 1]:
                            if tp_deg == 1 and enable_vocab_tp == 1:
                                continue
                            
                            extra_overrides_dict = {
                                'runtime.parallel.pp_deg': pp_deg, # pipeline parallelism
                                'runtime.parallel.global_tp_deg': tp_deg, # tensor parallelism
                                'runtime.parallel.global_checkpoint': 0, # no checkpoint
                                'runtime.parallel.vocab_tp': tp_deg if enable_vocab_tp == 1 else 1,
                                'runtime.model.num_layers': layer_num,
                                'runtime.train.seq_length': seq_length_tuple[0], # decoder-only or encoder-only
                            }
                            extra_overrides_dict.update(basic_overrides_dict)
                            log_name = f'pp{pp_deg}_tp{tp_deg}_vocab{enable_vocab_tp}_ckpt0_layernum{layer_num}_seq{seq_length_tuple[0]}'
                            envs_string = self.dict_to_str(envs_dict, sep='=')
                            overrides_string = self.dict_to_str(extra_overrides_dict, sep='=')
                            cmd = f"{envs_string} {runtime_launcher} {self.args.runtime_yaml_template_path} {overrides_string} 2>&1 | tee {log_dir}/{log_name}.log"
                            cmd_list.append(cmd)
                    tp_deg *= 2
                    
        if self.args.profile_flow_control == "scripts_only":
            for cmd in cmd_list:
                print(cmd)
            print("Start to write memory profiling scripts ...")
            script_path = os.path.join(self.work_dir, f"scripts/memory_profile_scripts_{self.args.profile_unit}.sh")
            with open(script_path, "w") as f:
                for cmd in cmd_list:
                    f.write(cmd + "\n")
                    f.write("sleep 1\n")
            print(f"Memory profiling scripts have been written to {script_path}!")
        else:
            for cmd in cmd_list:
                print(cmd)
                os.system(cmd)
                
    def _launch_computation_profiling(self) -> None:
        assert self.num_layertype == 1, "Currently only support one layer type for computation profiling"

        if self.args.profile_flow_control == "data_only":
            return
        
        runtime_launcher = os.getenv("RUNTIME_LAUNCHER", None)
        assert runtime_launcher is not None, "RUNTIME_LAUNCHER is not set"
        
        global_batch_size_list = self.get_global_batch_size_list()
        layernum_tuple_list = self.get_layernum_tuple_list()     
        seq_length_tuple_list = self.get_seq_length_tuple_list()
        envs_dict = self.get_envs_dict()
        basic_overrides_dict = self.get_basic_overrides_dict()

        log_dir = os.path.join(self.work_dir, "logs/profile_computation")
        os.makedirs(log_dir, exist_ok=True)

        cmd_list = []
        for gbsz in global_batch_size_list:
            for layernum_tuple in layernum_tuple_list:
                for seq_length_tuple in seq_length_tuple_list:
                    extra_overrides_dict = {
                        'runtime.train.global_batch_size': gbsz,
                        'runtime.model.num_layers': layernum_tuple[0], # decoder-only or encoder-only
                        'runtime.train.seq_length': seq_length_tuple[0], # decoder-only or encoder-only
                    }
                    extra_overrides_dict.update(basic_overrides_dict)

                    log_name = f"layernum_{layernum_tuple[0]}_seq_{seq_length_tuple[0]}_gbsz_{gbsz}"
                    envs_string = self.dict_to_str(envs_dict, sep='=')
                    overrides_string = self.dict_to_str(extra_overrides_dict, sep='=')
                    cmd = f"{envs_string} {runtime_launcher} {self.args.runtime_yaml_template_path} {overrides_string} 2>&1 | tee {log_dir}/{log_name}.log"
                    cmd_list.append(cmd)

        if self.args.profile_flow_control == "scripts_only":
            for cmd in cmd_list:
                print(cmd)
            print("Start to write computation profiling scripts ...")
            script_path = os.path.join(self.work_dir, f"scripts/computation_profile_scripts_{self.args.profile_unit}.sh")
            with open(script_path, "w") as f:
                for cmd in cmd_list:
                    f.write(cmd + "\n")
                    f.write("sleep 1\n")
            print(f"Computation profiling scripts have been written to {script_path}!")
        else:
            for cmd in cmd_list:
                print(cmd)
                os.system(cmd)

    # =============== For Processing Profiled Memory and Time ===============
    def process_profiled_data(self) -> None:
        """Process profiled data for both computation and memory profiling

        This method handles two types of profiling data:
        1. Computation profiling:
            - Calculates average computation time per layer type
            - Processes batch size and sequence length variations
            - Accounts for other computation overhead

        2. Memory profiling:
            - Processes parameter and activation memory usage
            - Handles different parallelism strategies (TP, PP)
            - Calculates memory overhead for different configurations

        The results are written to corresponding config files:
        - Computation results: time_config_path
        - Memory results: memory_config_path
        """
        env_args = self.env_args()
        world_size = int(env_args["NUM_NODES"]) * int(env_args["NUM_GPUS_PER_NODE"])
        layernum_lists = [list(layernum_tuple) for layernum_tuple in self.get_layernum_tuple_list()]
        args = self.args

        if args.profile_type == "computation":
            self._process_computation_data(layernum_lists)
        elif args.profile_type == "memory":
            self._process_memory_data(world_size, layernum_lists)

    def _process_computation_data(self, layernum_lists: List[List[int]]) -> None:
        """Process computation profiling data

        Args:
            layernum_lists: Lists of layer numbers for different configurations

        This method:
        1. Reads profiled computation time data
        2. Calculates per-layer computation time for each layer type
        3. Processes results for different batch sizes and sequence lengths
        4. Writes processed results to config file
        """
        if self.args.profile_flow_control == "scripts_only" or self.args.profile_flow_control == "launch_only":
            return
        
        time_config_path = self.time_profiling_path()
        config = read_json_config(time_config_path)
        batch_size_list = self.get_global_batch_size_list()
        sequence_length_list = self.get_seq_length_tuple_list()

        for bsz in batch_size_list:
            for seq in sequence_length_list:
                # Process base configuration
                seq_info = num2str(list(seq), "seq")
                key_base = self.key_format(layernum_lists[0], bsz, seq_info)
                val_base = config[key_base]
                total_avg_time = []

                # Calculate per-layer computation time for each layer type
                for idx, layernum in enumerate(layernum_lists[1:]):
                    key = self.key_format(layernum, bsz, seq_info)
                    val = config[key]
                    avg_time = (val - val_base) / bsz / (
                        self.args.profile_layernum_max - self.args.profile_layernum_min
                    )
                    write_key = f"layertype_{idx}_bsz{bsz}_seq{seq[idx]}"
                    config[write_key] = avg_time
                    total_avg_time.append(avg_time)

                # Calculate other computation overhead
                other_time = val_base
                for idx in range(len(total_avg_time)):
                    other_time -= layernum_lists[0][idx] * total_avg_time[idx] * bsz
                other_time /= bsz
                write_key = f"layertype_other_bsz{bsz}_{seq_info}"
                config[write_key] = max(other_time, 0)

                # Write results to config file
                write_json_config(config, time_config_path)
                print(f"Already written processed computation time into env config file {time_config_path}!\n")

    def _process_memory_data(self, world_size: int, layernum_lists: List[List[int]]) -> None:
        """Process memory profiling data

        Args:
            world_size: Total number of GPUs
            layernum_lists: Lists of layer numbers for different configurations

        This method:
        1. Processes parameter and activation memory usage
        2. Handles different parallelism strategies:
            - Tensor Parallelism (TP)
            - Pipeline Parallelism (PP)
            - Sequence Parallelism (SP)
        3. Calculates memory overhead for different configurations
        4. Writes processed results to config file

        Note:
            Only supports sequence or static profile modes
        """
        if self.args.profile_flow_control == "scripts_only" or self.args.profile_flow_control == "launch_only":
            return
        
        assert (
            self.args.profile_mode == "static" or self.args.profile_mode == "sequence"
        ), "Memory profiling only support sequence or static profile mode."

        memory_config_path = self.memory_profiling_path()
        config = read_json_config(memory_config_path)

        # Initialize parameters
        assert self.args.profile_fixed_batch_size is not None, "Memory profiling data processing expects profile_fixed_batch_size"
        bsz = self.args.profile_fixed_batch_size
        layernum_list_base = layernum_lists[0]
        layertype = len(layernum_list_base)
        layernum_lists = layernum_lists[1:]
        layernum_diff = self.args.profile_layernum_max - self.args.profile_layernum_min

        # Process each sequence length configuration
        sequence_length_list = self.get_seq_length_tuple_list()
        for seq in sequence_length_list:
            self._process_single_sequence_config(
                seq, world_size, layernum_list_base, layertype, layernum_lists, layernum_diff, bsz, config
            )

        # Write final results
        write_json_config(config, memory_config_path)

    def _process_single_sequence_config(
        self,
        seq: Tuple[int, ...],
        world_size: int,
        layernum_list_base: List[int],
        layertype: int,
        layernum_lists: List[List[int]],
        layernum_diff: int,
        bsz: int,
        config: Dict,
    ) -> None:
        """Process memory profiling data for a single sequence length configuration

        Args:
            seq: Tuple of sequence lengths for each layer type
            world_size: Total number of GPUs
            layernum_list_base: Base layer numbers for each layer type
            layertype: Number of layer types
            layernum_lists: Lists of layer numbers for different configurations
            layernum_diff: Difference between max and min layer numbers
            bsz: Batch size
            config: Configuration dictionary to store results

        This method processes:
        1. Parameter memory usage for different TP degrees
        2. Activation memory usage with and without checkpointing
        3. Memory overhead for different parallelism strategies
        4. Pipeline parallelism memory costs
        """
        seq_info = num2str(list(seq), "seq")
        print(f"Processing sequence length: {seq_info}")

        # Initialize result containers
        param_result_list = [dict() for _ in range(layertype)]
        act_result_list = [dict() for _ in range(layertype)]
        param_list = [-1] * layertype

        # Process tensor parallelism memory costs
        pp_deg, tp_deg = 1, 1
        while pp_deg * tp_deg <= world_size:
            strategy = f"{pp_deg}_{tp_deg}_{world_size//pp_deg//tp_deg}"
            if self.args.sequence_parallel:
                strategy += "_sp"

            if strategy in config:
                re = config[strategy]
                # Calculate memory costs for each layer type
                for l in range(layertype):
                    layernum_key_0 = layernum_list_base
                    layernum_key_1 = layernum_lists[l]

                    # Calculate parameter memory per layer
                    param_per_layer = (
                        (
                            re[self.key_format(layernum_key_1, bsz, seq_info, 0, "ms")]
                            - re[self.key_format(layernum_key_0, bsz, seq_info, 0, "ms")]
                        )
                        / layernum_diff
                        * pp_deg
                        / 4
                    )

                    # Calculate activation memory per sample
                    act_per_layer_per_sample = (
                        (
                            re[self.key_format(layernum_key_1, bsz, seq_info, 0, "act")]
                            - re[self.key_format(layernum_key_0, bsz, seq_info, 0, "act")]
                        )
                        / layernum_diff
                        * pp_deg
                        / (pp_deg * tp_deg)
                    )
                    act_per_layer_per_sample *= world_size / bsz

                    # Adjust for ZeRO-3
                    if self.args.profile_dp_type == "zero3":
                        param_per_layer *= world_size // pp_deg // tp_deg

                    # Update results
                    param_result_list[l][tp_deg] = param_per_layer
                    act_result_list[l][tp_deg] = act_per_layer_per_sample
                    param_list[l] = max(param_list[l], param_per_layer * tp_deg)

            tp_deg *= 2

        for l in range(layertype):
            print(f"layertype {l}:")
            print(f"param: {param_list[l]}")
            print(f"act_dict: {act_result_list[l]}")
        # Process checkpoint memory costs
        act_dict_c_list = [dict() for _ in range(layertype)]
        act_cpt_list = [-1] * layertype

        pp_deg, tp_deg = 1, 1
        while pp_deg * tp_deg <= world_size:
            strategy = f"{pp_deg}_{tp_deg}_{world_size//pp_deg//tp_deg}_c"
            if self.args.sequence_parallel:
                strategy += "_sp"

            if strategy in config:
                re = config[strategy]
                for l in range(layertype):
                    layernum_key_0 = layernum_list_base
                    layernum_key_1 = layernum_lists[l]

                    # Calculate activation memory with checkpointing
                    act_per_layer_per_sample = (
                        (
                            re[self.key_format(layernum_key_1, bsz, seq_info, 0, "act")]
                            - re[self.key_format(layernum_key_0, bsz, seq_info, 0, "act")]
                        )
                        / layernum_diff
                        * pp_deg
                        / (pp_deg * tp_deg)
                    )
                    act_per_layer_per_sample *= world_size / bsz

                    act_dict_c_list[l][tp_deg] = act_per_layer_per_sample
                    act_cpt_list[l] = max(act_cpt_list[l], act_per_layer_per_sample)

            tp_deg *= 2

        # Update activation results with checkpoint information
        for l in range(layertype):
            print(f"layertype {l}:")
            print(f"act_dict_c: {act_dict_c_list[l]}")
            print(f"act_cpt: {act_cpt_list[l]}")
            act_result_list[l]["checkpoint"] = act_cpt_list[l]

        # Process pipeline parallelism memory costs
        inf = 1e6
        other_memory_pp_off = {"model_states": defaultdict(lambda: inf), "activation": defaultdict(lambda: inf)}
        other_memory_pp_on_first = {"model_states": defaultdict(lambda: inf), "activation": defaultdict(lambda: inf)}
        other_memory_pp_on_last = {"model_states": defaultdict(lambda: inf), "activation": defaultdict(lambda: inf)}

        pp_deg = 1
        while pp_deg <= world_size:
            tp_deg = 1
            while pp_deg * tp_deg <= world_size:
                # Process different vocabulary parallelism configurations
                for enable_vocab_tp in [0, 1]:
                    if tp_deg == 1 and enable_vocab_tp == 1:
                        continue

                    strategy = f"{pp_deg}_{tp_deg}_{world_size//pp_deg//tp_deg}"
                    if enable_vocab_tp and tp_deg != 1:
                        strategy += "_vtp"
                    if self.args.sequence_parallel:
                        strategy += "_sp"

                    if strategy not in config:
                        continue

                    re = config[strategy]
                    # Calculate memory costs for current configuration
                    layernum = pp_deg if pp_deg > 1 else layernum_list_base[0]
                    layernum_list = [layernum] * layertype if pp_deg > 1 else layernum_list_base

                    # Calculate per-layer memory costs
                    ms_cost = [param_result_list[l][tp_deg] * 4 for l in range(layertype)]
                    act_cost = [act_result_list[l][tp_deg] for l in range(layertype)]

                    # Calculate total memory costs for first and last pipeline stages
                    layer_ms_costs_first = self.total_memcost(pp_deg, layernum, layertype, ms_cost, 0)
                    layer_ms_costs_last = self.total_memcost(pp_deg, layernum, layertype, ms_cost, pp_deg - 1)
                    layer_act_costs_first = self.total_memcost(pp_deg, layernum, layertype, act_cost, 0)
                    layer_act_costs_last = self.total_memcost(pp_deg, layernum, layertype, act_cost, pp_deg - 1)

                    # Calculate other memory costs
                    other_ms_first = re[self.key_format(layernum_list, bsz, seq_info, 0, "ms")] - layer_ms_costs_first
                    other_ms_last = (
                        re[self.key_format(layernum_list, bsz, seq_info, world_size - 1, "ms")] - layer_ms_costs_last
                    )

                    # Adjust for ZeRO-3
                    if self.args.profile_dp_type == "zero3":
                        other_ms_first = (
                            (
                                re[self.key_format(layernum_list, bsz, seq_info, 0, "ms")]
                                - layer_ms_costs_first / (world_size // pp_deg // tp_deg)
                            )
                            * (world_size // pp_deg)
                            / (tp_deg if enable_vocab_tp == 1 else 1)
                        )
                        other_ms_last = (
                            (
                                re[self.key_format(layernum_list, bsz, seq_info, world_size - 1, "ms")]
                                - layer_ms_costs_last / (world_size // pp_deg // tp_deg)
                            )
                            * (world_size // pp_deg)
                            / (tp_deg if enable_vocab_tp == 1 else 1)
                        )
                    # Calculate activation memory peaks
                    act_peak_first = max(
                        re[self.key_format(layernum_list, bsz, seq_info, 0, "act_peak")],
                        re[self.key_format(layernum_list, bsz, seq_info, 0, "act")],
                    )
                    act_peak_last = max(
                        re[self.key_format(layernum_list, bsz, seq_info, world_size - 1, "act_peak")],
                        re[self.key_format(layernum_list, bsz, seq_info, world_size - 1, "act")],
                    )

                    # Calculate other activation memory
                    other_act_first = (
                        act_peak_first - layer_act_costs_first * (bsz / (world_size // (pp_deg * tp_deg)))
                    ) / (bsz / world_size * pp_deg * (tp_deg if enable_vocab_tp else 1))
                    other_act_last = (
                        act_peak_last - layer_act_costs_last * (bsz / (world_size // (pp_deg * tp_deg)))
                    ) / (bsz / world_size * pp_deg * (tp_deg if enable_vocab_tp else 1))

                    # Ensure non-negative values
                    other_ms_first = max(other_ms_first, 0)
                    other_ms_last = max(other_ms_last, 0)
                    other_act_first = max(other_act_first, 0)
                    other_act_last = max(other_act_last, 0)

                    # Update memory dictionaries
                    tp_key = tp_deg if enable_vocab_tp else 1
                    if pp_deg == 1:
                        other_memory_pp_off["model_states"][tp_key] = min(
                            other_memory_pp_off["model_states"][tp_key], other_ms_first
                        )
                        other_memory_pp_off["activation"][tp_key] = min(
                            other_memory_pp_off["activation"][tp_key], other_act_first
                        )
                    else:
                        other_memory_pp_on_first["model_states"][tp_key] = min(
                            other_memory_pp_on_first["model_states"][tp_key], other_ms_first
                        )
                        other_memory_pp_on_first["activation"][tp_key] = min(
                            other_memory_pp_on_first["activation"][tp_key], other_act_first
                        )
                        other_memory_pp_on_last["model_states"][tp_key] = min(
                            other_memory_pp_on_last["model_states"][tp_key], other_ms_last
                        )
                        other_memory_pp_on_last["activation"][tp_key] = min(
                            other_memory_pp_on_last["activation"][tp_key], other_act_last
                        )

                tp_deg *= 2
            pp_deg *= 2

        # Handle sequence parallelism memory scaling
        if self.args.sequence_parallel:
            for tp in [2, 4, 8]:
                if tp not in act_result_list[0]:
                    act_result_list[0][tp] = act_result_list[0][tp // 2] / 2
                for memory_dict in [other_memory_pp_off, other_memory_pp_on_first, other_memory_pp_on_last]:
                    for key in ["model_states", "activation"]:
                        if tp not in memory_dict[key]:
                            memory_dict[key][tp] = memory_dict[key][tp // 2] / 2

        print("other_memory_pp_off:", other_memory_pp_off)
        print("other_memory_pp_on_first:", other_memory_pp_on_first)
        print("other_memory_pp_on_last:", other_memory_pp_on_last)
        # Store results in config
        config_key = "layertype_%d_sp" if self.args.sequence_parallel else "layertype_%d"
        for l in range(layertype):
            if config_key % l not in config:
                config[config_key % l] = dict()
            config[config_key % l][str(seq[l])] = {
                "parameter_size": param_list[l],
                "tp_activation_per_bsz_dict": act_result_list[l],
            }

        # Store other memory costs
        memory_keys = {
            "other_memory_pp_off": other_memory_pp_off,
            "other_memory_pp_on_first": other_memory_pp_on_first,
            "other_memory_pp_on_last": other_memory_pp_on_last,
        }

        suffix = "_sp" if self.args.sequence_parallel else ""
        for key, value in memory_keys.items():
            config_key = f"{key}{suffix}"
            if config_key not in config:
                config[config_key] = {}
            if seq_info.startswith("seq_"):
                seq_key = seq_info[4:]
            elif seq_info.startswith("seq"):
                seq_key = seq_info[3:]
            else:
                seq_key = seq_info
            config[config_key][seq_key] = copy.deepcopy(value)

    # =============== Util functions ===============
    def key_format(
        self,
        layernum: Union[List[int], int],
        bsz: Optional[int] = None,
        seq: Optional[Union[str, int]] = None,
        rank: Optional[int] = None,
        type: Optional[str] = None,
    ) -> str:
        """Format key for config dictionary

        Args:
            layernum: Layer number or list of layer numbers
            bsz: Batch size (optional)
            seq: Sequence length or sequence info string (optional)
            rank: GPU rank (optional)
            type: Memory type ('ms' for model states or 'act' for activations) (optional)

        Returns:
            str: Formatted key string

        Example:
            >>> key_format([1,2,3], 32, "seq128", 0, "ms")
            "layernum1_2_3_bsz32_seq128_rank0_ms"
        """
        if isinstance(layernum, list):
            s = "layernum" + "_".join(str(v) for v in layernum)
        else:
            s = f"layernum{layernum}"

        if bsz is not None:
            s += f"_bsz{bsz}"
        if seq is not None:
            if isinstance(seq, str):
                s += f"_{seq}"
            else:
                s += f"_seq{seq}"
        if rank is not None and type is not None:
            s += f"_rank{rank}_{type}"
        return s

    def total_memcost(
        self, pp_deg: int, layernum: int, layertype: int, per_layer_cost: List[float], stage_idx: int
    ) -> float:
        """Calculate total memory cost for a pipeline stage

        Args:
            pp_deg: Pipeline parallelism degree
            layernum: Number of layers per type
            layertype: Number of layer types
            per_layer_cost: Memory cost per layer for each layer type
            stage_idx: Pipeline stage index

        Returns:
            float: Total memory cost for the specified pipeline stage

        Note:
            Assumes equal distribution of layers across pipeline stages
        """
        # Calculate memory cost for each layer
        layer_costs = []
        for l in range(layertype):
            layer_costs.extend([per_layer_cost[l]] * layernum)

        # Calculate layer distribution across pipeline stages
        total_layer_num = layertype * layernum
        avg_layer_num = int(total_layer_num // pp_deg)
        last_layer_num = total_layer_num - avg_layer_num * (pp_deg - 1)
        pp_divide = [avg_layer_num] * (pp_deg - 1) + [last_layer_num]

        # Verify equal distribution
        assert avg_layer_num == last_layer_num

        # Sum memory costs for the specified stage
        start_idx = int(np.sum(pp_divide[:stage_idx]))
        end_idx = int(np.sum(pp_divide[: stage_idx + 1]))
        return np.sum(layer_costs[start_idx:end_idx])

    def argval2str(self, val: Union[List, Any]) -> str:
        """Convert argument value to string format

        Args:
            val: Value to convert, can be a list or single value

        Returns:
            str: Space-separated string for lists, or string representation for single values
        """
        if isinstance(val, list):
            return " ".join(str(i) for i in val).strip()
        return str(val)

    def arg2str(self, key: str, val: Union[List, Any]) -> str:
        """Format single argument as command line parameter

        Args:
            key: Argument name
            val: Argument value

        Returns:
            str: Formatted argument string (e.g., '--key value')
        """
        return f" --{key} {self.argval2str(val)}"

    def args2str(self, args: Union[Dict, List[Tuple]], exclude_args: List[str] = []) -> str:
        """Convert multiple arguments to command line format

        Args:
            args: Dictionary of arguments or list of (key, value) tuples
            exclude_args: List of argument names to exclude

        Returns:
            str: Space-separated string of formatted arguments
        """
        s = ""
        if isinstance(args, dict):
            for key, val in args.items():
                if key not in exclude_args:
                    s += self.arg2str(key, val)
        elif isinstance(args, (list, tuple)) and len(args) > 0 and len(args[0]) == 2:
            for key, val in args:
                if key not in exclude_args:
                    s += self.arg2str(key, val)
        return s

    def env_args(self) -> Dict[str, Union[str, int]]:
        """Get environment configuration arguments

        Returns:
            Dict: Dictionary of environment variables with defaults:
                - PROFILE_LAUNCHER: Launcher command
                - PROFILE_TRAINER: Trainer script path
                - NUM_NODES: Number of nodes
                - NUM_GPUS_PER_NODE: GPUs per node
                - MASTER_ADDR/PORT: Communication settings
                - NCCL settings
        """
        return {
            "PROFILE_LAUNCHER": os.getenv("PROFILE_LAUNCHER", "torchrun"),
            "PROFILE_TRAINER": os.getenv("PROFILE_TRAINER", "train_dist.py"),
            "NUM_NODES": os.getenv("NUM_NODES", "1") if self.args.profile_type == "memory" else "1",
            "NUM_GPUS_PER_NODE": os.getenv("NUM_GPUS_PER_NODE", "8") if self.args.profile_type == "memory" else "1",
            "MASTER_ADDR": os.getenv("MASTER_ADDR", ""),
            "MASTER_PORT": os.getenv("MASTER_PORT", ""),
            "NCCL_SOCKET_IFNAME": os.getenv("NCCL_SOCKET_IFNAME", ""),
            "NODE_RANK": os.getenv("NODE_RANK", "0"),
        }

    def launch_scripts(self, env_args: Dict[str, str]) -> str:
        """Generate launch script command

        Args:
            env_args: Dictionary of environment arguments

        Returns:
            str: Formatted launch command string

        Note:
            Currently uses simplified launch command without node configuration
        """
        return f"{env_args['PROFILE_LAUNCHER']} {env_args['PROFILE_TRAINER']}"
