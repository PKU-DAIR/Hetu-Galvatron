from galvatron.utils import read_json_config, num2str
from scipy.optimize import curve_fit
import os
from galvatron.utils import (
    read_allreduce_bandwidth_config, 
    read_json_config, 
    read_p2p_bandwidth_config, 
    remap_config,
    num2str
)
import logging
from .cost_model_args_optimize import TrainArgsOptimize, ProfileModelArgsOptimize, ProfileHardwareArgsOptimize, UtilsArgsOptimize, VersionOptionArgsOptimize, EstimateTPTimeType
from .time_costmodel_optimize import TimeCostModelOptimize, OtherTimeCostModelOptimize
from .memory_costmodel_optimize import MemoryCostModelOptimize, OtherMemoryCostModelOptimize

class GalvatronCostModelHandler:
    def __init__(self, args):
        self.args = args
        args.gpu_num = args.num_nodes * args.num_gpus_per_node
        self.layernum_arg_names = None
        self.mem_path = None
        self.time_path = None
        self.model_name = None
        self.time_config = None
        self.memory_config = None
        self.param_sizes = None
        self.act_sizes = None
        self.other_memory_pp_off = None
        self.other_memory_pp_on = None
        self.time_profiled_list = None
        self.model_type = 'gpt'
        self.memory_constraint = args.memory_constraint * 1024
        self.logger = None

    def set_cost_model_handler_info(self, path,  model_layer_configs, model_name):
        self.set_model_layer_configs(model_layer_configs)
        self.set_path(path)
        self.set_model_name(model_name)
        self.memory_profiling_path()
        self.time_profiling_path()
    
    def set_path(self, path):
        self.path = path

    def set_model_type(self, model_type):
        self.model_type = model_type

    def set_model_name(self, name):
        self.model_name = name
        
    def memory_profiling_path(self):
        if self.mem_path is not None:
            return self.mem_path
        assert self.model_name is not None, 'Should specify the model name!'
        args = self.args
        memory_config_name = 'memory_profiling_%s_%s_all.json'%(args.mixed_precision, self.model_name) # TODO Add support for separate mode.
        if args.memory_profiling_path is None:
            memory_config_path = os.path.join(self.path, 'configs')
        else:
            memory_config_path = args.memory_profiling_path
        self.mem_path = os.path.join(memory_config_path, memory_config_name)
        return self.mem_path
    
    def time_profiling_path(self):
        if self.time_path is not None:
            return self.time_path
        assert self.model_name is not None, 'Should specify the model name!'
        args = self.args
        time_config_name = "computation_profiling_%s_%s_all.json"%(args.mixed_precision, self.model_name) # TODO Add support for separate mode.
        if args.time_profiling_path is None:
            self.time_path = os.path.join(self.path, "configs")
        else:
            self.time_path = args.time_profiling_path

        self.time_path = os.path.join(self.time_path, time_config_name)
        return self.time_path
    
    def set_model_layer_configs(self, model_layer_configs):
        if model_layer_configs is None:
            return
        self.hiddensize_list = [config['hidden_size'] for config in model_layer_configs]
        self.layernum_list = [config['layer_num'] for config in model_layer_configs]
        self.seqlen_list = [config['seq_len'] for config in model_layer_configs]
        self.num_layertype = len(self.layernum_list)
        
    def initialize_cost_model_handler(self):
        self.handler_init_logger()
        self.get_profiled_model_configs()
        self.get_profiled_hardware_configs()
        self.set_cost_model_args()
        self.show_handler_info()
    
    def get_profiled_model_configs(self): # TODO Add support for separate mode.
        args = self.args
        
        # [Step1] Profile Model Computation Configs
        self.time_config = read_json_config(self.time_profiling_path())
        if args.profile_granularity == 'together':
            if args.time_profile_mode == 'static':
                self.time_profiled_list = []
                self.other_time_profiled_list = []
                for i in range(self.num_layertype):
                    for key, value in self.time_config.items():
                        if key.startswith(f'layertype_{i}_'):
                            self.time_profiled_list.append(value)
                        if key.startswith('layertype_other_'):
                            self.other_time_profiled_list.append(value)
                            
            elif args.time_profile_mode == 'batch':
                self.time_profiled_list = []
                self.other_time_profiled_list = []
                for i in range(self.num_layertype):
                    x_data, y_data = [], []
                    for key, value in self.time_config.items():
                        if key.startswith(f'layertype_{i}_') and f'seq{self.seqlen_list[i]}' in key:
                            x_data.append(int(key.split('_')[-2][3:]))
                            y_data.append(value * x_data[-1])
                    assert len(x_data) >= 8, f"Different batch size in computation profile of layertype_{i} should not be lower than 8."
                    def linear_func(x, m, c):
                        return m * x + c
                    popt, _ = curve_fit(linear_func, x_data, y_data)
                    print("Fitted parameters:", popt)
                    self.time_profiled_list.append(popt)
                
                for i in range(self.num_layertype):
                    x_data, y_data = [], []
                    for key, value in self.time_config.items():
                        if key.startswith('layertype_other_') and f'seq{self.seqlen_list[i]}' in key:
                            x_data.append(int(key.split('_')[-2][3:]))
                            y_data.append(value * x_data[-1])
                    assert len(x_data) >= 8, f"Different batch size in computation profile of layertype_other should not be lower than 8."
                    def linear_func(x, m, c):
                        return m * x + c
                    popt, _ = curve_fit(linear_func, x_data, y_data)
                    print("Fitted parameters:", popt)
                    self.other_time_profiled_list.append(popt)
            
            elif args.time_profile_mode == 'sequence':
                self.time_profiled_list = []
                self.other_time_profiled_list = []
                for i in range(self.num_layertype):
                    x_data, y_data = [], []
                    for key, value in self.time_config.items():
                        if key.startswith(f'layertype_{i}_') and f'_bsz1_' in key:
                            x_data.append(int(key.split('seq')[-1]))
                            y_data.append(value)
                    def quadratic_func(x, a, b, c):
                        return a * x * x + b * x + c
                    popt, _ = curve_fit(quadratic_func, x_data, y_data)
                    print("Fitted parameters:", popt)
                    self.time_profiled_list.append(quadratic_func(self.seqlen_list[i], *popt))
                
                for i in range(self.num_layertype):
                    x_data, y_data = [], []
                    for key, value in self.time_config.items():
                        if key.startswith(f'layertype_{i}_') and f'_bsz1_' in key:
                            x_data.append(int(key.split('_')[-3][3:]))
                            y_data.append(value)
                    def linear_func(x, m, c):
                        return m * x + c
                    popt, _ = curve_fit(linear_func, x_data, y_data)
                    print("Fitted parameters:", popt)
                    self.other_time_profiled_list.append(linear_func(self.seqlen_list[i], *popt))
            else:
                raise ValueError("Unsupported time profile mode: %s"%(args.time_profile_mode))
        elif args.profile_granularity == 'split': # TODO add code for this mode
            pass
        else:
            raise ValueError("Unsupported profile granularity: %s"%(args.profile_granularity))
        
        # [Step2] Profile Model Memory Configs
        self.memory_config = read_json_config(self.memory_profiling_path())
        self.memory_config = self.convert_keys_to_int(self.memory_config)
        if args.profile_granularity == 'together':
            self.param_memory_list = [0 for _ in range(self.num_layertype)]
            self.act_memory_list = [{} for _ in range(self.num_layertype)] 
            if args.memory_profile_mode == 'static':
                if args.sequence_parallel:
                    for i in range(self.num_layertype):
                        layer_mem_config = self.memory_config[f'layertype_{i}_sp']
                        self.param_memory_list[i] = layer_mem_config[self.seqlen_list[i]]['parameter_size']
                        self.act_memory_list[i] = layer_mem_config[self.seqlen_list[i]]['tp_activation_per_bsz_dict'].copy()
                    seq_info = num2str(self.seqlen_list, 'seq')[3:]
                    if seq_info.isdigit():
                        seq_info = int(seq_info)
                    self.other_memory_pp_off = self.memory_config['other_memory_pp_off_sp'][int(seq_info)]
                    self.other_memory_pp_on = {'first_stage':self.memory_config['other_memory_pp_on_first_sp'][seq_info], 'last_stage':self.memory_config['other_memory_pp_on_last_sp'][seq_info]}
                else:
                    for i in range(self.num_layertype):
                        layer_mem_config = self.memory_config[f'layertype_{i}']
                        self.param_memory_list[i] = layer_mem_config[self.seqlen_list[i]]['parameter_size']
                        self.act_memory_list[i] = layer_mem_config[self.seqlen_list[i]]['tp_activation_per_bsz_dict'].copy()
                    seq_info = num2str(self.seqlen_list, 'seq')[3:]
                    if seq_info.isdigit():
                        seq_info = int(seq_info)
                    self.other_memory_pp_off = self.memory_config['other_memory_pp_off'][int(seq_info)]
                    self.other_memory_pp_on = {'first_stage':self.memory_config['other_memory_pp_on_first'][seq_info], 'last_stage':self.memory_config['other_memory_pp_on_last'][seq_info]}
            elif args.memory_profile_mode == 'sequence':
                pass
            
        elif args.profile_granularity == 'split': # TODO Add support for separate mode.
            pass    
        else:
            raise ValueError("Unsupported profile granularity: %s"%(args.profile_granularity))
            
    def get_profiled_hardware_configs(self):
        args = self.args
        if args.allreduce_bandwidth_config_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            allreduce_bandwidth_config_path = os.path.join(self.path, hardware_configs_dir)
        else:
            allreduce_bandwidth_config_path = args.allreduce_bandwidth_config_path
        allreduce_bandwidth_config_name = 'allreduce_bandwidth_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        args.allreduce_bandwidth_config_path  = os.path.join(allreduce_bandwidth_config_path, allreduce_bandwidth_config_name)
        self.allreduce_bandwidth, self.allreduce_comm_coe = read_allreduce_bandwidth_config(args.allreduce_bandwidth_config_path, gpu_num=args.gpu_num)
        
        if args.p2p_bandwidth_config_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            p2p_bandwidth_config_path = os.path.join(self.path, hardware_configs_dir)
        else:
            p2p_bandwidth_config_path = args.p2p_bandwidth_config_path
        p2p_bandwidth_config_name = 'p2p_bandwidth_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        args.p2p_bandwidth_config_path  = os.path.join(p2p_bandwidth_config_path, p2p_bandwidth_config_name)
        self.p2p_bandwidth, self.p2p_comm_coe = read_p2p_bandwidth_config(args.p2p_bandwidth_config_path)
        
        if args.overlap_coe_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            overlap_coe_path = os.path.join(self.path, hardware_configs_dir)
        else:
            overlap_coe_path = args.overlap_coe_path
        overlap_coe_name = 'overlap_coefficient.json'
        args.overlap_coe_path = os.path.join(overlap_coe_path, overlap_coe_name)
        self.overlap_coe = read_json_config(args.overlap_coe_path)['overlap_coe']
        
        if args.sp_time_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            sp_time_path = os.path.join(self.path, hardware_configs_dir)
        else:
            sp_time_path = args.sp_time_path
        sp_time_config_name = 'sp_time_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        args.sp_time_path = os.path.join(sp_time_path, sp_time_config_name)
        sp_config = read_json_config(args.sp_time_path)
        self.sp_allreduce = remap_config(sp_config, "allreduce")
        self.sp_all2all = remap_config(sp_config, "all2all")

        return self.allreduce_bandwidth, self.p2p_bandwidth, self.overlap_coe, self.sp_allreduce, self.sp_all2all

    def set_cost_model_args(self):
        self.train_args_list, self.profile_model_args_list, self.profile_hardware_args_list, self.utils_args_list, self.version_option_args_list = [], [], [], [], []
        for i in range(self.num_layertype):
            train_args = TrainArgsOptimize(
                seq_length=self.seqlen_list[i],
                hidden_size=self.hiddensize_list[i],
                mixed_precision=False if self.args.mixed_precision == 'fp32' else True,
                use_zero2_for_dp=True if self.args.default_dp_type == 'zero2' else False,
                async_grad_reduce=self.args.async_grad_reduce,
                disable_vtp=self.args.disable_vtp,
                sequence_parallel=self.args.sequence_parallel,
                pipeline_type=self.args.pipeline_type
            )
            self.train_args_list.append(train_args)
            
            profile_model_args = ProfileModelArgsOptimize(
                forward_computation_time=self.time_profiled_list[i],
                other_time_profiled=self.other_time_profiled_list[0], # actually the same for all layertypes
                parameter_memory=self.param_memory_list[i],
                tp_activation_per_bsz_dict=self.act_memory_list[i],
                other_memory_pp_off=self.other_memory_pp_off,
                other_memory_pp_on=self.other_memory_pp_on
            )
            self.profile_model_args_list.append(profile_model_args)
            
            profile_hardware_args = ProfileHardwareArgsOptimize(
                bct_fct_coe=2,
                overlap_slowdown_coe=self.overlap_coe,
                comm_coe_dict=self.allreduce_comm_coe,
                p2p_comm_coe_dict=self.p2p_comm_coe,
                allreduce_dict=self.sp_allreduce,
                all2all_dict=self.sp_all2all
            )
            self.profile_hardware_args_list.append(profile_hardware_args)
            
            utils_args = UtilsArgsOptimize()
            self.utils_args_list.append(utils_args)
            
            version_option_args = VersionOptionArgsOptimize(
                estimate_tp_time_type=EstimateTPTimeType.FIXED if self.args.estimate_tp_time_type == 'fixed' else EstimateTPTimeType.FIT,
                zero_with_slight_noise=True if self.args.zero_with_slight_noise else False
            )
            self.version_option_args_list.append(version_option_args)
    
    def show_handler_info(self):
        print('================================================================================')
        print('--- Optimization Configs ----')
        print('Memory constraint: %d GB'%self.args.memory_constraint)
        print('Pipeline Type:', self.args.pipeline_type)
        print('Default DP Type:', self.args.default_dp_type)
        print('Mixed Precision:', self.args.mixed_precision)
        print('================================================================================')
        print('---- Environment Configs ----')
        print('Allreduce Bandwidth (GB/s):', self.allreduce_bandwidth)
        print('Allreduce Communication Coefficient (ms/MB):', self.allreduce_comm_coe)
        print('P2P Bandwidth (GB/s):', self.p2p_bandwidth)
        print('P2P Communication Coefficient (ms/MB):', self.p2p_comm_coe)
        print('Overlap coefficient:', self.overlap_coe)
        print('================================================================================')
        print('------- Model Configs -------')
        print('Model Name:', self.model_name)
        print('Num layertype:', self.num_layertype)
        print('Layer_num:', self.layernum_list)
        print('Hidden_size:', self.hiddensize_list)
        print('Seq_len:', self.seqlen_list)
        print('================================================================================')
        print('--- Model Computation Configs ---')
        print('Forward computation time:', self.time_profiled_list)
        print('================================================================================')
        print('--- Model Memory Configs ---')
        print('Parameter Memory Cost:', self.param_memory_list)
        print('Activation Memory Cost of Different TP degree (per bsz):')
        print(self.act_memory_list)
        print('Other Memory Cost (pp = 1):')
        print(self.other_memory_pp_off)
        print('Other Memory Cost (pp > 1):')
        print(self.other_memory_pp_on)
        print('================================================================================')
        print('Train Args List:')
        print(self.train_args_list)
        print('================================================================================')
        print('Profile Model Args List:')
        print(self.profile_model_args_list)
        print('================================================================================')
        print('Profile Hardware Args List:')
        print(self.profile_hardware_args_list)
        print('================================================================================')
        print('Utils Args List:')
        print(self.utils_args_list)
        print('================================================================================')
        print('Version Option Args List:')
        print(self.version_option_args_list)
        print('================================================================================')
    
    # ============== Cost Model Functions ===============
    def get_time_cost_for_specific_strategy(self, strategy, global_batch_size, chunks):
        # [step1] get basic info
        pp_size, tp_size, dp_size = strategy[0], strategy[1], strategy[2]
        use_ulysses = True if 'sp' in strategy[-1].keys() and strategy[-1]['sp'] else False
        fsdp = True if 'fsdp' in strategy[-1].keys() and strategy[-1]['fsdp'] else False
        world_size = pp_size * tp_size * dp_size
        mbsz = global_batch_size // chunks // dp_size
        
        # [step2] get time cost for each layertype
        timecosts_dict, timecosts_no_sync_gradient_dict = {}, {}
        for layer_type_id in range(self.num_layertype):
            time_cost = TimeCostModelOptimize(strategy=strategy, global_batch_size=global_batch_size, chunks=chunks,
                                            train_args=self.train_args_list[layer_type_id],
                                            profile_model_args=self.profile_model_args_list[layer_type_id],
                                            profile_hardware_args=self.profile_hardware_args_list[layer_type_id],
                                            utils_args=self.utils_args_list[layer_type_id],
                                            version_option_args=self.version_option_args_list[layer_type_id],
                                            logger=self.logger).gen_result()
            time_cost_no_sync_gradient = TimeCostModelOptimize(strategy=strategy, global_batch_size=global_batch_size, chunks=chunks,
                                            no_sync_gradients=True,
                                            train_args=self.train_args_list[layer_type_id],
                                            profile_model_args=self.profile_model_args_list[layer_type_id],
                                            profile_hardware_args=self.profile_hardware_args_list[layer_type_id],
                                            utils_args=self.utils_args_list[layer_type_id],
                                            version_option_args=self.version_option_args_list[layer_type_id],
                                            logger=self.logger).gen_result()
            timecosts_dict[layer_type_id] = time_cost
            timecosts_no_sync_gradient_dict[layer_type_id] = time_cost_no_sync_gradient
        self.handler_log(f"Time costs for each layertype: {timecosts_dict}")
        self.handler_log(f"Time costs for each layertype (no sync gradient): {timecosts_no_sync_gradient_dict}")
        
        # [step3] get other time cost
        other_time_cost, other_time_cost_no_sync_gradient = OtherTimeCostModelOptimize(
            mbsz=mbsz,
            pp_deg=pp_size,
            world_size=world_size,
            vsp=use_ulysses,
            embed_sdp=fsdp,
            min_tp=tp_size,
            max_tp=tp_size,
            sequence_length_list=self.seqlen_list,
            train_args=self.train_args_list[0], # actually the same for all layertypes
            profile_model_args=self.profile_model_args_list[0], # actually the same for all layertypes
            profile_hardware_args=self.profile_hardware_args_list[0], # actually the same for all layertypes
            version_option_args=self.version_option_args_list[0], # actually the same for all layertypes
            logger=self.logger,
        ).gen_result()
        self.handler_log(f"Other time cost: {other_time_cost}")
        self.handler_log(f"Other time cost (no sync gradient): {other_time_cost_no_sync_gradient}")
        
        # [step4] compose total time cost
        if pp_size == 1:
            time_cost_per_chunk, time_cost_no_sync_gradient_per_chunk = other_time_cost[tp_size][0], other_time_cost_no_sync_gradient[tp_size][0]
            for i in range(self.num_layertype):
                time_cost_per_chunk += timecosts_dict[i] * self.layernum_list[i]
                time_cost_no_sync_gradient_per_chunk += timecosts_no_sync_gradient_dict[i] * self.layernum_list[i]
            self.handler_log(f"Time cost per chunk: {time_cost_per_chunk}")
            self.handler_log(f"Time cost per chunk (no sync gradient): {time_cost_no_sync_gradient_per_chunk}")
            # total time cost
            result = time_cost_no_sync_gradient_per_chunk * (chunks - 1) + time_cost_per_chunk
            return result
        
        # TODO Add modeling for pp_size > 1.
        raise NotImplementedError("Currently only support pp_size = 1")
        
    def get_memory_cost_for_specific_strategy(self, strategy, global_batch_size, chunks):
        # [step1] get basic info
        pp_size, tp_size, dp_size = strategy[0], strategy[1], strategy[2]
        use_ulysses = True if 'sp' in strategy[-1].keys() and strategy[-1]['sp'] else False
        fsdp = True if 'fsdp' in strategy[-1].keys() and strategy[-1]['fsdp'] else False
        world_size = pp_size * tp_size * dp_size
        
        # [step2] get memory cost for each layertype
        memorycosts_dict = {}
        for layer_type_id in range(self.num_layertype):
            all_stage_cost = {}
            for stage_idx in range(pp_size):
                memory_cost = MemoryCostModelOptimize(strategy=strategy, global_batch_size=global_batch_size, chunks=chunks, stage_idx=stage_idx,
                                                    train_args=self.train_args_list[layer_type_id],
                                                    profile_model_args=self.profile_model_args_list[layer_type_id],
                                                    version_option_args=self.version_option_args_list[layer_type_id],
                                                    logger=self.logger).get_memory_cost()
                all_stage_cost[stage_idx] = memory_cost
            memorycosts_dict[layer_type_id] = all_stage_cost
        self.handler_log(f"Memory costs for each layertype: {memorycosts_dict}")
        
        # [step3] get other memory cost
        other_memory_cost = OtherMemoryCostModelOptimize(
            pp_deg=pp_size,
            global_batch_size=global_batch_size,
            chunks=chunks,
            world_size=world_size,
            vsp=use_ulysses,
            embed_sdp=fsdp,
            min_tp=tp_size,
            max_tp=tp_size,
            train_args=self.train_args_list[0], # actually the same for all layertypes
            profile_model_args=self.profile_model_args_list[0], # actually the same for all layertypes
            utils_args=self.utils_args_list[0], # actually the same for all layertypes
            version_option_args=self.version_option_args_list[0], # actually the same for all layertypes
            logger=self.logger,
        ).get_other_memory_cost()
        self.handler_log(f"Other memory cost: {other_memory_cost}")
        
        # [step4] compose total memory cost
        result = dict()
        for stage_idx in range(pp_size):
            result[f'stage{stage_idx}'] = dict()
        if pp_size == 1:
            # calculate model_states_memory
            result['stage0']['model_states_memory'] = other_memory_cost[tp_size][0]['other_ms_memory']
            for i in range(self.num_layertype):
                result['stage0']['model_states_memory'] += memorycosts_dict[i][0]['model_states_memory'] * self.layernum_list[i]
            
            # calculate activation_memory
            result['stage0']['activation_memory'] = other_memory_cost[tp_size][0]['other_activation_memory']
            for i in range(self.num_layertype):
                result['stage0']['activation_memory'] += memorycosts_dict[i][0]['activation_memory'] * self.layernum_list[i]
            
            # calculate total memory
            result['stage0']['total_memory'] = other_memory_cost[tp_size][0]['other_total_memory']
            for i in range(self.num_layertype):
                result['stage0']['total_memory'] += memorycosts_dict[i][0]['total_memory'] * self.layernum_list[i]
        else:
            # TODO Add modeling for pp_size > 1.
            raise NotImplementedError("Currently only support pp_size = 1")
    
        return result

    # =============== Utils Functions ===============
    def convert_keys_to_int(self, d):
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                if isinstance(k, str) and k.isdigit():
                    new_dict[int(k)] = self.convert_keys_to_int(v)
                else:
                    new_dict[k] = self.convert_keys_to_int(v)
            return new_dict
        return d
    
        
    def handler_init_logger(self):
        """Initialize a compact, colored console logger without timestamps.

        Uses ANSI color codes for levels and avoids adding duplicate handlers
        if the logger already has a stream handler.
        """
        logger = logging.getLogger('CostModelHandler')
        logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers in case this is called repeatedly
        has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        if has_stream_handler:
            # Make sure messages don't propagate to root handlers (avoid duplicate prints)
            logger.propagate = False
            self.logger = logger
            return

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Compact format: [LEVEL] message
        # No timestamp, no logger name
        class _LevelColorFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\u001b[36m',    # cyan
                'INFO': '\u001b[32m',     # green
                'WARNING': '\u001b[33m',  # yellow
                'ERROR': '\u001b[31m',    # red
                'CRITICAL': '\u001b[35m', # magenta
            }
            RESET = '\u001b[0m'

            def format(self, record):
                levelname = record.levelname
                color = self.COLORS.get(levelname, '')
                msg = super().format(record)
                return f"{color}[{levelname}] [CostModelHandler]{self.RESET} {msg}"

        formatter = _LevelColorFormatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # Prevent propagation to root logger to avoid duplicate lines
        logger.propagate = False
        self.logger = logger
    
    def handler_log(self, message):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(f'[CostModelHandler] {message}', flush=True)
        