import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Optional, Callable, Union

from .cost_model_args import ModelArgs, TrainArgs, ParallelArgs, ProfileModelArgs, ProfileHardwareArgs


class MemoryCostModel:
    memory_args_list = {
        'ModelArgs':['parameter_size'], 
        'TrainArgs':['mixed_precision', 'async_grad_reduce', 'pytorch_context_mem'], 
        'ParallelArgs':['use_zero2_for_dp', 'max_tp_deg', 'disable_vtp', 'sequence_parallel', 'sp_space', 'pipeline_type', 'optimal_chunk_func', 'chunks'], 
        'ProfileModelArgs':['tp_activation_per_bsz_dict', 'other_memory_pp_off', 'other_memory_pp_on']
    }
    
    def __init__(self, 
                stategy, 
                global_batch_size:int = 8, 
                mbsz: int = -1, 
                min_tp: int = -1, 
                max_tp: int = -1,
                stage_idx: int = 0,
                vsp: int = 0, 
                embed_sdp: bool = False,
                model_args: ModelArgs = None,
                train_args: TrainArgs = None,
                parallel_args: ParallelArgs = None,
                profile_model_args: ProfileModelArgs = None):
        
        self.__post_init__(stategy, global_batch_size, mbsz, min_tp, max_tp, stage_idx, vsp, embed_sdp, model_args, train_args, parallel_args, profile_model_args)
        self.initialize()
        self.estimate_parameter_size()
        self.estimate_model_states_size()
        self.estimate_activation_size()
        self.estimate_other_memory_cost()
        
    
    def __post_init__(self, stategy, global_batch_size: int = 8, mbsz: int = -1, min_tp: int = -1, max_tp: int = -1, stage_idx: int = 0, vsp: int = 0, embed_sdp: bool = False,
                        model_args: ModelArgs = None, train_args:TrainArgs = None, parallel_args: ParallelArgs = None, profile_model_args: ProfileModelArgs = None):
        # validate arguments
        assert mbsz > -1, f'Invalid mbsz: {mbsz}'
        assert min_tp > -1, f'Invalid min_tp: {min_tp}'
        assert all(x is not None for x in (model_args, train_args, parallel_args, profile_model_args)), "One or more variables are None"

        # Aggregate all arguments
        self.args = SimpleNamespace()
        self.args.strategy = stategy
        self.args.global_batch_size = global_batch_size
        self.args.mbsz = mbsz
        self.args.min_tp = min_tp
        self.args.max_tp = max_tp
        self.args.stage_idx = stage_idx
        self.args.vsp = vsp
        self.args.embed_sdp = embed_sdp
        components = {'ProfileModelArgs': profile_model_args, 'ModelArgs': model_args, 'TrainArgs': train_args, 'ParallelArgs': parallel_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in self.memory_args_list[class_name]:
                    setattr(self.args, key, value)

    def initialize(self):
        args = self.args
        
        # [initialize]:initialize strategy
        self.pp_size = args.strategy[0]
        self.tp_size = args.strategy[1]
        self.dp_size = args.strategy[2]
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.sdp_size = self.tp_size * self.dp_size
        else:
            self.sdp_size = self.dp_size
    
        # [adjust]:calculate chunks
        if args.chunks is None:
            args.chunks = args.optimal_chunk_func(args.global_batch_size // self.dp_size, args.strategy, args.mbsz, args.min_tp)
        max_chunks = args.global_batch_size // (self.tp_size * self.dp_size // args.min_tp)
        max_chunks = 1 if max_chunks == 0 else max_chunks
        self.chunks = max_chunks if args.chunks > max_chunks else args.chunks
        self.chunks = int(self.chunks)
        
        # [initialize]:initialize local batch size and pp stage act_1f1b_ratio
        self.bsz = args.global_batch_size / self.dp_size
        if (args.pipeline_type == 'pipedream_flush' and self.pp_size > 1) or self.pp_size == 1:
            microbatches = [t.shape[0] for t in chunk_like_torch(int(args.global_batch_size / self.dp_size / (self.tp_size // args.min_tp)), self.chunks)]
            assert self.chunks == len(microbatches)
            end = self.pp_size - args.stage_idx if self.pp_size - args.stage_idx <= self.chunks else self.chunks
            self.act_1f1b_ratio = np.sum(microbatches[:end]) / np.sum(microbatches)
            self.act_1f1b_ratio_first = np.sum(microbatches[:min(self.pp_size, self.chunks)]) / np.sum(microbatches)
            self.act_1f1b_ratio_last = microbatches[0] / np.sum(microbatches)
            self.bsz = self.act_1f1b_ratio * self.bsz
        else:
            microbatches = [t.shape[0] for t in chunk_like_torch(int(args.global_batch_size / self.dp_size / (self.tp_size // args.min_tp)), self.chunks)]
            self.bsz = microbatches[0]

        # [initialize]:initialize zero2 and zero3 ratio
        if self.chunks == 1:
            self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            self.zero3_ratio = lambda d: (1/d + 0.003)
        else:
            if args.async_grad_reduce:
                self.zero2_ratio = (lambda d: (6/8 * (1/d + 0.003) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d + 0.003) + 2/4))
                self.zero3_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            else:
                self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8) * 5/4) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                self.zero3_ratio = lambda d: (1/d + 0.003) * 5/4
                # *5/4: for fp32 grad 
    
    def estimate_parameter_size(self):
        args = self.args
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.parameter_size = args.parameter_size
        else:
            self.parameter_size = args.parameter_size / self.tp_size
        
    def estimate_model_states_size(self):
        args = self.args
        self.model_states_size = 4 * self.parameter_size
        if 'fsdp' in args.strategy[-1].keys() and args.strategy[-1]['fsdp']:
            # fsdp_model_states memory is slightly larger than dp_model_states/dp_size
            # we add a small bias to ensure the predicted fsdp memory NOT smaller than real value
            # Actually, this bias barely affect search result.
            self.model_states_size *= self.zero3_ratio(self.sdp_size)
        elif 'fsdp' in args.strategy[-1].keys() and args.strategy[-1]['fsdp'] == 0 and args.use_zero2_for_dp:
            self.model_states_size *= self.zero2_ratio(self.sdp_size)
        
    def estimate_activation_size(self):
        args = self.args
        if 'cpt' in args.strategy[-1].keys() and args.strategy[-1]['cpt']:
            assert(args.tp_activation_per_bsz_dict['checkpoint'] is not None)
            self.activation_size = args.tp_activation_per_bsz_dict['checkpoint'] * self.bsz
            if args.sequence_parallel:
                self.activation_size /= self.tp_size
        else:
            self.activation_size = args.tp_activation_per_bsz_dict[self.tp_size] * self.bsz
    
    def estimate_other_memory_cost(self):
        args = self.args
        
        # [initialize]:initialize total_min_tp
        if args.disable_vtp:
            total_min_tp = [1]
        else:
            total_min_tp, i = [], args.min_tp
            gpu_num = args.strategy[0] * args.strategy[1] * args.strategy[2]
            while i * self.pp_size <= gpu_num and i <= args.max_tp:
                total_min_tp.append(i)
                i *= 2
                
        # [validate]: add validation for total_min_tp
        total_min_tp = [tp for tp in total_min_tp
            if tp in args.other_memory_pp_off['model_states'].keys() and 
               tp in args.other_memory_pp_on['first_stage']['model_states'] and 
               tp in args.other_memory_pp_on['last_stage']['model_states']]
        
        # [calculate]:calculate other memory costs
        self.other_memory_cost = dict()
        for tp in total_min_tp:
            tp_other_memory_cost = [0] * self.pp_size
            other_layers_bsz = args.global_batch_size * tp / self.tp_size / self.dp_size
            other_layers_bsz /= self.chunks
            
            # Determine the memory ratio for Zero optimization
            if args.vsp:
                model_tp = 1
                other_ms_zero2_ratio = self.zero3_ratio(self.tp_size * self.dp_size) if args.embed_sdp else (self.zero2_ratio(self.tp_size * self.dp_size) if args.use_zero2_for_dp else 1.0)
            else:
                model_tp = tp
                other_ms_zero2_ratio = self.zero3_ratio(self.tp_size * self.dp_size // tp) if args.embed_sdp else (self.zero2_ratio(self.tp_size * self.dp_size // tp) if args.use_zero2_for_dp else 1.0)
                        
            # Handle different memory consumption scenarios based on pipeline size (PP Size)
            if self.pp_size == 1:
                tp_other_memory_cost[0] += (
                    args.other_memory_pp_off['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    args.other_memory_pp_off['activation'][tp] * 
                    other_layers_bsz * 
                    self.act_1f1b_ratio)
            else:
                if args.pipeline_type == 'pipedream_flush':
                    other_layers_bsz_first = other_layers_bsz * self.act_1f1b_ratio_first
                    other_layers_bsz_last = other_layers_bsz * self.act_1f1b_ratio_last
                else:
                    other_layers_bsz_first = other_layers_bsz_last = other_layers_bsz
                # TODO: check the correctness of other memory cost for first stage and last stage
                tp_other_memory_cost[0] += (
                    args.other_memory_pp_on['first_stage']['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    args.other_memory_pp_on['first_stage']['activation'][tp] * 
                    other_layers_bsz_first
                )
                tp_other_memory_cost[-1] += (
                    args.other_memory_pp_on['last_stage']['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    args.other_memory_pp_on['last_stage']['activation'][tp] * 
                    other_layers_bsz_last
                )

            # if checkpoint:
            #     for i in range(len(tp_other_memory_cost)):
            #         tp_other_memory_cost[i] += tp_activation_per_bsz_dict[self.tp_size] * mbsz

            for i in range(len(tp_other_memory_cost)):
                tp_other_memory_cost[i] += args.pytorch_context_mem
                
            self.other_memory_cost[tp] = tp_other_memory_cost
    
    def get_memory_cost(self):
        result = dict()
        result['parameter'] = self.parameter_size
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.model_states_size + self.activation_size
        result['other'] = self.other_memory_cost
        return result
    
class TimeCostModel:
    time_args_list = {
        'ModelArgs':['parameter_size', 'seq_length', 'hidden_size', 'layer_num'],
        'TrainArgs':['mixed_precision', 'async_grad_reduce'],
        'ParallelArgs':['sp_space', 'optimal_chunk_func'],
        'ProfileModelArgs': ['forward_computation_time'],
        'ProfileHardwareArgs':['bct_fct_coe', 'extra_overhead', 'comm_coe_dict', 'dp_overlap_coe', 'bct_overlap_coe', 'p2p_comm_coe_dict', 'costmodel_coe', 'allreduce_dict', 'all2all_dict']
    }
    
    def __init__(self, 
                strategy, 
                global_batch_size:int = 8, 
                no_comm: bool = False, 
                model_args: ModelArgs=None, 
                train_args:TrainArgs = None,
                parallel_args:ParallelArgs = None, 
                profile_model_args:ProfileModelArgs = None,
                profile_hardware_args:ProfileHardwareArgs = None):
        self.__post_init__(strategy, global_batch_size, no_comm, model_args, train_args, parallel_args, profile_model_args, profile_hardware_args)
        self.initialize()
        self.estimate_computation_time()
        self.estimate_dp_communication_cost()
        self.estimate_tp_communication_cost()
        self.estimate_pp_communication_cost()
    
    def __post_init__(self,strategy, global_batch_size:int = 8, no_comm: bool = False, 
                      model_args=None, train_args=None, parallel_args=None, profile_model_args=None, profile_hardware_args=None):
        # Validate and correct arguments
        assert all(x is not None for x in (model_args, train_args, parallel_args, profile_hardware_args)), "One or more variables are None"
        model_args.layer_num = 24 if model_args.layer_num is None else model_args.layer_num
        
        # Aggregate all arguments
        self.args = SimpleNamespace()
        self.args.strategy = strategy
        self.args.global_batch_size = global_batch_size
        self.args.no_comm = no_comm
        components = {'ModelArgs': model_args, 'TrainArgs': train_args, 'ParallelArgs': parallel_args, 'ProfileModelArgs': profile_model_args, 'ProfileHardwareArgs': profile_hardware_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in TimeCostModel.time_args_list[class_name]:
                    setattr(self.args, key, value)
    
    def initialize(self):
        args = self.args
        
        # [initialize]:initialize strategy
        self.pp_size = args.strategy[0]
        self.tp_size = args.strategy[1]
        self.dp_size = args.strategy[2]
        self.sp_space = args.sp_space
        self.fsdp = True if 'fsdp' in args.strategy[-1].keys() and args.strategy[-1]['fsdp'] else False
        self.checkpoint = True if 'cpt' in args.strategy[-1].keys() and args.strategy[-1]['cpt'] else False
        
        # Debug strategy and hardware args
        print(f"DEBUG - Initialize: strategy={args.strategy}, sp_space={args.sp_space}")
        print(f"DEBUG - Hardware args: allreduce_dict={args.allreduce_dict}, type={type(args.allreduce_dict)}")
        print(f"DEBUG - Hardware args: all2all_dict={args.all2all_dict}, type={type(args.all2all_dict)}")
        
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.sdp_size = self.tp_size * self.dp_size
            if self.tp_size == 1:
                self.sp_dict = np.inf
            else:
                print(f"DEBUG - Setting sp_dict from all2all_dict with tp_size={self.tp_size}")
                if self.tp_size in args.all2all_dict:
                    self.sp_dict = args.all2all_dict[self.tp_size]
                elif "popt" in args.all2all_dict:
                    self.sp_dict = args.all2all_dict  # Use the full dictionary for interpolation
                else:
                    print(f"WARNING: tp_size {self.tp_size} not found in all2all_dict and no popt key")
                    self.sp_dict = 0.1  # Default value
        else:
            self.sdp_size = self.dp_size
            if self.tp_size == 1:
                self.sp_dict = np.inf
            else:
                print(f"DEBUG - Setting sp_dict from allreduce_dict with tp_size={self.tp_size}")
                if self.tp_size in args.allreduce_dict:
                    self.sp_dict = args.allreduce_dict[self.tp_size]
                elif "popt" in args.allreduce_dict:
                    self.sp_dict = args.allreduce_dict  # Use the full dictionary for interpolation
                else:
                    print(f"WARNING: tp_size {self.tp_size} not found in allreduce_dict and no popt key")
                    self.sp_dict = 0.1  # Default value
        
        print(f"DEBUG - Final sp_dict: {self.sp_dict}, type={type(self.sp_dict)}")
                
        # [initialize]:copy some attributes and initialize local batch size, optimal_microbatch, parameter_size
        self.seq_len = args.seq_length
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.bsz = args.global_batch_size / self.dp_size
        self.optimal_microbatch = 1
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.parameter_size = args.parameter_size
        else:
            self.parameter_size = args.parameter_size / self.tp_size

    def estimate_computation_time(self):
        # forward & backward computation time of whole model (depending on dummy layer_num)
        args = self.args
        if isinstance(args.forward_computation_time, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            self.fct = linear_func(self.bsz / self.tp_size, *args.forward_computation_time) * self.layer_num
        else:
            self.fct = args.forward_computation_time * self.bsz / self.tp_size * self.layer_num 
            
        self.bct = self.fct * args.bct_fct_coe
        if self.checkpoint:
            self.bct += self.fct #  * 0.5
  
    def estimate_dp_communication_cost(self):
        args = self.args
        # [calculate]:calculate dp message size of whole model (depending on dummy layer_num)
        self.dp_message_size = (2 * (self.dp_size - 1) / self.dp_size * self.parameter_size) * self.layer_num
        if args.mixed_precision:
            self.dp_message_size /= 2
        
        # [calculate]:calculate fsdp_allgather_message_size 
        self.fsdp_allgather_message_size = self.dp_message_size * 0.5

        if args.no_comm:
            self.dp_message_size = 0
            
        # [calculate]:calculate dc
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.dc = args.comm_coe_dict['%d'%self.sdp_size] if '%d'%self.sdp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%self.sdp_size]
        else:
            if self.tp_size == 1 or self.dp_size == 1:
                self.dc = args.comm_coe_dict['%d'%self.dp_size] if '%d'%self.dp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%self.dp_size]
            else:
                # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                info = args.strategy[-1]
                assert 'tp' in info.keys() and info['tp'] in [0, 1]
                tp_consecutive_flag = info['tp']
                if tp_consecutive_flag:
                    self.dc = args.comm_coe_dict['%d_0'%self.dp_size]
                else:
                    self.dc = args.comm_coe_dict['%d_1'%self.dp_size]
        
        # [calculate]:calculate dc_overlap
        self.dc_overlap = self.dc * args.dp_overlap_coe 
    
    def estimate_tp_communication_cost(self):
        args = self.args
        if self.sp_space == 'tp+sp':
            # [calculate]:calculate tp comm time
            self.tp_comm_num = 4 * self.layer_num
            if self.checkpoint:
                self.tp_comm_num *= 1.5
            
            # [calculate]:calculate per_tp_message_time
            if self.tp_size == 1:
                per_tp_message_time = 0
            else:
                self.per_tp_message_size = self.bsz * self.seq_len * self.hidden_size * (2 if args.mixed_precision else 4)
                
                # Debugging
                print(f"DEBUG - TP Communication: self.tp_size={self.tp_size}, self.sp_dict={self.sp_dict}, type={type(self.sp_dict)}")
                print(f"DEBUG - TP Communication: self.per_tp_message_size={self.per_tp_message_size}, type={type(self.per_tp_message_size)}")
                
                # Check if sp_dict is iterable and has popt key
                if isinstance(self.sp_dict, dict):
                    if "popt" in self.sp_dict and isinstance(self.sp_dict["popt"], list):
                        def linear_func(x, m, c):
                            return m * x + c
                        per_tp_message_time = linear_func(1 / 1024 / 1024 * self.per_tp_message_size, *self.sp_dict["popt"])
                    elif self.per_tp_message_size in self.sp_dict:
                        per_tp_message_time = self.sp_dict[self.per_tp_message_size]
                    else:
                        print(f"WARNING: Cannot find message size {self.per_tp_message_size} in sp_dict and no popt key found")
                        per_tp_message_time = 0.1  # Default value
                elif isinstance(self.sp_dict, (int, float)):
                    print(f"WARNING: sp_dict is a scalar value: {self.sp_dict}")
                    per_tp_message_time = self.sp_dict
                else:
                    print(f"WARNING: Unknown sp_dict type: {type(self.sp_dict)}")
                    per_tp_message_time = 0.1  # Default value
            
            # [calculate]:calculate tp time
            self.tp_communication_time = self.tp_comm_num * per_tp_message_time
        else:
            # [calculate]:calculate tp message size of whole model (depending on dummy layer_num)
            tp_comm_times = 4 
            self.tp_message_size = 2 * (self.tp_size - 1) / self.tp_size * (self.bsz * self.seq_len * self.hidden_size * tp_comm_times * 4 / 1024 / 1024) * self.layer_num
            if self.checkpoint:
                self.tp_message_size *= 1.5
            if args.mixed_precision:
                self.tp_message_size /= 2
            
            # [calculate]:calculate tc
            if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
                if self.tp_size == 1 or self.dp_size == 1:
                    tc = args.comm_coe_dict['%d'%self.tp_size] if '%d'%self.tp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%self.tp_size]
                else:
                    # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                    info = args.strategy[-1]
                    assert 'tp' in info.keys() and info['tp'] in [0, 1]
                    tp_consecutive_flag = info['tp']
                    if tp_consecutive_flag:
                        tc = args.comm_coe_dict['%d_1'%self.tp_size]
                    else:
                        tc = args.comm_coe_dict['%d_0'%self.tp_size]
            else:
                if self.tp_size == 1 or self.dp_size == 1:
                    tc = args.comm_coe_dict['%d'%self.tp_size] if '%d'%self.tp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%self.tp_size]
                else:
                    # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                    info = args.strategy[-1]
                    assert 'tp' in info.keys() and info['tp'] in [0, 1]
                    tp_consecutive_flag = info['tp']
                    if tp_consecutive_flag:
                        tc = args.comm_coe_dict['%d_1'%self.tp_size]
                    else:
                        tc = args.comm_coe_dict['%d_0'%self.tp_size]  
                                      
            # [calculate]:calculate tp time
            self.tp_communication_time = self.tp_message_size * tc
  
    def estimate_pp_communication_cost(self):
        args = self.args
        self.p2p_comm_coe = None
        if self.pp_size > 1 and args.p2p_comm_coe_dict is not None:
            self.p2p_comm_coe = args.p2p_comm_coe_dict[self.pp_size]
            self.p2p_message_size = self.pp_size * 2 * self.bsz * self.seq_len * self.hidden_size * 4 / 1024 / 1024
            if args.mixed_precision:
                self.p2p_message_size = self.p2p_message_size / 2
        
    def bct_dp_overlap(self, dp_message_size, bct):
        args = self.args
        dp_overlap_time = dp_message_size * self.dc_overlap
        bct_overlap_time = bct * args.bct_overlap_coe
        if dp_overlap_time > bct_overlap_time:
            overlap_part = bct_overlap_time
            rest_part = (dp_message_size - bct_overlap_time / self.dc_overlap) * self.dc
            rest_dp_flag = True
        elif dp_overlap_time < bct_overlap_time:
            overlap_part = dp_overlap_time
            rest_part = (bct - dp_overlap_time / args.bct_overlap_coe) 
            rest_dp_flag = False
        else:
            overlap_part = bct_overlap_time
            rest_part = 0
            rest_dp_flag = False
        rest_dp_flag = False
        return overlap_part, rest_part, rest_dp_flag
    
    def gen_result(self):
        args = self.args
        if self.pp_size >= 1:
            if self.tp_size == 1 and self.dp_size > 1: # pp+dp
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                overall_overhead = self.fct + overlap_part + rest_part + args.extra_overhead
                result = overall_overhead
            elif self.dp_size == 1 and self.tp_size > 1: # pp+tp
                result = self.fct + self.bct + self.tp_communication_time
            elif self.dp_size == 1 and self.tp_size == 1: # pure pp
                result = self.fct + self.bct
            else: # pp+dp+tp
                if self.tp_size < self.tp_size * self.dp_size // 2:
                    overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                    overall_overhead = self.fct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
                    result = overall_overhead
                else:
                    overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct * 1 / 2)
                    overall_overhead = self.fct + 1 / 2 * self.bct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
                    result = overall_overhead

        # For fsdp, add allgather time of forward and backward
        if self.fsdp:
            forward_allgather_time = self.fsdp_allgather_message_size * self.dc
            result = result + forward_allgather_time * self.optimal_microbatch

        if self.pp_size > 1 and self.p2p_comm_coe is not None:
            result = result + self.p2p_message_size * self.p2p_comm_coe
        
        coe = 0.001 * args.costmodel_coe
        result = result * coe
        result = result / self.layer_num
        return result
    
class OtherTimeCostModel:
    othertime_args_list = {
        'ModelArgs': ['hidden_size'],
        'TrainArgs': ['mixed_precision'],
        'ParallelArgs': ['sp_space'],
        'ProfileModelArgs': ['other_memory_pp_on', 'other_memory_pp_off', 'other_time_profiled'],
        'ProfileHardwareArgs':['comm_coe_dict', 'allreduce_dict']
    }
    
    def __init__(self, 
                mbsz:int = 1, 
                pp_deg:int = 2, 
                world_size:int = 8, 
                vsp:bool = False, 
                embed_sdp:bool = False,
                min_tp:int = 1, 
                max_tp:int = 8, 
                sequence_length_list:list = [512], 
                model_args:ModelArgs = None, 
                train_args:TrainArgs = None, 
                parallel_args:ParallelArgs = None, 
                profile_model_args:ProfileModelArgs = None, 
                profile_hardware_args:ProfileHardwareArgs = None):
        self.__post_init__(mbsz, pp_deg, world_size, vsp, embed_sdp, min_tp, max_tp, sequence_length_list, model_args, train_args, parallel_args, profile_model_args, profile_hardware_args)
    
        args = self.args
        
        self.sequence_length_list = args.sequence_length_list
        self.hidden_size = args.hidden_size
        self.sp_space = args.sp_space
        
        self.dp_coe = dict()
        self.fct = dict()
        self.tp_time = dict()
        self.sp_size = dict()
        self.dp_size = dict()
        self.comm_factor = dict()
        
        self.estimate_tp_time()
        self.estimate_fct_time()
        self.estimate_dp_time()
    
    def __post_init__(self, mbsz:int = 1, pp_deg:int = 2, world_size:int = 8, vsp:bool = False, embed_sdp:bool = False, min_tp:int = 1, max_tp:int = 8, sequence_length_list:list = [512],
             model_args:ModelArgs = None, train_args:TrainArgs = None, parallel_args:ParallelArgs = None, profile_model_args:ProfileModelArgs = None, profile_hardware_args:ProfileHardwareArgs = None):
        # Validate
        assert all(x is not None for x in (model_args, train_args, parallel_args, profile_model_args, profile_hardware_args)), "One or more variables are None"
        
        # Aggregate all arguments
        self.args = SimpleNamespace()
        self.args.mbsz = mbsz
        self.args.pp_deg = pp_deg
        self.args.world_size = world_size
        self.args.vsp = vsp
        self.args.embed_sdp = embed_sdp
        self.args.min_tp = min_tp
        self.args.max_tp = max_tp
        self.args.sequence_length_list = sequence_length_list
        components = {'ModelArgs': model_args, 'TrainArgs': train_args, 'ParallelArgs': parallel_args, 'ProfileModelArgs': profile_model_args, 'ProfileHardwareArgs': profile_hardware_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in OtherTimeCostModel.othertime_args_list[class_name]:
                    setattr(self.args, key, value)
    
    def estimate_tp_time(self):
        args = self.args
        # calc tp comm size
        k = args.min_tp
        while k <= args.max_tp and args.world_size // args.pp_deg >= k:
            self.per_tp_message_size = []
            self.per_tp_message_time = []
            self.tp_message_size = []
            for seq_len in self.sequence_length_list:
                if args.vsp == 0:
                    if self.sp_space == 'tp+sp':
                        self.per_tp_message_size.append(args.mbsz * seq_len * args.hidden_size * (2 if args.mixed_precision else 4))
                        if k == 1:
                            self.per_tp_message_time.append(0)
                        else:
                            if self.per_tp_message_size[-1] in args.allreduce_dict:
                                self.per_tp_message_time.append(args.allreduce_dict[self.per_tp_message_size[-1]])
                            else:
                                def linear_func(x, m, c):
                                    return m * x + c
                                self.per_tp_message_time.append(linear_func( 1 / 1024 / 1024 * self.per_tp_message_size[-1], *args.allreduce_dict[k]["popt"] ))
                    else:
                        dp_size = args.world_size // args.pp_deg // k
                        if k == 1 or dp_size == 1:
                            tp_coe = args.comm_coe_dict['%d'%k] if '%d'%k in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%k]
                        else:
                            tp_coe = args.comm_coe_dict['%d_0'%k]

                        self.tp_message_size.append((k - 1) / k * (args.mbsz * seq_len * self.hidden_size / 1024/1024) * (2 if args.mixed_precision else 4))
                        self.per_tp_message_time.append(self.tp_message_size[-1] * tp_coe)
                else:
                    self.per_tp_message_time.append(0)
            if args.pp_deg == 1:
                self.tp_time[k] = sum(self.per_tp_message_time) + self.per_tp_message_time[-1] # For T5 model
            else:
                # TODO: consider embedding layer in middle stage
                self.tp_time[k] = (self.per_tp_message_time[0], self.per_tp_message_time[-1])
            k *= 2
            
    def estimate_fct_time(self):
        args = self.args
        # calc calc time (ms)
        k = args.min_tp
        while k <= args.max_tp and args.world_size // args.pp_deg >= k:
            def linear_func(x, m, c):
                return m * x + c
            if args.pp_deg == 1:
                if isinstance(args.other_time_profiled ,np.ndarray):
                    self.fct[k] = linear_func(args.mbsz / args.min_tp, *args.other_time_profiled)
                else:
                    self.fct[k] = args.mbsz / args.min_tp * k * args.other_time_profiled
            else:
                if isinstance(args.other_time_profiled, np.ndarray):
                    self.fct[k] = (linear_func(args.mbsz / args.min_tp, *args.other_time_profiled) / 2, \
                                linear_func(args.mbsz / args.min_tp, *args.other_time_profiled) / 2)
                else:
                    self.fct[k] = (args.mbsz / args.min_tp * k * args.other_time_profiled / 2, \
                                args.mbsz / args.min_tp * k * args.other_time_profiled / 2)
            k *= 2
    
    def estimate_dp_time(self):
        args = self.args
        # calc dp comm size
        k = args.min_tp
        while k <= args.max_tp and args.world_size //args.pp_deg >= k:
            if args.vsp == 0:
                dp_size = args.world_size // args.pp_deg // k
                if k == 1 or dp_size == 1:
                    self.dp_coe[k] = args.comm_coe_dict['%d'%dp_size] if '%d'%dp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%dp_size]
                else:
                    self.dp_coe[k] = args.comm_coe_dict['%d_0'%dp_size]
                self.dp_coe[k] *= (dp_size - 1) / dp_size # bus -> alg
            else:
                dp_size = args.world_size // args.pp_deg
                self.dp_coe[k] = args.comm_coe_dict['%d'%dp_size] if '%d'%dp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%dp_size]
                self.dp_coe[k] *= (dp_size - 1) / dp_size # bus -> alg
            if args.pp_deg == 1:
                if args.vsp == 0:
                    self.dp_size[k] = args.other_memory_pp_off['model_states'][k] / 4
                else:
                    self.dp_size[k] = args.other_memory_pp_off['model_states'][1] / 4
            else:
                if args.vsp == 0:
                    self.dp_size[k] = (args.other_memory_pp_on['first_stage']['model_states'][k] / 4, args.other_memory_pp_on['first_stage']['model_states'][k] / 4)
                else:
                    self.dp_size[k] = (args.other_memory_pp_on['last_stage']['model_states'][1] / 4, args.other_memory_pp_on['last_stage']['model_states'][1] / 4)
            k *= 2

        if args.embed_sdp:
            self.factor = 1.5
        else:
            self.factor = 1.0

    def gen_result(self):
        args = self.args
        other_time_cost = dict()
        k = args.min_tp
        for k in self.dp_size.keys():
            other_time_cost[k] = [0] * args.pp_deg 
            # TODO: add overlap
            if args.pp_deg  == 1:
                other_time_cost[k][0] = 0.001 * (self.dp_size[k] * self.dp_coe[k] * self.factor + self.fct[k] * 3 + self.tp_time[k]) # + 4 * self.sp_time[k] # fwd + bwd
            else:
                other_time_cost[k][0] = 0.001 * (self.dp_size[k][0] * self.dp_coe[k] * self.factor + self.fct[k][0] * 3 + self.tp_time[k][0]) # + 2 * self.sp_time[k]
                other_time_cost[k][-1] = 0.001 * (self.dp_size[k][-1] * self.dp_coe[k] * self.factor + self.fct[k][-1] * 3 + self.tp_time[k][-1]) # + 2 * self.sp_time[k]
        return other_time_cost


def chunk_like_torch(size, chunks):
    """Implement torch.arange(size).chunk(chunks) behavior using numpy"""
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    
    # Calculate chunk size like PyTorch does
    chunk_size = (size + chunks - 1) // chunks  # ceiling division
    
    # Create splits
    splits = []
    for i in range(chunks):
        start = i * chunk_size
        if start >= size:
            break
        end = min(start + chunk_size, size)
        splits.append(np.arange(start, end))
    
    return splits

def get_real_chunk(local_bsz, chunk):
    if chunk == 1:
        return 1
    chunk = int(chunk)
    re = [t.shape[0] for t in chunk_like_torch(int(local_bsz), chunk)]
    return len(re)

def get_time_cost_all_stages(layer_timecosts, pp_stage_division):
    assert(np.sum(pp_stage_division)==len(layer_timecosts))
    stage_timecosts = []
    for stage_id in range(len(pp_stage_division)):
        layer_start_id, layer_end_id = int(np.sum(pp_stage_division[:stage_id])), int(np.sum(pp_stage_division[:stage_id+1]))
        stage_timecosts.append(np.sum(layer_timecosts[layer_start_id:layer_end_id]))
    return stage_timecosts

def pipeline_costmodel(timecostmodel, layer_num_list, model_args_list, train_args_list, parallel_args_list, profile_model_args_list, profile_hardware_args_list, strategies, partition, chunks, bsz, min_tp, other_time_cost, return_stage_cost=False):
    if strategies is None:
        if return_stage_cost:
            return [np.inf] * len(partition), np.inf
        else:
            return np.inf
    layer_type_ids = []
    # print(layer_num_list)
    for layer_type_id in range(len(layer_num_list)):
        layer_type_ids += [layer_type_id] * layer_num_list[layer_type_id]
    if isinstance(chunks, list):
        chunks = [get_real_chunk(int(bsz/(strategies[0][1] * strategies[0][2] // min_tp)), chunks_) for chunks_ in chunks]
        bsz_chunked = [bsz / chunks_ for chunks_ in chunks]
        max_chunk = np.max(chunks)
        # print('Detected multi chunks!', chunks, 'Using %d as chunks!'%max_chunk)
    else:
        chunks = get_real_chunk(int(bsz/(strategies[0][1] * strategies[0][2] // min_tp)), chunks)
        bsz_chunked = [bsz / chunks] * len(layer_num_list)
        # print(bsz, bsz/chunks, chunks)
        max_chunk = chunks
         
    pp_deg = len(partition)
    layer_num = len(strategies)
    from galvatron.utils import form_strategy, strategy_str2list
    strategies_set = list(set([form_strategy(s) for s in strategies]))
    timecosts_dict_bsz_chunked, timecosts_dict_compute = {}, {}
    for layer_type_id in range(len(layer_num_list)):
        timecosts_dict_bsz_chunked[layer_type_id], timecosts_dict_compute[layer_type_id] = {}, {}
        for s in strategies_set:
            timecosts_dict_bsz_chunked[layer_type_id][s] = timecostmodel(strategy_str2list(s), bsz_chunked[layer_type_id],
                                                                        model_args=model_args_list[layer_type_id], train_args=train_args_list[layer_type_id],
                                                                        parallel_args=parallel_args_list[layer_type_id], profile_model_args=profile_model_args_list[layer_type_id],
                                                                        profile_hardware_args=profile_hardware_args_list[layer_type_id]).gen_result()
            timecosts_dict_compute[layer_type_id][s] = timecostmodel(strategy_str2list(s), bsz_chunked[layer_type_id], no_comm=True, 
                                                                    model_args=model_args_list[layer_type_id], train_args=train_args_list[layer_type_id],
                                                                    parallel_args=parallel_args_list[layer_type_id], profile_model_args=profile_model_args_list[layer_type_id],
                                                                    profile_hardware_args=profile_hardware_args_list[layer_type_id]).gen_result()
    timecosts_bsz_chunked = [timecosts_dict_bsz_chunked[layer_type_ids[i]][form_strategy(strategies[i])] for i in range(layer_num)]
    timecosts_bsz_compute = [timecosts_dict_compute[layer_type_ids[i]][form_strategy(strategies[i])] for i in range(layer_num)]
    stage_costs_bsz_chunked = get_time_cost_all_stages(timecosts_bsz_chunked, partition)
    stage_costs_compute = get_time_cost_all_stages(timecosts_bsz_compute, partition)
    assert(len(other_time_cost) == len(stage_costs_compute))
    for i in range(len(other_time_cost)):
        stage_costs_compute[i] += other_time_cost[i]
    # print(timecosts_bsz_chunked, stage_costs_bsz_chunked, np.sum(stage_costs_bsz_chunked))
    # print(stage_costs_compute, np.max(stage_costs_compute))
    # print(np.sum(stage_costs_bsz_chunked), np.max(stage_costs_compute), np.max(stage_costs_compute) * (max_chunk-1))
    
    # # p2p & reduce sync
    # result = np.sum(stage_costs_bsz_chunked) + np.max(stage_costs_compute) * (max_chunk-1)
    
    # p2p & reduce async
    stage_costs_reduce = [total for total in stage_costs_bsz_chunked]
    # print(stage_costs_compute, stage_costs_reduce, stage_costs_bsz_chunked)
    result = np.sum(stage_costs_compute) + stage_costs_compute[-1] * (max_chunk - 1)
    # assume t_rank0 > t_rank1 > ... , warmup and cool down bubble can be overlapped
    result = max( result,
            max( min(pp_deg - 1, max_chunk - 1) * stage_costs_compute[0] * 1/3, np.sum(stage_costs_compute[1:]) * 1/3) + 
            max( min(pp_deg - 1, max_chunk - 1) * stage_costs_compute[0] * 2/3, np.sum(stage_costs_compute[1:]) * 2/3) + 
            stage_costs_compute[0] * max(0, max_chunk + 1 - pp_deg))
    
    # result += max(np.max(stage_costs_compute) * 2/3 * (max_chunk - 1), stage_costs_compute[-1] * (max_chunk - 1))
    # result = np.max(stage_costs_compute) * (max_chunk-1+pp_deg)
    for i in range(pp_deg):
        stage_costs_reduce[i] -= np.sum(stage_costs_compute[:i+1])
    reduce_time = np.max(stage_costs_reduce)
    reduce_time = reduce_time if reduce_time > 0 else 0
    
    # print(result,reduce_time)
    result += reduce_time
    
    if return_stage_cost:
        return stage_costs_bsz_chunked, result
    return result