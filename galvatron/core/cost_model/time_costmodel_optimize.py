import numpy as np
from logging import Logger
from types import SimpleNamespace
from .cost_model_args_optimize import TrainArgsOptimize, ProfileModelArgsOptimize, ProfileHardwareArgsOptimize, UtilsArgsOptimize, VersionOptionArgsOptimize
from .cost_model_args_optimize import EstimateTPTimeType

class TimeCostModelOptimize:
    time_args_list = {
        'TrainArgsOptimize': ['seq_length', 'hidden_size', 'mixed_precision'],
        'ProfileModelArgsOptimize': ['forward_computation_time', 'parameter_memory'],
        'ProfileHardwareArgsOptimize': ['bct_fct_coe', 'overlap_slowdown_coe', 'comm_coe_dict', 'p2p_comm_coe_dict', 'allreduce_dict', 'all2all_dict'],
        'UtilsArgsOptimize': ['extra_overhead', 'costmodel_coe', 'dummy_layer_num'],
        'VersionOptionArgsOptimize': ['estimate_tp_time_type'],
    }
    
    def __init__(self, 
                strategy,
                global_batch_size: int = 8,
                chunks: int = 1,
                no_sync_gradients: bool = False,
                logger: Logger = None,
                train_args: TrainArgsOptimize = None,
                profile_model_args: ProfileModelArgsOptimize = None,
                profile_hardware_args: ProfileHardwareArgsOptimize = None,
                utils_args: UtilsArgsOptimize = None,
                version_option_args: VersionOptionArgsOptimize = None,
               ):
        self.__post_init__(strategy, global_batch_size, chunks, no_sync_gradients, logger, train_args, profile_model_args, profile_hardware_args, utils_args, version_option_args)
        self.initialize()
        self.estimate_computation_time()
        self.estimate_dp_communication_cost()
        self.estimate_tp_communication_cost()
        self.estimate_pp_communication_cost()
        
    def __post_init__(self, strategy, global_batch_size, chunks, no_sync_gradients, logger, train_args, profile_model_args, profile_hardware_args, utils_args, version_option_args):
        # validate args
        assert all(x is not None for x in (train_args, profile_model_args, profile_hardware_args, utils_args, version_option_args)), "All argument groups must be provided."

        # Aggregate all arguments
        self.args = SimpleNamespace()
        self.args.strategy = strategy
        self.args.global_batch_size = global_batch_size
        self.args.chunks = chunks
        self.args.no_sync_gradients = no_sync_gradients
        self.args.logger = logger
        
        components = {'TrainArgsOptimize': train_args, 'ProfileModelArgsOptimize': profile_model_args, 'ProfileHardwareArgsOptimize': profile_hardware_args, 'UtilsArgsOptimize': utils_args, 'VersionOptionArgsOptimize': version_option_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in TimeCostModelOptimize.time_args_list[class_name]:
                    setattr(self.args, key, value)
    
    def initialize(self):
        args = self.args
        
        # initialize strategy related attributes
        self.pp_size = args.strategy[0]
        self.tp_size = args.strategy[1]
        self.dp_size = args.strategy[2]
        self.use_ulysses = True if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1 else False
        self.sdp_size = self.tp_size * self.dp_size if self.use_ulysses else self.dp_size
        self.fsdp = True if 'fsdp' in args.strategy[-1].keys() and args.strategy[-1]['fsdp'] else False
        self.checkpoint = True if 'cpt' in args.strategy[-1].keys() and args.strategy[-1]['cpt'] else False
        
        # select tp/sp communication dict
        self.sp_dict = np.inf if self.tp_size == 1 else (args.all2all_dict[self.tp_size] if self.use_ulysses else args.allreduce_dict[self.tp_size])
                
        # calculate some information
        self.mbsz = args.global_batch_size // args.chunks // self.dp_size # still use dp_size here
        self.parameter_memory = args.parameter_memory if self.use_ulysses else args.parameter_memory / self.tp_size
    
        # copy some attributes
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        self.dummy_layer_num = args.dummy_layer_num
    
    def estimate_computation_time(self):
        # forward & backward computation time of whole model (depending on dummy layer_num)
        args = self.args
        if isinstance(args.forward_computation_time, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            self.fct = linear_func(self.mbsz / self.tp_size, *args.forward_computation_time) * self.dummy_layer_num # (self.mbsz/self.tp_size) means computation split over tp devices.
        else:
            self.fct = args.forward_computation_time * self.mbsz / self.tp_size * self.dummy_layer_num # (self.mbsz/self.tp_size) means computation split over tp devices.

        self.bct = self.fct * args.bct_fct_coe
        if self.checkpoint:
            self.bct += self.fct
            
    def estimate_dp_communication_cost(self):
        args = self.args
        
        # Get dc and dc_slowdown
        self.dc = args.comm_coe_dict[f'{self.sdp_size}']
        self.dc_slowdown = self.dc * args.overlap_slowdown_coe
        
        # [calculate]:calculate dp message size of whole model (depending on dummy layer_num)
        self.dp_message_size = (2 * (self.dp_size - 1) / self.dp_size * self.parameter_memory) * self.dummy_layer_num # still use dp_size here
        if args.mixed_precision:
            self.dp_message_size /= 2
        
        # [calculate]:calculate fsdp_allgather_message_size 
        self.fsdp_allgather_message_size = self.dp_message_size * 0.5
        self.fsdp_allgather_time = self.fsdp_allgather_message_size * self.dc # TODO consider overlap

        # when no_sync_gradients is set, no dp communication
        if args.no_sync_gradients:
            self.dp_message_size = 0
    
    def estimate_tp_communication_cost(self): # TODO when use ulysses, consider GQA
        args = self.args
        self.tp_communication_time = 0
        
        if args.estimate_tp_time_type == EstimateTPTimeType.FIXED:
            if self.use_ulysses == False:
                tp_comm_times = 4 * self.dummy_layer_num
                if self.checkpoint:
                    tp_comm_times *= 1.5
                # self.tp_message_size = 2 * (self.tp_size - 1) / self.tp_size * self.mbsz * self.seq_length * self.hidden_size * tp_comm_times * 4 / 1024 / 1024
                self.tp_message_size = (self.tp_size - 1) / self.tp_size * self.mbsz * self.seq_length * self.hidden_size * tp_comm_times * 4 / 1024 / 1024
                if args.mixed_precision:
                    self.tp_message_size /= 2
                self.tc = args.comm_coe_dict[f'{self.tp_size}']
                self.tp_communication_time = self.tc * self.tp_message_size
            else:
                raise NotImplementedError("Ulysses TP communication time estimation not implemented yet.")
        
        elif args.estimate_tp_time_type == EstimateTPTimeType.FIT:
            tp_comm_times = 4 * self.dummy_layer_num
            if self.checkpoint:
                tp_comm_times *= 1.5
            if self.tp_size == 1:
                self.tp_communication_time = 0
            else:
                self.tp_message_size = self.mbsz * self.seq_length * self.hidden_size * tp_comm_times * 4 / 1024 / 1024 # check 是否是真的没有*2
                if args.mixed_precision:
                    self.tp_message_size /= 2
                if self.tp_message_size in self.sp_dict:
                    self.tp_communication_time = self.sp_dict[f'{self.tp_message_size}']
                else:
                    def linear_func(x, m, c):
                        return m * x + c
                    self.tp_communication_time = linear_func(self.tp_message_size, *self.sp_dict[["popt"]])
        else:
            raise ValueError(f"Invalid estimate_tp_time_type: {args.estimate_tp_time_type}. Supported types are 'FIXED' and 'FIT'.")
    
    def estimate_pp_communication_cost(self): # TODO Consider high-speed P2P and low-speed P2P
        args = self.args
        if self.pp_size > 1:
            self.pp_comm_coe = args.p2p_comm_coe_dict[f'{self.pp_size}']
            self.pp_message_size = self.pp_size * 2 * self.mbsz * self.seq_length * self.hidden_size * 4 / 1024 / 1024
            if args.mixed_precision:
                self.pp_message_size /= 2
            self.pp_time = self.pp_comm_coe * self.pp_message_size
        else:
            self.pp_time = 0
            
    def bct_dp_overlap(self, dp_message_size, bct):
        args = self.args
        dp_slowdown_time = dp_message_size * self.dc_slowdown
        bct_slowdown_time = bct * args.overlap_slowdown_coe
        
        if dp_slowdown_time > bct_slowdown_time:
            overlap_part = bct_slowdown_time
            rest_part = (dp_message_size - overlap_part / self.dc_slowdown) * self.dc
            rest_dp_flag = True
        elif dp_slowdown_time < bct_slowdown_time:
            overlap_part = dp_slowdown_time
            rest_part = bct - overlap_part / args.overlap_slowdown_coe
            rest_dp_flag = False
        else:
            overlap_part = dp_slowdown_time
            rest_part = 0
            rest_dp_flag = False
        return overlap_part, rest_part, rest_dp_flag
        
    def gen_result(self):
        args = self.args
        
        # calculate dp+tp time
        if self.tp_size == 1 and self.dp_size > 1:
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
            overall_overhead = self.fct + overlap_part + rest_part + args.extra_overhead
            result = overall_overhead
        elif self.dp_size == 1 and self.tp_size > 1:
            result = self.fct + self.bct + self.tp_communication_time
        else:
            if self.tp_size < self.tp_size * self.dp_size // 2:
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                overall_overhead = self.fct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
                result = overall_overhead
            else:
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct * 1 / 2)
                overall_overhead = self.fct + 1 / 2 * self.bct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
                result = overall_overhead
        
        # consider fsdp and scale by dummy_layer_num
        if self.fsdp:
            result += self.fsdp_allgather_time
        result /= self.dummy_layer_num 
        
        # consider pp communication time
        if self.pp_size > 1:
            result += self.pp_time
        
        # consider costmodel coefficient and transform from ms to s
        result = result * args.costmodel_coe / 1000

        return result

class OtherTimeCostModelOptimize:
    othertime_args_list = {
        'TrainArgsOptimize': ['hidden_size', 'mixed_precision'],
        'ProfileModelArgsOptimize': ['other_memory_pp_on', 'other_memory_pp_off', 'other_time_profiled'],
        'ProfileHardwareArgsOptimize': ['comm_coe_dict', 'allreduce_dict', 'overlap_slowdown_coe', 'bct_fct_coe'],
        'VersionOptionArgsOptimize': ['estimate_tp_time_type'],
    }
    
    def __init__(self,
                mbsz: int = 1, # mbsz = global_batch_size // chunks // (world_size // pp_size // min_tp)
                pp_deg: int = 1,
                world_size: int = 8,
                vsp: bool = False,
                embed_sdp: bool = False,
                min_tp: int = 1,
                max_tp: int = 1,
                sequence_length_list:list = [512],
                train_args: TrainArgsOptimize = None,
                profile_model_args: ProfileModelArgsOptimize = None,
                profile_hardware_args: ProfileHardwareArgsOptimize = None,
                version_option_args: VersionOptionArgsOptimize = None,
                logger: Logger=None):
        self.__post_init__(mbsz, pp_deg, world_size, vsp, embed_sdp, min_tp, max_tp, sequence_length_list, train_args, profile_model_args, profile_hardware_args, version_option_args, logger)
        self.initialize()
        self.estimate_fct_time()
        self.estimate_dp_time()
        self.estimate_tp_time()
        
    def __post_init__(self, mbsz, pp_deg, world_size, vsp, embed_sdp, min_tp, max_tp, sequence_length_list, train_args, profile_model_args, profile_hardware_args, version_option_args, logger):
        # validate args
        assert all(x is not None for x in (train_args, profile_model_args, profile_hardware_args, version_option_args)), "All argument groups must be provided."
        
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
        self.logger = logger
        components = {'TrainArgsOptimize': train_args, 'ProfileModelArgsOptimize': profile_model_args, 'ProfileHardwareArgsOptimize': profile_hardware_args, 'VersionOptionArgsOptimize': version_option_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in OtherTimeCostModelOptimize.othertime_args_list[class_name]:
                    setattr(self.args, key, value)
    
    def initialize(self):
        args = self.args
        
        # calculate all tp range
        self.tp_range = []
        tp_consider = args.min_tp
        while tp_consider <= args.max_tp and tp_consider * args.pp_deg <= args.world_size:
            self.tp_range.append(tp_consider)
            tp_consider *= 2
    
    def estimate_fct_time(self):
        args = self.args
        
        self.fct = dict()
        for tp_size in self.tp_range:
            def linear_func(x, m, c):
                return m * x + c
            """
                Assuming tp_size = x * min_tp, dp_size = max_dp / x, and current_mbsz = x * mbsz,
                the model parameter load decreases by x while data load increases by x, keeping total load similar.
                Thus, the formula below uses the load at min_tp, with mbsz spread over min_tp devices.
            """
            if isinstance(args.other_time_profiled, np.ndarray):
                self.fct_time = linear_func(args.mbsz / args.min_tp, *args.other_time_profiled)
            else:
                self.fct_time = args.other_time_profiled * args.mbsz / args.min_tp
                
            if args.pp_deg == 1:
                self.fct[tp_size] = self.fct_time
            else:
                self.fct[tp_size] = (self.fct_time / 2, self.fct_time / 2) 
    
    def estimate_dp_time(self):
        args = self.args
        
        self.dp_coe = dict()
        self.dp_message_size = dict()
        for tp_size in self.tp_range:
            sdp_size = args.world_size // args.pp_deg if args.vsp else args.world_size // args.pp_deg // tp_size
            self.dp_coe[tp_size] = args.comm_coe_dict[f'{sdp_size}'] * (sdp_size - 1) / sdp_size
            
            if args.pp_deg == 1:
                if args.vsp == 0:
                    self.dp_message_size[tp_size] = args.other_memory_pp_off['model_states'][tp_size] / 4
                else:
                    self.dp_message_size[tp_size] = args.other_memory_pp_off['model_states'][1] / 4
            else:
                if args.vsp == 0:
                    self.dp_message_size[tp_size] = (args.other_memory_pp_on['first_stage']['model_states'][tp_size] / 4, args.other_memory_pp_on['first_stage']['model_states'][tp_size] / 4)
                else:
                    self.dp_message_size[tp_size] = (args.other_memory_pp_on['last_stage']['model_states'][1] / 4, args.other_memory_pp_on['last_stage']['model_states'][1] / 4)
    
        if args.embed_sdp:
            self.fwd_factor = 0.5 # fsdp allgather param(0.5)
            self.bwd_factor = 1.0 # fsdp allgather param(0.5) + dp reduce gradient(0.5)
        else:
            self.fwd_factor = 0.0 # nothing
            self.bwd_factor = 0.5 # dp reduce gradient(0.5)
            
    def estimate_tp_time(self):
        args = self.args
        
        self.tp_time = dict()
        for tp_size in self.tp_range:
            tp_message_time = []
            for seq_len in args.sequence_length_list:
                if args.vsp:
                    tp_message_time.append(0)
                else:
                    if args.estimate_tp_time_type == EstimateTPTimeType.FIXED:
                        tp_coe = args.comm_coe_dict[f'{tp_size}']
                        bsz_scale = args.mbsz * (tp_size // args.min_tp)
                        tp_message_size = (tp_size - 1) / tp_size * bsz_scale * seq_len * args.hidden_size * 4 / 1024 / 1024
                        if args.mixed_precision:
                            tp_message_size /= 2
                        tp_message_time.append(tp_coe * tp_message_size)
                        
                    elif args.estimate_tp_time_type == EstimateTPTimeType.FIT:
                        bsz_scale = args.mbsz * (tp_size // args.min_tp)
                        tp_message_size = bsz_scale * seq_len * args.hidden_size * 4 / 1024 / 1024
                        if args.mixed_precision:
                            tp_message_size /= 2
                        if tp_size == 1:
                            tp_message_time.append(0)
                        else:
                            if tp_message_size in args.allreduce_dict:
                                tp_message_time.append(args.allreduce_dict[f'{tp_message_size}'])
                            else:
                                def linear_func(x, m, c):
                                    return m * x + c
                                tp_message_time.append(linear_func(tp_message_size, *args.allreduce_dict[f'{tp_size}'][["popt"]]))
                    else:
                        raise ValueError(f"Invalid estimate_tp_time_type: {args.estimate_tp_time_type}. Supported types are 'FIXED' and 'FIT'.")
            if args.pp_deg == 1:
                self.tp_time[tp_size] = sum(tp_message_time) + tp_message_time[-1] # For T5 model
            else:
                self.tp_time[tp_size] = (tp_message_time[0], tp_message_time[-1])
    
    # In new vesion, we assume that comm overlap_coe(bct_overlap_coe)=1, so we only need to calculate comp overlap time
    def get_overlap_time(self, forward_comm_time, forward_comp_time, backward_comm_time, backward_comp_time, tp_time):
        forward_comp_time = forward_comp_time * self.args.overlap_slowdown_coe
        backward_comp_time = backward_comp_time * self.args.overlap_slowdown_coe
        if forward_comp_time > forward_comm_time:
            forward_time = forward_comm_time + (forward_comp_time - forward_comm_time) / self.args.overlap_slowdown_coe
        else:
            forward_time = forward_comm_time
        if backward_comp_time > backward_comm_time:
            backward_time = backward_comm_time + (backward_comp_time - backward_comm_time) / self.args.overlap_slowdown_coe
        else:
            backward_time = backward_comm_time
        return forward_time + backward_time + tp_time
    
    def gen_result(self):
        args = self.args
        other_time_cost = dict()
        other_time_cost_no_sync_gradient = dict()
        
        for tp_size in self.tp_range:
            other_time_cost[tp_size] = [0] * args.pp_deg
            other_time_cost_no_sync_gradient[tp_size] = [0] * args.pp_deg
            if args.pp_deg == 1:
                other_time_cost[tp_size][0] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size], self.dp_message_size[tp_size] * self.dp_coe[tp_size] * self.bwd_factor, self.fct[tp_size] * self.args.bct_fct_coe, self.tp_time[tp_size])
                other_time_cost_no_sync_gradient[tp_size][0] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size], self.dp_message_size[tp_size] * self.dp_coe[tp_size] * (self.bwd_factor - 0.5), self.fct[tp_size] * self.args.bct_fct_coe, self.tp_time[tp_size])
            else:
                other_time_cost[tp_size][0] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size][0] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size][0], self.dp_message_size[tp_size][0] * self.dp_coe[tp_size] * self.bwd_factor, self.fct[tp_size][0] * self.args.bct_fct_coe, self.tp_time[tp_size][0])
                other_time_cost[tp_size][-1] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size][-1] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size][-1], self.dp_message_size[tp_size][-1] * self.dp_coe[tp_size] * self.bwd_factor, self.fct[tp_size][-1] * self.args.bct_fct_coe, self.tp_time[tp_size][-1])
                other_time_cost_no_sync_gradient[tp_size][0] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size][0] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size][0], self.dp_message_size[tp_size][0] * self.dp_coe[tp_size] * (self.bwd_factor - 0.5), self.fct[tp_size][0] * self.args.bct_fct_coe, self.tp_time[tp_size][0])
                other_time_cost_no_sync_gradient[tp_size][-1] = 0.001 * self.get_overlap_time(self.dp_message_size[tp_size][-1] * self.dp_coe[tp_size] * self.fwd_factor, self.fct[tp_size][-1], self.dp_message_size[tp_size][-1] * self.dp_coe[tp_size] * (self.bwd_factor - 0.5), self.fct[tp_size][-1] * self.args.bct_fct_coe, self.tp_time[tp_size][-1])
        return other_time_cost, other_time_cost_no_sync_gradient

class AttentionTimeCostModelOptimize(TimeCostModelOptimize):
    def __init__(self, 
                strategy,
                global_batch_size: int = 8,
                chunks: int = 1,
                no_sync_gradients: bool = False,
                logger: Logger = None,
                train_args: TrainArgsOptimize = None,
                profile_model_args: ProfileModelArgsOptimize = None,
                profile_hardware_args: ProfileHardwareArgsOptimize = None,
                utils_args: UtilsArgsOptimize = None,
                version_option_args: VersionOptionArgsOptimize = None,
               ):
        super().__init__(strategy, global_batch_size, chunks, no_sync_gradients, logger, 
                         train_args, profile_model_args, profile_hardware_args, utils_args, version_option_args)

class FFNTimeCostModelOptimize:
    # TODO: implement FFN time cost model
    def __init__(self, logger: Logger=None):
        self.logger = logger

