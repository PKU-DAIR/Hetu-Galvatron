from logging import Logger
from .cost_model_args_optimize import TrainArgsOptimize, ProfileModelArgsOptimize, ProfileHardwareArgsOptimize, UtilsArgsOptimize, VersionOptionArgsOptimize
import numpy as np
from types import SimpleNamespace

class MemoryCostModelOptimize:
    memory_cost_model_dict = {
        'TrainArgsOptimize': ['mixed_precision', 'async_grad_reduce', 'use_zero2_for_dp', 'sequence_parallel', 'pipeline_type'],
        'ProfileModelArgsOptimize': ['parameter_memory', 'tp_activation_per_bsz_dict' ],
        'VersionOptionArgsOptimize': ['zero_with_slight_noise' ],
    }
    
    def __init__(self,
                strategy,
                global_batch_size: int = 8,
                chunks: int = 1,
                stage_idx: int = 0,
                train_args: TrainArgsOptimize = None,
                profile_model_args: ProfileModelArgsOptimize = None,
                version_option_args: VersionOptionArgsOptimize = None,
                logger: Logger = None): 
        self.__post_init__(strategy, global_batch_size, chunks, stage_idx, train_args, profile_model_args, version_option_args, logger)
        self.initialize()
        self.estimate_parameter_memory()
        self.estimate_model_states_memory()
        self.estimate_activation_memory()

    def __post_init__(self, strategy, global_batch_size, chunks, stage_idx, train_args, profile_model_args, version_option_args, logger):
        # validation
        assert all(x is not None for x in (train_args, profile_model_args, version_option_args)), "All arguments must be provided and not None."
    
        # Aggregate arguments
        self.args = SimpleNamespace()
        self.args.strategy = strategy
        self.args.global_batch_size = global_batch_size
        self.args.chunks = chunks
        self.args.stage_idx = stage_idx
        self.logger = logger
        
        components = {'TrainArgsOptimize': train_args, 'ProfileModelArgsOptimize': profile_model_args, 'VersionOptionArgsOptimize': version_option_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in MemoryCostModelOptimize.memory_cost_model_dict[class_name]:
                    setattr(self.args, key, value)
                    
    def initialize(self):
        args = self.args
        
        # initialize parallel sizes
        self.pp_size = args.strategy[0]
        self.tp_size = args.strategy[1]
        self.dp_size = args.strategy[2]
        self.use_ulysses = True if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] else False
        self.sdp_size = self.tp_size * self.dp_size if self.use_ulysses else self.dp_size
    
        # copy some attributes
        self.chunks = args.chunks
        
        # initialize bsz according to stage_idx
        self.bsz = args.global_batch_size // self.dp_size
        if (args.pipeline_type == 'pipedream_flush' and self.pp_size > 1) or self.pp_size == 1:
            microbatches = [t.shape[0] for t in chunk_like_torch(args.global_batch_size // self.dp_size, self.chunks)]
            assert self.chunks == len(microbatches)
            end = self.pp_size - args.stage_idx if self.pp_size - args.stage_idx <= self.chunks else self.chunks
            act_1f1b_ratio = np.sum(microbatches[:end]) / np.sum(microbatches)
            self.bsz = act_1f1b_ratio * self.bsz
        else:
            microbatches = [t.shape[0] for t in chunk_like_torch(args.global_batch_size // self.dp_size, self.chunks)]
            self.bsz = microbatches[0]
    
        # initialize zero2 and zero3 ratio
        if args.zero_with_slight_noise:
            if self.chunks == 1:
                self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                self.zero3_ratio = lambda d: (1/d + 0.003)
            else:
                if args.async_grad_reduce:
                    self.zero2_ratio = (lambda d: (6/8 * (1/d + 0.003) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d + 0.003) + 2/4))
                    self.zero3_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                else:
                    self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                    self.zero3_ratio = lambda d: (1/d + 0.003)
                    # *5/4: for fp32 grad 
        else:
            if self.chunks == 1:
                self.zero2_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                self.zero3_ratio = lambda d: (1/d)
            else:
                if args.async_grad_reduce:
                    self.zero2_ratio = (lambda d: (6/8 * (1/d) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d) + 2/4))
                    self.zero3_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                else:
                    self.zero2_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                    self.zero3_ratio = lambda d: (1/d)
    
    def estimate_parameter_memory(self):
        args = self.args
        if self.use_ulysses:
            self.parameter_memory = args.parameter_memory
        else:
            self.parameter_memory = args.parameter_memory / self.tp_size
            
    def estimate_model_states_memory(self):
        args = self.args
        self.model_states_memory = 4 * self.parameter_memory
        if 'fsdp' in args.strategy[-1].keys() and args.strategy[-1]['fsdp']:
            # fsdp_model_states memory is slightly larger than dp_model_states/dp_size
            # we add a small bias to ensure the predicted fsdp memory NOT smaller than real value
            # Actually, this bias barely affect search result.
            self.model_states_memory *= self.zero3_ratio(self.sdp_size)
        elif 'fsdp' in args.strategy[-1].keys() and args.strategy[-1]['fsdp'] == 0 and args.use_zero2_for_dp:
            self.model_states_memory *= self.zero2_ratio(self.sdp_size)
    
    def estimate_activation_memory(self):
        args = self.args
        if 'cpt' in args.strategy[-1].keys() and args.strategy[-1]['cpt']:
            assert(args.tp_activation_per_bsz_dict['checkpoint'] is not None)
            self.activation_memory = args.tp_activation_per_bsz_dict['checkpoint'] * self.bsz
            if args.sequence_parallel:
                self.activation_memory /= self.tp_size
        else:
            self.activation_memory = args.tp_activation_per_bsz_dict[self.tp_size] * self.bsz

    def get_memory_cost(self):
        result = dict()
        result['parameter_memory'] = self.parameter_memory
        result['model_states_memory'] = self.model_states_memory
        result['activation_memory'] = self.activation_memory
        result['total_memory'] = self.model_states_memory + self.activation_memory
        return result

class OtherMemoryCostModelOptimize:
    other_memory_cost_model_dict = {
        'TrainArgsOptimize': ['disable_vtp', 'mixed_precision', 'async_grad_reduce', 'use_zero2_for_dp', 'pipeline_type'],
        'ProfileModelArgsOptimize': ['other_memory_pp_off', 'other_memory_pp_on' ],
        'UtilsArgsOptimize': ['pytorch_context_mem' ],
        'VersionOptionArgsOptimize': ['zero_with_slight_noise' ],
    }
    def __init__(self, 
                pp_deg: int = 1,
                global_batch_size: int = 8,
                chunks: int = 1,
                world_size: int = 8,
                vsp: bool = False,
                embed_sdp: bool = False,
                min_tp: int = 1,
                max_tp: int = 8,
                train_args: TrainArgsOptimize=None,
                profile_model_args: ProfileModelArgsOptimize=None,
                utils_args: UtilsArgsOptimize=None,
                version_option_args: VersionOptionArgsOptimize=None,
                logger: Logger=None):
        self.__post_init__(pp_deg, global_batch_size, chunks, world_size, vsp, embed_sdp, min_tp, max_tp, train_args, profile_model_args, utils_args, version_option_args, logger)
        self.initialize()
        self.estimate_other_memory()

    def __post_init__(self, pp_deg, global_batch_size, chunks, world_size, vsp, embed_sdp, min_tp, max_tp, train_args, profile_model_args, utils_args, version_option_args, logger):
        assert all(x is not None for x in (train_args, profile_model_args, utils_args, version_option_args)), "All arguments must be provided and not None."
        
        self.args = SimpleNamespace()
        self.args.pp_deg = pp_deg
        self.args.global_batch_size = global_batch_size
        self.args.chunks = chunks
        self.args.world_size = world_size
        self.args.vsp = vsp
        self.args.embed_sdp = embed_sdp
        self.args.min_tp = min_tp
        self.args.max_tp = max_tp
        self.logger = logger
        
        components = {'TrainArgsOptimize': train_args, 'ProfileModelArgsOptimize': profile_model_args, 'UtilsArgsOptimize': utils_args, 'VersionOptionArgsOptimize': version_option_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in OtherMemoryCostModelOptimize.other_memory_cost_model_dict[class_name]:
                    setattr(self.args, key, value)

    def initialize(self):
        args = self.args
        
        # initialize tp_range
        if args.disable_vtp:
            self.tp_range = [1]
        else:
            self.tp_range = []
            tp_consider = args.min_tp
            while tp_consider <= args.max_tp and tp_consider * args.pp_deg <= args.world_size:
                self.tp_range.append(tp_consider)
                tp_consider *= 2
        
        # initialize zero2 and zero3 ratio
        if args.zero_with_slight_noise:
            if args.chunks == 1:
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
        else:
            if args.chunks == 1:
                self.zero2_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                self.zero3_ratio = lambda d: (1/d)
            else:
                if args.async_grad_reduce:
                    self.zero2_ratio = (lambda d: (6/8 * (1/d) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d) + 2/4))
                    self.zero3_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                else:
                    self.zero2_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                    self.zero3_ratio = lambda d: (1/d)
 
    def estimate_other_memory(self):
        args = self.args
        self.other_memory_cost = dict()
        for tp_size in self.tp_range:
            tp_other_memory_cost = [{'pytorch_context_mem': args.pytorch_context_mem} for _ in range(args.pp_deg)]

            other_layers_bsz = args.global_batch_size // args.chunks // (args.world_size // args.pp_deg // tp_size)
            
            if args.vsp:
                model_tp = 1
                sdp_size = args.world_size // args.pp_deg
            else:
                model_tp = tp_size
                sdp_size = args.world_size // args.pp_deg // tp_size
            zero_ratio = self.zero3_ratio(sdp_size) if args.embed_sdp else (self.zero2_ratio(sdp_size) if args.use_zero2_for_dp else 1.0)

            if args.pp_deg == 1:
                tp_other_memory_cost[0]['other_ms_memory'] = args.other_memory_pp_off['model_states'][model_tp] * zero_ratio
                tp_other_memory_cost[0]['other_activation_memory'] = args.other_memory_pp_off['activation'][tp_size] * other_layers_bsz
                tp_other_memory_cost[0]['other_total_memory'] = tp_other_memory_cost[0]['other_ms_memory'] + tp_other_memory_cost[0]['other_activation_memory'] + tp_other_memory_cost[0]['pytorch_context_mem']
            else:
                if args.pipeline_type == 'pipedream_flush':
                    other_layers_bsz_first = other_layers_bsz * args.pp_deg
                    other_layers_bsz_last = other_layers_bsz * 1
                else:
                    other_layers_bsz_first = other_layers_bsz_last = other_layers_bsz * args.chunks
                
                tp_other_memory_cost[0]['other_ms_memory'] = args.other_memory_pp_on['first_stage']['model_states'][model_tp] * zero_ratio
                tp_other_memory_cost[0]['other_activation_memory'] = args.other_memory_pp_on['first_stage']['activation'][tp_size] * other_layers_bsz_first
                tp_other_memory_cost[0]['other_total_memory'] = tp_other_memory_cost[0]['other_ms_memory'] + tp_other_memory_cost[0]['other_activation_memory'] + tp_other_memory_cost[0]['pytorch_context_mem']
                
                tp_other_memory_cost[-1]['other_ms_memory'] = args.other_memory_pp_on['last_stage']['model_states'][model_tp] * zero_ratio
                tp_other_memory_cost[-1]['other_activation_memory'] = args.other_memory_pp_on['last_stage']['activation'][tp_size] * other_layers_bsz_last
                tp_other_memory_cost[-1]['other_total_memory'] = tp_other_memory_cost[-1]['other_ms_memory'] + tp_other_memory_cost[-1]['other_activation_memory'] + tp_other_memory_cost[-1]['pytorch_context_mem']
            
            self.other_memory_cost[tp_size] = tp_other_memory_cost
            
    def get_other_memory_cost(self):
        return self.other_memory_cost

class AttentionMemoryCostModelOptimize:
    def __init__(self, logger: Logger):
        self.logger = logger

class FFNMemoryCostModelOptimize:
    def __init__(self, logger: Logger):
        self.logger = logger
        
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