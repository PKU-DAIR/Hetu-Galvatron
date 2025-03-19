import os
import time
import torch
import numpy as np
from galvatron.utils import save_profiled_memory, print_peak_memory, save_profiled_time, array2str, str2array, read_json_config, write_json_config
import re

class GalvatronProfiler():
    def __init__(self, args):
        self.args = args
        self.layernum_arg_names = None
        self.mem_path = None
        self.time_path = None
        self.model_name = None
        
    # =============== For Setting Galvatron Profiler ===============
    def set_profiler_dist(self, path=None, model_layer_configs=None, model_name=None, profile_ranks=None, start_iter=10, end_iter=20, rank=None):
        rank = torch.distributed.get_rank() if rank is None else rank
        if profile_ranks is None:
            world_size = torch.distributed.get_world_size()
            profile_ranks = [0, world_size-1]
        self.set_model_layer_configs(model_layer_configs)
        self.set_path(path)
        self.set_memory_profiler(rank, profile_ranks)
        exit_ = self.args.exit_after_profiling if 'exit_after_profiling' in self.args else True
        self.set_time_profiler(start_iter=start_iter, end_iter=end_iter, exit=exit_)
        self.set_model_name(model_name)
    
    def set_profiler_single(self, start_iter=10, end_iter=20):
        self.set_memory_profiler(0)
        exit_ = self.args.exit_after_profiling if 'exit_after_profiling' in self.args else True
        self.set_time_profiler(start_iter=start_iter, end_iter=end_iter, exit=exit_)
    
    def set_profiler_launcher(self, path, layernum_arg_names=None, model_name=None, layernum_listed=False):
        self.set_layernum_arg_names(layernum_arg_names, layernum_listed)
        self.set_path(path)
        self.set_model_name(model_name)
    
    def set_layernum_arg_names(self, layernum_arg_names, layernum_listed):
        self.layernum_listed = layernum_listed
        self.layernum_arg_names = layernum_arg_names
        if not self.layernum_listed:
            self.num_layertype = len(layernum_arg_names)
            self.layernum_list = [getattr(self.args, name) for name in layernum_arg_names]
        else: # zsh: A small fix for list-format layernum args like swin `depths`
            assert len(layernum_arg_names) == 1
            self.num_layertype = sum([len(getattr(self.args, name)) for name in layernum_arg_names])
            self.layernum_list = [layernum for name in layernum_arg_names for layernum in getattr(self.args, name)]
    
    def set_model_layer_configs(self, model_layer_configs):
        if model_layer_configs is None:
            return
        self.hiddensize_list = [config['hidden_size'] for config in model_layer_configs]
        self.layernum_list = [config['layer_num'] for config in model_layer_configs]
        self.seqlen_list = [config['seq_len'] for config in model_layer_configs]
    
    def set_path(self, path):
        self.path = path
        
    def set_model_name(self, name):
        self.model_name = name
        
    def memory_profiling_path(self):
        if self.mem_path is not None:
            return self.mem_path
        assert self.model_name is not None, 'Should specify the model name!'
        args = self.args
        memory_config_path = 'configs/memory_profiling_%s_%s.json'%(args.mixed_precision, self.model_name)
        self.mem_path = os.path.join(self.path, memory_config_path)
        return self.mem_path
    
    def time_profiling_path(self):
        if self.time_path is not None:
            return self.time_path
        assert self.model_name is not None, 'Should specify the model name!'
        args = self.args
        time_config_path = "configs/computation_profiling_%s_%s.json"%(args.mixed_precision, self.model_name)
        self.time_path = os.path.join(self.path, time_config_path)
        return self.time_path
    
    # =============== For Runtime Memory Profiling ===============
    def set_memory_profiler(self, rank, profile_ranks=[], max_profile_iter=5):
        self.rank = rank
        self.profile_ranks = profile_ranks if len(profile_ranks) > 0 else [rank]
        self.mem_dict = {}
        self.max_profile_iter = max_profile_iter
    
    def profile_memory(self, iter, stage=""):
        args, rank, profile_ranks, mem_dict, max_profile_iter = \
            self.args, self.rank, self.profile_ranks, self.mem_dict, self.max_profile_iter
        if args.profile and rank in profile_ranks and iter <= max_profile_iter:
            local_rank = args.local_rank if 'local_rank' in args else 0
            profile_type = args.profile_type if 'profile_type' in args else 'allocated'
            if stage == "Before Forward":
                torch.cuda.reset_peak_memory_stats(local_rank)
                _, cur_mem = print_peak_memory("\n"+stage, local_rank, profile_type)
                mem_dict['iter_%d_before_forward'%iter] = cur_mem
            elif stage == "After Forward":
                _, cur_mem = print_peak_memory(stage, local_rank, profile_type)
                mem_dict['iter_%d_after_forward'%iter] = cur_mem
            elif stage == "After Backward":
                max_mem, cur_mem = print_peak_memory(stage, local_rank, profile_type)
                mem_dict['iter_%d_after_backward'%iter] = cur_mem
                mem_dict['iter_%d_after_backward_max'%iter] = max_mem
            else:
                print_peak_memory(stage, local_rank, profile_type)
    
    def post_profile_memory(self, iter):
        args, rank, profile_ranks, mem_dict, max_profile_iter = \
            self.args, self.rank, self.profile_ranks, self.mem_dict, self.max_profile_iter
        if args.profile and iter == max_profile_iter:
            if rank in profile_ranks:
                mem_dict['model_states'] = mem_dict['iter_4_after_backward']
                if 'pipeline_type' not in args or args.pipeline_type == "gpipe":
                    mem_dict['model_states_and_activation'] = mem_dict['iter_4_after_forward']
                    mem_dict['activation'] = mem_dict['iter_4_after_forward'] - mem_dict['iter_4_before_forward']
                mem_dict['model_states_and_peak_activation'] = mem_dict['iter_4_after_backward_max']
                mem_dict['peak_activation'] = mem_dict['iter_4_after_backward_max'] - mem_dict['iter_4_after_backward']
                time.sleep(0.2*rank)
                print('[Profiled memory for rank %d]:'%rank)
                for key, val in mem_dict.items():
                    print("\t%s: %.2f MB"%(key, val))
                if 'save_profiled_memory' in args and args.save_profiled_memory:
                    assert self.layernum_list is not None
                    world_size = torch.distributed.get_world_size()
                    memory_config_path = self.memory_profiling_path()
                    save_profiled_memory(memory_config_path, args.pp_deg, args.global_tp_deg, world_size, self.layernum_list, \
                                        args.global_train_batch_size, rank, mem_dict['model_states'], mem_dict['activation'], mem_dict['peak_activation'], args.global_checkpoint)
            if 'save_profiled_memory' in args and args.save_profiled_memory:
                exit(0)
    
    # =============== For Runtime Time Profiling ===============
    def set_time_profiler(self, start_iter, end_iter, exit=False):
        self.start_iter = start_iter
        self.end_iter = end_iter
        assert end_iter > start_iter
        self.exit = exit
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.time_list = []
    
    def profile_time_start(self, iter):
        if not self.args.profile:
            return
        if iter >= self.start_iter and iter < self.end_iter:
            torch.cuda.synchronize()
            self.start.record()
        if iter == self.end_iter:
            avg_time = sum(self.time_list) / len(self.time_list)
            local_rank = self.args.local_rank if 'local_rank' in self.args else 0
            if local_rank == 0:
                print("Average iteration time is: %.4f s"%avg_time)
            args = self.args
            if 'profile_forward' in args and args.profile_forward:
                assert self.layernum_list is not None
                time_config_path = self.time_profiling_path()
                save_profiled_time(time_config_path, avg_time*1e3, args.global_train_batch_size, self.layernum_list)
            if self.exit:
                exit(0)
            else:
                self.time_list = []
                self.start_iter, self.end_iter = self.end_iter, (self.end_iter-self.start_iter+self.end_iter)
                torch.cuda.synchronize()
                self.start.record()
            
    def profile_time_end(self, iter):
        if not self.args.profile:
            return
        if iter >= self.start_iter and iter < self.end_iter:
            self.end.record()
            torch.cuda.synchronize()
            iter_time = self.start.elapsed_time(self.end)/1e3
            self.time_list.append(iter_time)
    
    def profile_time_python(self, iter):
        if not self.args.profile:
            return
        if iter == self.start_iter:
            self.total_start_time = time.time()
        elif iter == self.end_iter:
            self.total_end_time = time.time()
            avg_time = (self.total_end_time-self.total_start_time)/(self.end_iter-self.start_iter)
            print("Average iteration time is: %.4f s"%avg_time)
            args = self.args
            if args.profile_forward:
                assert self.layernum_list is not None
                time_config_path = self.time_profiling_path()
                save_profiled_time(time_config_path, avg_time, args.global_train_batch_size, self.layernum_list)
            if self.exit:
                exit(0)
            else:
                self.start_iter, self.end_iter = self.end_iter, (self.end_iter-self.start_iter+self.end_iter)
                self.total_start_time = time.time()
    
    # =============== For Launching Profiling Scripts ===============
    def launch_profiling_scripts(self):
        args = self.args
        MODEL_ARGS, PROFILE_ARGS, LAUNCH_SCRIPTS, world_size, layernum_lists = self.prepare_launch_args()
        if args.profile_type == 'memory':
            max_tp_deg = min(world_size, self.args.max_tp_deg)
            pp_deg = 1
            for checkpoint in [0, 1]:
                tp_deg = 1
                while tp_deg <= max_tp_deg:
                    if pp_deg * tp_deg <= world_size:
                        for layernum_list in layernum_lists:
                            args_ = self.get_layernum_args(layernum_list)
                            args_['pp_deg'] = pp_deg
                            args_['global_tp_deg'] = tp_deg
                            args_['global_checkpoint'] = checkpoint
                            ARGS_ = self.args2str(args_)
                            CMD = LAUNCH_SCRIPTS+MODEL_ARGS+PROFILE_ARGS+ARGS_
                            print(CMD)
                            os.system(CMD)
                    if checkpoint:
                        break
                    tp_deg *= 2
            
            for pp_deg in [2,4]:
                layernum = pp_deg
                tp_deg = 1
                while tp_deg <= max_tp_deg:
                    if pp_deg * tp_deg <= world_size:
                        args_ = self.get_layernum_args([layernum] * self.num_layertype)
                        args_['pp_deg'] = pp_deg
                        args_['global_tp_deg'] = tp_deg
                        args_['global_checkpoint'] = 0
                        ARGS_ = self.args2str(args_)
                        CMD = LAUNCH_SCRIPTS+MODEL_ARGS+PROFILE_ARGS+ARGS_
                        print(CMD)
                        os.system(CMD)
                    tp_deg *= 2
        elif args.profile_type == 'computation':
            for layernum_list in layernum_lists:
                args_ = self.get_layernum_args(layernum_list)
                args_['pp_deg'] = 1
                args_['global_tp_deg'] = 1
                args_['global_checkpoint'] = 0
                ARGS_ = self.args2str(args_)
                CMD = LAUNCH_SCRIPTS+MODEL_ARGS+PROFILE_ARGS+ARGS_
                print(CMD)
                os.system(CMD)
    
    # =============== For Processing Profiled Memory and Time ===============
    def process_profiled_data(self):
        _, _, _, world_size, layernum_lists = self.prepare_launch_args()
        args = self.args
        if args.profile_type == 'computation':
            time_config_path = self.time_profiling_path()
            config = read_json_config(time_config_path)
            key_base = self.key_format(layernum_lists[0], args.profile_batch_size)
            val_base = config[key_base]
            for idx, layernum in enumerate(layernum_lists[1:]):
                key = self.key_format(layernum, args.profile_batch_size)
                val = config[key]
                avg_time = val - val_base
                avg_time = avg_time / args.profile_batch_size / (args.layernum_max-args.layernum_min)
                write_key = 'layertype_%d'%idx
                config[write_key] = avg_time
            write_json_config(config, time_config_path)
            print('Already written processed computation time into env config file %s!\n'%(time_config_path))
        elif args.profile_type == 'memory':
            memory_config_path = self.memory_profiling_path()
            config = read_json_config(memory_config_path)
            bsz = args.profile_batch_size
            layernum_list_base = layernum_lists[0]
            layertype = len(layernum_list_base)
            layernum_lists = layernum_lists[1:]
            layernum_diff = args.layernum_max - args.layernum_min
            param_result_list, act_result_list, param_list = [dict() for _ in range(layertype)], [dict() for _ in range(layertype)], [-1]*layertype
            
            pp_deg, tp_deg = 1, 1
            while True:
                if pp_deg * tp_deg > world_size:
                    break
                print(pp_deg, tp_deg)
                strategy = '%d_%d_%d'%(pp_deg,tp_deg,world_size//pp_deg//tp_deg)
                if strategy not in config:
                    tp_deg *= 2
                    continue
                re = config[strategy]
                for l in range(layertype):
                    layernum_key_0, layernum_key_1 = layernum_list_base, layernum_lists[l]
                    param_per_layer = (re[self.key_format(layernum_key_1, bsz, 0, 'ms')] - re[self.key_format(layernum_key_0, bsz, 0, 'ms')])/layernum_diff*pp_deg/4
                    act_per_layer_per_sample = (re[self.key_format(layernum_key_1, bsz, 0, 'act')] - re[self.key_format(layernum_key_0, bsz, 0, 'act')])/layernum_diff*pp_deg/(pp_deg*tp_deg)
                    act_per_layer_per_sample *= world_size / bsz
                    if args.profile_dp_type == 'zero3':
                        param_per_layer *= world_size//pp_deg//tp_deg
                    param_result, act_result, param = param_result_list[l], act_result_list[l], param_list[l]
                    param = max(param, param_per_layer*tp_deg)
                    print(param_per_layer, act_per_layer_per_sample, param)
                    param_result[tp_deg] = param_per_layer
                    act_result[tp_deg] = act_per_layer_per_sample
                    param_result_list[l], act_result_list[l], param_list[l] = param_result, act_result, param
                tp_deg *= 2
                
            for l in range(layertype):
                print('[layertype %d:]'%l)
                param_result, act_result, param = param_result_list[l], act_result_list[l], param_list[l]
                print('param:', param)
                # print('param_dict:', param_result)
                print('act_dict:', act_result)
                
            act_dict_c_list, act_cpt_list = [dict() for _ in range(layertype)], [-1]*layertype
            pp_deg, tp_deg = 1, 1
            while True:
                if pp_deg * tp_deg > world_size:
                    break
                print(pp_deg, tp_deg)
                strategy = '%d_%d_%d_c'%(pp_deg,tp_deg,world_size//pp_deg//tp_deg)
                if strategy not in config:
                    tp_deg *= 2
                    continue
                re = config[strategy]
                for l in range(layertype):
                    layernum_key_0, layernum_key_1 = layernum_list_base, layernum_lists[l]
                    act_per_layer_per_sample = (re[self.key_format(layernum_key_1, bsz, 0, 'act')] - re[self.key_format(layernum_key_0, bsz, 0, 'act')])/layernum_diff*pp_deg/(pp_deg*tp_deg)
                    act_per_layer_per_sample *= world_size / bsz
                    print(act_per_layer_per_sample)
                    act_dict_c, act_cpt = act_dict_c_list[l], act_cpt_list[l]
                    act_cpt = max(act_cpt, act_per_layer_per_sample)
                    act_dict_c[tp_deg] = act_per_layer_per_sample
                    act_dict_c_list[l], act_cpt_list[l] = act_dict_c, act_cpt
                tp_deg *= 2

            for l in range(layertype):
                print('[layertype %d:]'%l)
                act_dict_c, act_cpt = act_dict_c_list[l], act_cpt_list[l]
                print('act_dict_c:', act_dict_c)
                print('act_cpt:', act_cpt)
                act_result_list[l]['checkpoint'] = act_cpt

            inf=1e6
            other_memory_pp_off, other_memory_pp_on_first, other_memory_pp_on_last = \
                {'model_states': inf, 'activation': inf}, {'model_states': inf, 'activation': inf}, {'model_states': inf, 'activation': inf}
            pp_deg = 1
            while True:
                if pp_deg > world_size:
                    break
                tp_deg = 1
                while True:
                    if pp_deg * tp_deg > world_size:
                        break
                    print(pp_deg, tp_deg)
                    strategy = '%d_%d_%d'%(pp_deg,tp_deg,world_size//pp_deg//tp_deg)
                    if strategy not in config:
                        tp_deg *= 2
                        continue
                    re = config[strategy]
                    if pp_deg == 1:
                        layernum_list = layernum_list_base
                        layernum = layernum_list_base[0]
                    else:
                        layernum = pp_deg
                        layernum_list = [layernum] * layertype
                    ms_cost, act_cost = [], []
                    for l in range(layertype):
                        ms_cost.append(param_result_list[l][tp_deg]*4)
                        act_cost.append(act_result_list[l][tp_deg])
                    layer_ms_costs_first = self.total_memcost(pp_deg, layernum, layertype, ms_cost, 0)
                    layer_ms_costs_last = self.total_memcost(pp_deg, layernum, layertype, ms_cost, pp_deg-1)
                    layer_act_costs_first = self.total_memcost(pp_deg, layernum, layertype, act_cost, 0)
                    layer_act_costs_last = self.total_memcost(pp_deg, layernum, layertype, act_cost, pp_deg-1)
                    other_ms_first = re[self.key_format(layernum_list, bsz, 0, 'ms')] - layer_ms_costs_first
                    if args.profile_dp_type == 'zero3':
                        other_ms_first = (re[self.key_format(layernum_list, bsz, 0, 'ms')] - layer_ms_costs_first / (world_size//pp_deg//tp_deg)) * world_size//pp_deg
                    other_ms_last = re[self.key_format(layernum_list, bsz, world_size-1, 'ms')] - layer_ms_costs_last
                    if args.profile_dp_type == 'zero3':
                        other_ms_last = (re[self.key_format(layernum_list, bsz, world_size-1, 'ms')] - layer_ms_costs_last / (world_size//pp_deg//tp_deg)) * world_size//pp_deg
                    act_peak_first = max(re[self.key_format(layernum_list, bsz, 0, 'act_peak')], re[self.key_format(layernum_list, bsz, 0, 'act')])
                    act_peak_last = max(re[self.key_format(layernum_list, bsz, world_size-1, 'act_peak')], re[self.key_format(layernum_list, bsz, world_size-1, 'act')])
                    other_act_first = act_peak_first * world_size / bsz  - layer_act_costs_first * (pp_deg*tp_deg) 
                    other_act_last = act_peak_last * world_size / bsz - layer_act_costs_last * (pp_deg*tp_deg) 
                    print(other_ms_first, other_act_first, other_ms_last, other_act_last)
                    other_ms_first = other_ms_first if other_ms_first > 0 else 0
                    other_ms_last = other_ms_last if other_ms_last > 0 else 0
                    other_act_first = other_act_first if other_act_first > 0 else 0
                    other_act_last = other_act_last if other_act_last > 0 else 0
                    if pp_deg == 1:
                        other_memory_pp_off['model_states'] = min(other_memory_pp_off['model_states'], other_ms_first)
                        other_memory_pp_off['activation'] = min(other_memory_pp_off['activation'], other_act_first)
                    else:
                        other_memory_pp_on_first['model_states'] = min(other_memory_pp_on_first['model_states'], other_ms_first)
                        other_memory_pp_on_first['activation'] = min(other_memory_pp_on_first['activation'], other_act_first / pp_deg)
                        other_memory_pp_on_last['model_states'] = min(other_memory_pp_on_last['model_states'], other_ms_last)
                        other_memory_pp_on_last['activation'] = min(other_memory_pp_on_last['activation'], other_act_last / pp_deg)
                    tp_deg *= 2
                pp_deg *=2

            # other_memory_pp_on_first['activation'] = other_memory_pp_on_last['activation'] = max(other_memory_pp_on_first['activation'], other_memory_pp_on_last['activation'])
            print('other_memory_pp_off:', other_memory_pp_off)
            print('other_memory_pp_on_first:', other_memory_pp_on_first)
            print('other_memory_pp_on_last:', other_memory_pp_on_last)

            for l in range(layertype):
                if 'layertype_%d'%l not in config.keys():
                    config['layertype_%d'%l] = dict()
                config['layertype_%d'%l]['parameter_size'] = param_list[l]
                config['layertype_%d'%l]['tp_activation_per_bsz_dict'] = act_result_list[l]
            config['other_memory_pp_off'] = other_memory_pp_off
            config['other_memory_pp_on_first'] = other_memory_pp_on_first
            config['other_memory_pp_on_last'] = other_memory_pp_on_last
            write_json_config(config, memory_config_path)


    # =============== For Launching Nccl-test for Hardware Profiling ===============
    def profile_bandwidth(self):
        args = self.args
        world_size = args.num_nodes * args.num_gpus_per_node
        hardware_config_dir = os.path.join(self.path, './hardware_configs')
        if not os.path.exists(hardware_config_dir):
            os.mkdir(hardware_config_dir)
        
        nccl_file = 'build/all_reduce_perf'
        ARGS = self.prepare_nccltest_args(nccl_file)
        hardware_config_file = 'allreduce_bandwidth_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        hardware_config_path = os.path.join(hardware_config_dir, hardware_config_file)
        allreduce_size = world_size
        while allreduce_size > 1:
            for allreduce_consec in [1, 0]:
                if world_size == allreduce_size and allreduce_consec == 0:
                    continue
                print('============= allreduce_size: %d, allreduce_consec: %d ============='%(allreduce_size, allreduce_consec))
                allreduce_groups = self.generate_allreduce_groups(world_size, allreduce_size, allreduce_consec)
                bandwidth = self.launch_nccl_test(allreduce_groups, args.num_gpus_per_node, ARGS)
                key = 'allreduce_size_%d_consec_%d'%(allreduce_size, allreduce_consec)
                self.write_config(hardware_config_path, key, bandwidth)
                print('='*70, '\n')
            allreduce_size /= 2
        
        nccl_file = 'build/sendrecv_perf'
        ARGS = self.prepare_nccltest_args(nccl_file)
        hardware_config_file = 'p2p_bandwidth_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        hardware_config_path = os.path.join(hardware_config_dir, hardware_config_file)
        pp_deg = 2
        while pp_deg <= world_size and pp_deg <= args.max_pp_deg:
            print('============= pp_size: %d ============='%(pp_deg))
            p2p_groups = self.generate_p2p_groups(world_size, pp_deg)
            bandwidth = self.launch_nccl_test(p2p_groups, args.num_gpus_per_node, ARGS)
            key = 'pp_size_%d'%pp_deg
            self.write_config(hardware_config_path, key, bandwidth)
            print('='*70, '\n')
            pp_deg *= 2
            
        os.system('rm -rf %s'%(os.path.join(self.path, 'nccl_test.log')))
    
    def write_config(self, hardware_config_path, key, bandwidth):
        config = read_json_config(hardware_config_path) if os.path.exists(hardware_config_path) else dict()
        config[key] = bandwidth
        write_json_config(config, hardware_config_path)
        print('Already written bandwidth %s into hardware config file %s!'%(key, hardware_config_path))
    
    def read_hostfile(self):
        args = self.args
        hostfile = os.path.join(self.path, args.hostfile)
        with open(hostfile, 'r') as f:
            hostnames = f.readlines()
        hostnames = [hostname.strip() for hostname in hostnames]
        return hostnames
    
    def prepare_nccltest_args(self, nccl_file='build/all_reduce_perf'):
        args = self.args
        nccl_file = os.path.join(self.path, args.nccl_test_dir, nccl_file)
        if not os.path.exists(nccl_file):
            print('Nccl test file %s does not exist!'%nccl_file)
            print('Building nccl-test...')
            if args.num_nodes == 1:
                os.system('USE_EXPORT_VARIABLE=1 MAKE_MPI=0 sh %s'%(os.path.join(self.path, 'scripts/build_nccl_test.sh')))
            else:
                os.system('USE_EXPORT_VARIABLE=1 MAKE_MPI=1 MPI_PATH=%s sh %s'%(args.mpi_path, os.path.join(self.path, 'scripts/build_nccl_test.sh')))
            print('Nccl-test built succesfully!')
        ARGS = ''
        ARGS += 'USE_EXPORT_VARIABLE=1 '
        ARGS += 'START_MB=%d '%args.start_mb
        ARGS += 'END_MB=%d '%args.end_mb
        ARGS += 'SCALE=%d '%args.scale
        ARGS += 'NCCLTEST_FILE=%s '%nccl_file
        ARGS += 'OUTPUT_TO_LOG=1 '
        return ARGS
    
    def generate_allreduce_groups(self, world_size, allreduce_size, allreduce_consec):
        allreduce_size = int(allreduce_size)
        num_allreduce_groups = int(world_size // allreduce_size)
        allreduce_groups = []
        for i in range(num_allreduce_groups):
            if allreduce_consec:
                ranks = list(range(i * allreduce_size, (i+1) * allreduce_size))
            else:
                ranks = list(range(i, world_size, num_allreduce_groups))
            allreduce_groups.append(ranks)
        return allreduce_groups
    
    def generate_p2p_groups(self, world_size, pp_size):
        pp_size = int(pp_size)
        num_pp_groups = int(world_size // pp_size)
        pp_groups = []
        for i in range(num_pp_groups):
            ranks = list(range(i, world_size, num_pp_groups))
            pp_groups.append(ranks)
        return pp_groups
    
    def launch_nccl_test(self, groups, num_gpus_per_node, ARGS):
        hostnames = self.read_hostfile()
        bandwidths = []
        for group in groups:
            print('device group:', group)
            host_ids = sorted(list(set([rank // num_gpus_per_node for rank in group])))
            group_num_nodes = len(host_ids)
            group_num_gpus_per_node = len(group) // group_num_nodes
            cuda_visible_devices = sorted(list(set([rank % num_gpus_per_node for rank in group])))
            print('num_nodes: %d, host_ids:'%group_num_nodes, host_ids, ' num_gpus_per_node: %d, cuda_visible_devices:'%group_num_gpus_per_node, cuda_visible_devices)
            hostname = ','.join([hostnames[i] for i in host_ids])
            DEVICE_ARGS = ''
            DEVICE_ARGS += 'HOSTNAMES=%s '%hostname
            DEVICE_ARGS += 'NUM_NODES=%d '%group_num_nodes
            DEVICE_ARGS += 'NUM_GPUS_PER_NODE=%d '%group_num_gpus_per_node
            DEVICE_ARGS += 'DEVICES="CUDA_VISIBLE_DEVICES=%s" '%(','.join([str(i) for i in cuda_visible_devices]))
            # print(DEVICE_ARGS+ARGS)
            os.system(DEVICE_ARGS+ARGS+'sh %s'%(os.path.join(self.path, 'scripts/run_nccl_test.sh')))
            with open('nccl_test.log', 'r') as f:
                lines = f.readlines()
            for line in lines[::-1]:
                if 'Avg bus bandwidth' in line:
                    result = line
                    bandwidth = float(line.split()[-1])
                    break
            print(result)
            bandwidths.append(bandwidth)
            if self.args.avg_or_min_or_first == 'first':
                break
        bandwidth = np.min(bandwidths) if self.args.avg_or_min_or_first == 'min'  else np.mean(bandwidths)
        print('Bandwidths:', bandwidths, 'Average bandwidth:', bandwidth)
        print()
        return bandwidth

    # =============== For Launching Scripts for Profiling Overlap Slowdown Coefficient ===============
    def profile_overlap(self):
        args = self.args
        ARGS = ''
        ARGS += 'USE_EXPORT_VARIABLE=1 '
        ARGS += 'NUM_GPUS_PER_NODE=%d '%args.num_gpus_per_node
        ARGS += 'OVERLAP_TIME_MULTIPLY=%d '%args.overlap_time_multiply
        os.system(ARGS+'sh %s'%(os.path.join(self.path, 'scripts/profile_overlap.sh')))

    # =============== Util functions ===============    
    def key_format(self, layernum, bsz, rank=None, type=None):
        if isinstance(layernum, list):
            s =  "layernum[%s]_bsz%d"%(array2str(layernum), bsz)
        else:
            s =  "layernum%d_bsz%d"%(layernum, bsz)
        if rank is not None and type is not None:
            s += '_rank%d_%s'%(rank, type)
        return s
    
    def match_key_str(self, s):
        if '[' in s and ']' in s:
            layernum = str2array(s[s.find('[')+1:s.find(']')])
            s = s[s.find(']')+2:]
            if 'rank' in s:
                pattern = r'bsz(\d+)_rank(\d+)_(\w+)'
                match = re.match(pattern, s)
                bsz = int(match.group(1))
                rank = int(match.group(2))
                type = match.group(3)
                results = [layernum, bsz, rank, type]
            else:
                pattern = r'bsz(\d+)'
                match = re.match(pattern, s)
                bsz = int(match.group(1))
                results = [layernum, bsz]
        else:
            if 'rank' in s:
                pattern = r'layernum(\d+)_bsz(\d+)_rank(\d+)_(\w+)'
                match = re.match(pattern, s)
                layernum = int(match.group(1))
                bsz = int(match.group(2))
                rank = int(match.group(3))
                type = match.group(4)
                results = [layernum, bsz, rank, type]
            else:
                pattern = r'layernum(\d+)_bsz(\d+)'
                match = re.match(pattern, s)
                layernum = int(match.group(1))
                bsz = int(match.group(2))
                results = [layernum, bsz]
        return results
    
    def total_memcost(self, pp_deg, layernum, layertype, per_layer_cost, stage_idx):
        layer_costs = []
        for l in range(layertype):
            layer_costs += [per_layer_cost[l]] * layernum
        total_layer_num = layertype * layernum
        avg_layer_num = int(total_layer_num // pp_deg)
        last_layer_num = total_layer_num - avg_layer_num * (pp_deg-1)
        pp_divide = [avg_layer_num] * (pp_deg-1) + [last_layer_num]
        return np.sum(layer_costs[int(np.sum(pp_divide[:stage_idx])):int(np.sum(pp_divide[:stage_idx+1]))])
    
    def prepare_launch_args(self):
        assert self.layernum_arg_names is not None
        profile_arg_names = ['profile_type', 
                            'set_model_config_manually',
                            'set_layernum_manually',
                            'profile_batch_size', 
                            'layernum_min', 
                            'layernum_max', 
                            'max_tp_deg', 
                            'profile_dp_type', 
                            'mixed_precision',
                            'use_flash_attn',
                            'extra_args_str']
        exclude_arg_names = profile_arg_names+self.layernum_arg_names
        MODEL_ARGS = self.args2str(self.args._get_kwargs(), exclude_arg_names)
        # print(MODEL_ARGS)
                
        profile_args = self.profiling_general_args()
        
        PROFILE_ARGS = self.args2str(profile_args)        
        # zsh: Revise to accept extra_args_str
        if "extra_args_str" in self.args:
            extra_args_list = self.args.extra_args_str.split("/")
            for arg in extra_args_list:
                if arg != "":
                    PROFILE_ARGS += f" --{arg}"
        # print(PROFILE_ARGS)
        
        env_args = self.env_args()
        LAUNCH_SCRIPTS = self.launch_scripts(env_args)
        print('Get environment args:', env_args)
        
        world_size = int(env_args['NUM_NODES']) * int(env_args['NUM_GPUS_PER_NODE'])
        
        layernum_lists = self.get_layernum_list_for_profiling()
        
        return MODEL_ARGS, PROFILE_ARGS, LAUNCH_SCRIPTS, world_size, layernum_lists
    
    def get_layernum_list_for_profiling(self):
        layernum_lists = []
        base_list = [self.args.layernum_min] * self.num_layertype
        layernum_lists.append(base_list)
        for idx in range(self.num_layertype):
            l = base_list.copy()
            l[idx] = self.args.layernum_max
            layernum_lists.append(l)
        return layernum_lists
    
    def argval2str(self, val): # zsh: Specifically deal with list-format args
        if isinstance(val, list):
            s = ""
            for i in val:
                s += f"{i} "
            return s.strip()
        else:
            return str(val)
        
    def arg2str(self, key, val):
        return ' --%s %s'%(key, self.argval2str(val))
    
    def args2str(self, args, exclude_args=[]):
        s = ""
        if isinstance(args, dict):
            for key, val in args.items():
                if key in exclude_args:
                    continue
                s += self.arg2str(key, val)
        elif isinstance(args, (list, tuple)) and len(args) > 0 and len(args[0])==2:
            for key, val in args:
                if key in exclude_args:
                    continue
                s += self.arg2str(key, val)
        return s
    
    def profiling_general_args(self):
        args = {
            'set_model_config_manually': 0,
            'set_layernum_manually': 1,
            
            'global_train_batch_size': self.args.profile_batch_size,
            'epochs': 10,
            'lr': 1e-4,
            'adam_weight_decay': 0.01,
            'dropout_prob': 0.1,
            'check_loss': 0,
            'profile': 1,
            'save_profiled_memory': 1 if self.args.profile_type == 'memory' else 0,
            'profile_forward': 1 if self.args.profile_type == 'computation' else 0,
            'initialize_on_meta': 0,
            
            'global_tp_consec': 1,
            'sdp': 1 if self.args.profile_dp_type == 'zero3' and self.args.profile_type == 'memory' else 0,
            'chunks': 1,
            'pipeline_type': 'gpipe',
            'default_dp_type': self.args.profile_dp_type if self.args.profile_type == 'memory' else 'ddp',
            'mixed_precision': self.args.mixed_precision,
        }
        
        if self.args.use_flash_attn:
            args['use-flash-attn'] = ''
        return args
    
    def get_layernum_args(self, layernum_list):
        assert len(layernum_list) == self.num_layertype
        args = {}
        if not self.layernum_listed:
            for layernum, arg_name in zip(layernum_list, self.layernum_arg_names):
                args[arg_name] = layernum
        else: # zsh: A small fix for list-format layernum args like swin `depths`
            assert len(self.layernum_arg_names) == 1
            arg_name = self.layernum_arg_names[0]
            args[arg_name] = layernum_list
        return args
    
    def env_args(self):
        return {'PROFILE_LAUNCHER': os.getenv('PROFILE_LAUNCHER', 'python3 -m torch.distributed.launch'),
                'PROFILE_TRAINER': os.getenv('PROFILE_TRAINER', 'train_dist.py'),
                'NUM_NODES': os.getenv('NUM_NODES', 1) if self.args.profile_type == 'memory' else 1,
                'NUM_GPUS_PER_NODE': os.getenv('NUM_GPUS_PER_NODE', 8) if self.args.profile_type == 'memory' else 1,
                'MASTER_ADDR': os.getenv('MASTER_ADDR', ''),
                'MASTER_PORT': os.getenv('MASTER_PORT', ''),
                'NCCL_SOCKET_IFNAME': os.getenv('NCCL_SOCKET_IFNAME', ''),
                'NODE_RANK': os.getenv('NODE_RANK', 0),
                }
        
    def launch_scripts(self, env_args):
        args = '%s'%env_args['PROFILE_LAUNCHER']
        # args += ' --nnodes=%s'%env_args['NUM_NODES']
        # args += ' --nproc_per_node=%s'%env_args['NUM_GPUS_PER_NODE']
        # args += ' --master_addr=%s'%env_args['MASTER_ADDR']
        # args += ' --master_port=%s'%env_args['MASTER_PORT']
        # args += ' --node_rank=%s'%env_args['NODE_RANK']
        args += ' %s'%env_args['PROFILE_TRAINER']
        return args

