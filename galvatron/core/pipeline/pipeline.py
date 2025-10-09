import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Tuple, List
import operator
import copy
from torch import Tensor
from galvatron.core import wrap_modules_data_parallel, wrap_modules_checkpoint, get_args
import functools

version_str = torch.__version__
version_major, version_minor, _ = version_str.split('.')
version_major, version_minor = int(version_major), int(version_minor)
# if version_major > 1:
#     if version_minor > 0:
#        from torch.distributed.fsdp._runtime_utils import _register_post_backward_hook

#     else:
#        from torch.distributed.fsdp._runtime_utils import _register_post_backward_hooks
# else:
#     assert False, f"PyTorch version must be greater than 2.0, but found {torch.__version__}"

from .utils import *
from .grad_reduce import *
from .grad_reduce import _send_backward_hook, _allreduce_word_embedding_grads, _allreduce_word_embedding_grads_no_pipeline, _allreduce_word_embedding, _allreduce_word_embedding_no_pipeline

Shape = Union[List[int], torch.Size]

def forward_step_function(loss_func,**kwargs):
    def forward_step(inputs, model):
        if isinstance(inputs, (Tuple, List)):
            outputs = model(*inputs,**kwargs)
        else:
            outputs = model(inputs,**kwargs)
        return outputs, loss_func
    return forward_step

class PipelineParallel(nn.Module):
    def __init__(
        self, 
        model, 
        model_ranks, 
        layer_output_tensor_shapes,
        layer_output_tensor_dtypes=None,
        layer_dp_sizes=None,
        layer_tp_sizes=None,
        layer_sp_sizes=None,
        chunks = 1,
        process_group=None,
        embedding_group=None,
        nproc_per_node=None,
        require_loss=True,
        info = False,
        # async_grad_reduce=True,
        tied_wte_attr_names=None):
        super().__init__()
        self.total_model_len = len(model)
        assert(len(model) == len(model_ranks))
        assert(len(model) == len(layer_output_tensor_shapes))
        layer_output_tensor_dtypes = self.get_default_tensor_dtype(layer_output_tensor_shapes) if layer_output_tensor_dtypes is None else layer_output_tensor_dtypes
        self.check_tensor_dtype(layer_output_tensor_shapes, layer_output_tensor_dtypes)

        if layer_dp_sizes is None:
            layer_dp_sizes = [1] * len(model)
        if layer_tp_sizes is None:
            layer_tp_sizes = [1] * len(model)
        if layer_sp_sizes is None:
            layer_sp_sizes = [1] * len(model)
        assert(len(model) == len(layer_dp_sizes))
        self.world_size = torch.distributed.get_world_size()
        self.global_rank = torch.distributed.get_rank()
        self.device_count = nproc_per_node if nproc_per_node is not None and nproc_per_node <= torch.cuda.device_count() else torch.cuda.device_count()
        self.local_rank = self.global_rank % self.device_count

        self.pp_global_ranks = [i for i in range(self.world_size)] if process_group is None else sorted(list(set(list(process_group))))
        assert(self.global_rank in self.pp_global_ranks)
        self.group = torch.distributed.new_group(process_group)
        self.group_size = torch.distributed.get_world_size(self.group)
        self.group_rank = torch.distributed.get_rank(self.group)
        assert(len(list(set(model_ranks)))==self.group_size and np.max(model_ranks) == self.group_size-1 and np.min(model_ranks) == 0)

        self.stage_start_idx, cnt = model_ranks.index(self.group_rank), model_ranks.count(self.group_rank)
        self.stage_end_idx = self.stage_start_idx + cnt
        self.model_cur_stage = model[self.stage_start_idx:self.stage_end_idx]#.cuda(self.local_rank)
        self.chunks = int(chunks)
        assert(self.chunks >= 1)
        self.template_stage_input_tensor_shape = [None] if self.is_pipeline_first_stage() else layer_output_tensor_shapes[self.stage_start_idx - 1]
        self.template_stage_output_tensor_shape = [None] if self.is_pipeline_last_stage() else layer_output_tensor_shapes[self.stage_end_idx - 1]
        self.stage_input_tensor_dtype = [None] if self.is_pipeline_first_stage() else layer_output_tensor_dtypes[self.stage_start_idx - 1]
        self.stage_output_tensor_dtype = [None] if self.is_pipeline_last_stage() else layer_output_tensor_dtypes[self.stage_end_idx - 1]
        self.dp_size_prev_stage = None if self.is_pipeline_first_stage() else layer_dp_sizes[self.stage_start_idx - 1]
        self.dp_size_cur_stage = None if self.is_pipeline_last_stage() else layer_dp_sizes[self.stage_end_idx - 1]

        self.tp_size_prev_stage = None if self.is_pipeline_first_stage() else layer_tp_sizes[self.stage_start_idx - 1]
        self.tp_size_cur_stage = None if self.is_pipeline_last_stage() else layer_tp_sizes[self.stage_end_idx - 1]

        self.sp_size_prev_stage = None if self.is_pipeline_first_stage() else layer_sp_sizes[self.stage_start_idx - 1]
        self.sp_size_cur_stage = None if self.is_pipeline_last_stage() else layer_sp_sizes[self.stage_end_idx - 1]

        self.dp_size_input = layer_dp_sizes[0]
        self.info = info
        self.chunk_warning = True
        
        self.checkpoint_flags_stage = [0] * (self.stage_end_idx-self.stage_start_idx) # checkpoint default off
        self.require_loss = require_loss
        
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.shape_order = args.shape_order
        self.async_grad_reduce = args.async_grad_reduce
        # if not self.async_grad_reduce and self.group_size > 1:
        #     assert Fasle, "No async grad reduce only support pp = 1"
        # assert async_grad_reduce # Remove support for async_grad_reduce=False, which is the old version for gradient synchronization
        
        self.embedding_group = embedding_group
        self.tied_wte_attr_names = tied_wte_attr_names
        self.finalize_wte_grads = tied_wte_attr_names is not None #  and self.total_model_len > len(self.model_cur_stage)

    def check_tensor_dtype(self, layer_output_tensor_shapes, layer_output_tensor_dtypes):
        assert(len(layer_output_tensor_shapes) == len(layer_output_tensor_dtypes))
        for i in range(len(layer_output_tensor_shapes)):
            if layer_output_tensor_shapes[i] is not None:
                assert(len(layer_output_tensor_shapes[i]) == len(layer_output_tensor_dtypes[i]))

    def get_default_tensor_dtype(self, layer_output_tensor_shapes):
        layer_output_tensor_dtypes = []
        for tensor_shape in layer_output_tensor_shapes:
            if tensor_shape is None:
                layer_output_tensor_dtypes.append(None)
            else:
                layer_output_tensor_dtypes.append([torch.float] * len(tensor_shape))
        return layer_output_tensor_dtypes

    def wrap_pipeline_modules_data_parallel(self, dp_types, dp_groups, module_types, mixed_precision=torch.bfloat16, wrap_block_name=None, wrap_other_block_name=None, tp_groups=None, all_block_name=None, load_module_func=None):
        assert(self.total_model_len == len(dp_types))
        assert(self.total_model_len == len(dp_groups))
        assert(self.total_model_len == len(module_types))
        dp_types_cur_stage = dp_types[self.stage_start_idx:self.stage_end_idx]
        module_types_cur_stage = module_types[self.stage_start_idx:self.stage_end_idx]
        dp_groups_cur_stage = dp_groups[self.stage_start_idx:self.stage_end_idx]
        pp_devices_cur_stage = [self.local_rank]*(self.stage_end_idx-self.stage_start_idx)
        tp_groups_cur_stage = tp_groups[self.stage_start_idx:self.stage_end_idx]
        default_process_group = dp_groups[0]
        self.model_cur_stage = wrap_modules_data_parallel(
            module_list=self.model_cur_stage,
            dp_types=dp_types_cur_stage,
            dp_groups=dp_groups_cur_stage,
            module_types=module_types_cur_stage,
            pp_devices=pp_devices_cur_stage,
            mixed_precision=mixed_precision,
            default_process_group=default_process_group,
            wrap_block_name=wrap_block_name,
            wrap_other_block_name=wrap_other_block_name,
            tp_groups=tp_groups_cur_stage,
            all_block_name=all_block_name,
            load_module_func=load_module_func,
        )
        
        if self.finalize_wte_grads:
            self.sync_embedding()

    def wrap_pipeline_modules_checkpoint(self, checkpoint_flags, wrap_block_name=None):
        self.checkpoint_flags_stage = checkpoint_flags[self.stage_start_idx:self.stage_end_idx]
        if np.sum(checkpoint_flags) > 0:
            assert(self.total_model_len == len(checkpoint_flags))
            self.model_cur_stage = wrap_modules_checkpoint(self.model_cur_stage, self.checkpoint_flags_stage, wrap_block_name=wrap_block_name)
            if wrap_block_name is not None: # in this way, checkpoint will be warpped inside FSDP
                self.checkpoint_flags_stage = [0] * (self.stage_end_idx-self.stage_start_idx)
    
    def sync_embedding(self):
        if self.group_size == 1:
            _allreduce_word_embedding_no_pipeline(self.model_cur_stage[0], self.tied_wte_attr_names[0], self.model_cur_stage[-1], self.tied_wte_attr_names[-1])
        else:
            if self.is_pipeline_first_stage():
                _allreduce_word_embedding(self.model_cur_stage[0], self.tied_wte_attr_names[0], self.embedding_group.group)
            elif self.is_pipeline_last_stage():
                _allreduce_word_embedding(self.model_cur_stage[-1], self.tied_wte_attr_names[-1], self.embedding_group.group)
    
    def gen_sp_layernorm_info(self, layer_module_types, layer_tp_groups, ln_offset, ln_size, all_block_name):
        if self.sequence_parallel:
            self.layer_tp_groups = layer_tp_groups[self.stage_start_idx:self.stage_end_idx]
            self.ln_offset = ln_offset[self.stage_start_idx:self.stage_end_idx]
            self.ln_size = ln_size[self.stage_start_idx:self.stage_end_idx]
            idx = 0
            for block in self.model_cur_stage:
                for m in block.modules():
                    if isinstance(m, FSDP):
                        m.ln_offset = self.ln_offset[idx]
                        m.ln_size = self.ln_size[idx]
                        m.sp_group = self.layer_tp_groups[idx]
                idx += 1
    
    def set_last_batch(self, state):
        for block in self.model_cur_stage:
            for m in block.modules():
                if isinstance(m, FSDP):
                    m.last_batch = state
                        
    def update_tensor_shape(self, microbatches, dp_size_input, dp_size, tp_size, sp_size, template_tensor_shape):
        # Update tensor_shape with correct microbatch_size
        tensor_shape, tensor_shape_last = copy.deepcopy(template_tensor_shape), copy.deepcopy(template_tensor_shape)
        microbatch_size = microbatches[0][0][0].shape[0] * dp_size_input // dp_size
        microbatch_size_last = microbatches[0][-1][0].shape[0] * dp_size_input // dp_size
        if tp_size == 1:
            size = sp_size
        else:
            size = tp_size
        for i in range(len(tensor_shape)):
            for j in range(len(tensor_shape[i])):
                if tensor_shape[i][j] == -1:
                    tensor_shape[i][j] = microbatch_size
            if i == 0:
                if self.sequence_parallel:
                    if self.shape_order == "SBH":
                        tensor_shape[i][0] = tensor_shape[i][0] // size
                    else:
                        tensor_shape[i] = [tensor_shape[i][0] * tensor_shape[i][1] // size, tensor_shape[i][2]]
            for j in range(len(tensor_shape_last[i])):
                if tensor_shape_last[i][j] == -1:
                    tensor_shape_last[i][j] = microbatch_size_last
            if i == 0:
                if self.sequence_parallel:
                    if self.shape_order == "SBH":
                        tensor_shape_last[i][0] = tensor_shape_last[i][0] // size
                    else:
                        tensor_shape_last[i] = [tensor_shape_last[i][0] * tensor_shape_last[i][1] // size, tensor_shape_last[i][2]]
        return tensor_shape, tensor_shape_last

    def no_pipeline_forward_backward(
        self,
        batch, 
        loss_func, 
        forward_only=False,
        profiler=None,
        iter=0,
        **kwargs,
        ):
        """Run no pipeline method.
        
        Returns dictionary with losses.
        """
        model = self.model_cur_stage
        
        # forward_step_func = forward_step_function(loss_func,**kwargs)
        # Chunk input batch into microbatches
        if batch[0][0].shape[0] % self.chunks != 0:
            if self.global_rank == 0:
                print("[Warning]The global batch size is not divisible by chunks, the results may be skewed.")
        micro_kwargs = chunk_dict(kwargs, self.chunks)
        microbatches = [chunk_batch(batch[0], self.chunks), chunk_batch(batch[1], self.chunks)]
        self.real_chunks = len(microbatches[0])
        if self.chunks != self.real_chunks and self.chunk_warning:
            if self.global_rank == 0:
                print('\nWarning from PipelineParallel Module: Real chunks is %d !'%self.real_chunks, 'Microbatch sizes is', [m[0][0].shape[0] for m in microbatches])
                print()
                self.chunk_warning = False
        
        num_microbatches = self.real_chunks
        if num_microbatches > 1 and self.async_grad_reduce:
            enter_no_sync_context(model)
        
        losses_reduced = []

        self.set_last_batch(False)
        
        for i in range(num_microbatches):
            if i == num_microbatches - 1:
                self.set_last_batch(True)
            cur_microbatch = [microbatches[0][i], microbatches[1][i]]
            output_tensor = self.forward_step(
                forward_step_function(loss_func,**micro_kwargs[i]),
                # forward_step_func,
                cur_microbatch,
                model,
                None,
                losses_reduced,
            )
            
            if profiler is not None and i == num_microbatches - 1:
                profiler.profile_memory(iter,"After Forward")
            
            if forward_only:
                continue
            input_tensor_grad = self.backward_step(
                    None,
                    output_tensor,
                    None,
                )
            
        if forward_only:
            for m in model.modules():
                if isinstance(m,FSDP) and m._is_root:
                    m._exec_order_data.next_iter()
            return losses_reduced
        
        if num_microbatches > 1 and self.async_grad_reduce:
            exit_no_sync_context(model)
            fsdp_reduce_gradients(model)
            
        if self.finalize_wte_grads:
            torch.distributed.barrier()
            self.finalize_wte_grads_func()

        return losses_reduced
    
    def pipedream_flush_forward_backward(
        self,
        batch, 
        loss_func, 
        forward_only=False,
        **kwargs,
        ):
        """Run non-interleaved 1F1B schedule, with communication between pipeline
        stages.

        Returns dictionary with losses if the last stage, empty dict otherwise."""
        assert(self.group_size > 1)
        model = self.model_cur_stage
        
        # forward_step_func = forward_step_function(loss_func,**kwargs)
        micro_kwargs = chunk_dict(kwargs, self.chunks)
        # Chunk input batch into microbatches
        microbatches = [chunk_batch(batch[0], self.chunks), chunk_batch(batch[1], self.chunks)]
        self.real_chunks = len(microbatches[0])
        if self.chunks != self.real_chunks and self.chunk_warning:
            if self.global_rank == 0:
                print('\nWarning from PipelineParallel Module: Real chunks is %d !'%self.real_chunks, 'Microbatch sizes is', [m[0][0].shape[0] for m in microbatches])
                print()
                self.chunk_warning = False

        # Compute number of warmup microbatches.
        num_microbatches = self.real_chunks
        if num_microbatches > 1 and self.async_grad_reduce:
            enter_no_sync_context(model)
        num_warmup_microbatches = self.group_size - self.group_rank - 1
        num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches

        # Compute tensor shapes for all microbatches, note that the last microbatch may have different microbatch_size, thus different shape!
        batch_size = batch[0][0].shape[0] * self.dp_size_input

        # Update stage_input_tensor_shape with correct microbatch_size
        if self.is_pipeline_first_stage():
            self.stage_input_tensor_shape = self.stage_input_tensor_shape_last = [None]
        else:
            self.stage_input_tensor_shape, self.stage_input_tensor_shape_last = \
                self.update_tensor_shape(microbatches, self.dp_size_input, self.dp_size_prev_stage, self.tp_size_prev_stage, self.sp_size_prev_stage, self.template_stage_input_tensor_shape)

        # Update stage_output_tensor_shape with correct microbatch_size
        if self.is_pipeline_last_stage():
            self.stage_output_tensor_shape = self.stage_output_tensor_shape_last = [None]
        else:
            self.stage_output_tensor_shape, self.stage_output_tensor_shape_last = \
                self.update_tensor_shape(microbatches, self.dp_size_input, self.dp_size_cur_stage, self.tp_size_cur_stage, self.sp_size_cur_stage, self.template_stage_output_tensor_shape)

        # print('rank %d'%self.global_rank, self.stage_input_tensor_shape, self.stage_input_tensor_shape_last, self.stage_output_tensor_shape, self.stage_output_tensor_shape_last, self.stage_input_tensor_dtype, self.stage_output_tensor_dtype)

        input_tensors = []
        output_tensors = []
        losses_reduced = []
        fwd_num, bwd_num = 0, 0
        if self.info:
            print('rank %d'%self.global_rank, 'start warmup')
            print('rank %d'%self.global_rank, 'num_warmup_microbatches', num_warmup_microbatches)
        
        self.set_last_batch(False)
        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            recv_tensor_shapes_fwd = self.stage_input_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes_fwd = self.stage_output_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype
            input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes_fwd, dtypes=recv_tensor_dtypes)

            cur_microbatch = [microbatches[0][i], microbatches[1][i]]
            
            # if not self.async_grad_reduce:
            #     raise NotImplementedError('Already use uniform implementations for sync/async gradient reduction. Please set async_grad_reduce=True.')
            #     pre_pipeline_forward(num_microbatches, fwd_num, self.model_cur_stage)
            
            output_tensor = self.forward_step(
                forward_step_function(loss_func,**micro_kwargs[i]),
                # forward_step_func,
                cur_microbatch,
                model,
                input_tensor,
                losses_reduced,
            )
            
            # if not self.async_grad_reduce:
            #     raise NotImplementedError('Already use uniform implementations for sync/async gradient reduction. Please set async_grad_reduce=True.')
            #     post_pipeline_forward(num_microbatches, fwd_num, self.model_cur_stage, self.checkpoint_flags_stage)
            
            fwd_num += 1
            self.send_forward_multi(output_tensor, tensor_shapes=send_tensor_shapes_fwd, dtypes=send_tensor_dtypes)

            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

        if self.info:
            print('rank %d'%self.global_rank, 'finish warmup')

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            recv_tensor_shapes_fwd = self.stage_input_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes_fwd, dtypes=recv_tensor_dtypes)

        if self.info:
            print('rank %d'%self.global_rank, 'start 1f1b')
            print('rank %d'%self.global_rank, 'num_microbatches_remaining', num_microbatches_remaining)
        
        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            recv_tensor_shapes_fwd = self.stage_input_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes_fwd = self.stage_output_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_shapes_bwd = self.stage_input_tensor_shape_last if bwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes_bwd = self.stage_output_tensor_shape_last if bwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype
            last_iteration = (i == (num_microbatches_remaining - 1))
            cur_microbatch = [microbatches[0][i + num_warmup_microbatches], microbatches[1][i + num_warmup_microbatches]]
            # if not self.async_grad_reduce:
            #     raise NotImplementedError('Already use uniform implementations for sync/async gradient reduction. Please set async_grad_reduce=True.')
            #     pre_pipeline_forward(num_microbatches, fwd_num, self.model_cur_stage)

            output_tensor = self.forward_step(
                # forward_step_func,
                forward_step_function(loss_func,**micro_kwargs[i + num_warmup_microbatches]),
                cur_microbatch,
                model,
                input_tensor,
                losses_reduced,
            )
            
            # if not self.async_grad_reduce:
            #     raise NotImplementedError('Already use uniform implementations for sync/async gradient reduction. Please set async_grad_reduce=True.')
            #     post_pipeline_forward(num_microbatches, fwd_num, self.model_cur_stage, self.checkpoint_flags_stage)
            
            fwd_num += 1
            if forward_only:
                self.send_forward_multi(output_tensor, tensor_shapes=send_tensor_shapes_fwd, dtypes=send_tensor_dtypes)

                if not last_iteration:
                    input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes_fwd, dtypes=recv_tensor_dtypes)
            else:
                output_tensor_grad = self.send_forward_recv_backward_multi(output_tensor, tensor_shapes=send_tensor_shapes_bwd, dtypes=send_tensor_dtypes, tensor_shapes_send=send_tensor_shapes_fwd)
                recv_tensor_shapes_fwd = self.stage_input_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
                send_tensor_shapes_fwd = self.stage_output_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
                # # if send and recv is executed sequentially, dead lock will be caused!
                # self.send_forward_multi(output_tensor, tensor_shapes=send_tensor_shapes_fwd)
                # output_tensor_grad = self.recv_backward_multi(tensor_shapes=send_tensor_shapes_bwd)

                # Add input_tensor and output_tensor to end of list, then pop from the
                # start of the list for backward pass.
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

                # Pop input_tensor and output_tensor from the start of the list for the backward pass.
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                # if not self.async_grad_reduce:
                #     raise NotImplementedError('Already use uniform implementations for sync/async gradient reduction. Please set async_grad_reduce=True.')
                #     pre_pipeline_backward(num_microbatches, bwd_num, self.model_cur_stage, self.checkpoint_flags_stage)

                # Add to unshard param in backward (for zero3 with no sync context)
                if num_microbatches > 1:
                    if version_major > 1:
                        if version_minor > 0:
                            for m in model.modules():
                                if isinstance(m, FSDP):
                                    if hasattr(m, "_handle"):
                                        if m._handle != None:
                                            m._handle._needs_pre_backward_unshard = True
                    
                input_tensor_grad = self.backward_step(
                    input_tensor,
                    output_tensor,
                    output_tensor_grad,
                    # recv_tensor_shapes_bwd,
                    # recv_tensor_dtypes,
                    # recv_tensor_shapes_fwd,
                    # last_iteration
                )
                bwd_num += 1

                if last_iteration:
                    input_tensor = None
                    self.send_backward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes_bwd, dtypes=recv_tensor_dtypes)
                else:
                    input_tensor = self.send_backward_recv_forward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes_fwd, dtypes=recv_tensor_dtypes, tensor_shapes_send=recv_tensor_shapes_bwd)
                    # # if send and recv is executed sequentially, dead lock will be caused!
                    # self.send_backward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes_bwd)
                    # input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes_fwd)

        if self.info:
            print('rank %d'%self.global_rank, 'finish 1f1b')

        if self.info:
            print('rank %d'%self.global_rank, 'start cooldown')
            print('rank %d'%self.global_rank, 'num_warmup_microbatches', num_warmup_microbatches)
        # Run cooldown backward passes.
        if not forward_only:
            for i in range(num_warmup_microbatches):
                if i == num_warmup_microbatches - 1:
                    self.set_last_batch(True)
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                recv_tensor_shapes_bwd = self.stage_input_tensor_shape_last if bwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
                send_tensor_shapes_bwd = self.stage_output_tensor_shape_last if bwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
                recv_tensor_dtypes = self.stage_input_tensor_dtype
                send_tensor_dtypes = self.stage_output_tensor_dtype
                
                output_tensor_grad = self.recv_backward_multi(tensor_shapes=send_tensor_shapes_bwd, dtypes=send_tensor_dtypes)
                
                # if not self.async_grad_reduce:
                #     raise NotImplementedError('Already use uniform implementations for sync/async gradient reduction. Please set async_grad_reduce=True.')
                #     pre_pipeline_backward(num_microbatches, bwd_num, self.model_cur_stage, self.checkpoint_flags_stage)
                
                # Add to unshard param in backward (for zero3 with no sync context)
                if num_microbatches > 1:
                    if version_major > 1:
                        if version_minor > 0:
                            for m in model.modules():
                                if isinstance(m, FSDP):
                                    if hasattr(m, "_handle"):
                                        if m._handle != None:
                                            m._handle._needs_pre_backward_unshard = True
                    
                input_tensor_grad = self.backward_step(
                    input_tensor,
                    output_tensor,
                    output_tensor_grad,
                    # recv_tensor_shapes_bwd,
                    # recv_tensor_dtypes,
                )
                bwd_num += 1

                self.send_backward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes_bwd, dtypes=recv_tensor_dtypes)

        if self.info:
            print('rank %d'%self.global_rank, 'finish cooldown')
        
        if num_microbatches > 1 and self.async_grad_reduce:
            exit_no_sync_context(model)
            fsdp_reduce_gradients(model)
        
        if self.finalize_wte_grads and not forward_only:
            torch.distributed.barrier()
            self.finalize_wte_grads_func()
        
        return losses_reduced

    def gpipe_forward_backward(
        self,
        batch,
        loss_func, 
        forward_only=False,
        ):
        """Run gpipe schedule, with communication between pipeline stages.

        Returns dictionary with losses if the last stage, empty dict otherwise."""

        losses_reduced = self.gpipe_forward(batch, loss_func, forward_only)
        if not forward_only:
            self.gpipe_backward()
        return losses_reduced

    def gpipe_forward(
        self,
        batch,
        loss_func, 
        forward_only = False,
        **kwargs,
        ):
        assert(self.group_size > 1)
        model = self.model_cur_stage
        
        # forward_step_func = forward_step_function(loss_func,**kwargs)
        micro_kwargs = chunk_dict(kwargs, self.chunks)
        # Chunk input batch into microbatches
        microbatches = [chunk_batch(batch[0], self.chunks), chunk_batch(batch[1], self.chunks)]
        self.real_chunks = len(microbatches[0])
        if self.chunks != self.real_chunks and self.chunk_warning:
            if self.global_rank == 0:
                print('\nWarning from PipelineParallel Module: Real chunks is %d !'%self.real_chunks, 'Microbatch sizes is', [m[0].shape[0] for m in microbatches[0]])
                print()
            self.chunk_warning = False
        self.num_microbatches = self.real_chunks
        
        if self.num_microbatches > 1 and self.async_grad_reduce:
            enter_no_sync_context(model)

        # Compute tensor shapes for all microbatches, note that the last microbatch may have different microbatch_size, thus different shape!
        batch_size = batch[0][0].shape[0] * self.dp_size_input

        # Update stage_input_tensor_shape with correct microbatch_size
        if self.is_pipeline_first_stage():
            self.stage_input_tensor_shape = self.stage_input_tensor_shape_last = [None]
        else:
            self.stage_input_tensor_shape, self.stage_input_tensor_shape_last = \
                self.update_tensor_shape(microbatches, self.dp_size_input, self.dp_size_prev_stage, self.tp_size_prev_stage, self.sp_size_prev_stage, self.template_stage_input_tensor_shape)

        # Update stage_output_tensor_shape with correct microbatch_size
        if self.is_pipeline_last_stage():
            self.stage_output_tensor_shape = self.stage_output_tensor_shape_last = [None]
        else:
            self.stage_output_tensor_shape, self.stage_output_tensor_shape_last = \
                self.update_tensor_shape(microbatches, self.dp_size_input, self.dp_size_cur_stage, self.tp_size_cur_stage, self.sp_size_cur_stage, self.template_stage_output_tensor_shape)

        self.input_tensors = []
        self.output_tensors = []
        losses_reduced = []
        if self.info:
            print('rank %d'%self.global_rank, 'start forward')
        self.set_last_batch(False)
        # Run forward passes.
        for i in range(self.num_microbatches):
            recv_tensor_shapes = self.stage_input_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes = self.stage_output_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype
            input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes, dtypes=recv_tensor_dtypes)
            cur_microbatch = [microbatches[0][i], microbatches[1][i]]

            # if not self.async_grad_reduce:
            #     raise NotImplementedError('Already use uniform implementations for sync/async gradient reduction. Please set async_grad_reduce=True.')
            #     pre_pipeline_forward(self.num_microbatches, i, self.model_cur_stage)

            output_tensor = self.forward_step(
                forward_step_function(loss_func,**micro_kwargs[i]),
                cur_microbatch,
                model,
                input_tensor,
                losses_reduced,
            )
            
            # if not self.async_grad_reduce:
            #     raise NotImplementedError('Already use uniform implementations for sync/async gradient reduction. Please set async_grad_reduce=True.')
            #     post_pipeline_forward(self.num_microbatches, i, self.model_cur_stage, self.checkpoint_flags_stage)

            self.send_forward_multi(output_tensor, tensor_shapes=send_tensor_shapes, dtypes=send_tensor_dtypes)

            if not forward_only:
                self.input_tensors.append(input_tensor)
                self.output_tensors.append(output_tensor)

        if self.info:
            print('rank %d'%self.global_rank, 'finish forward')
        return losses_reduced

    def gpipe_backward(self):
        assert(self.group_size > 1)
        
        if self.info:
            print('rank %d'%self.global_rank, 'start backward')

        model = self.model_cur_stage
        # Run backward passes.
        for i in range(self.num_microbatches):
            if i == self.num_microbatches - 1:
                self.set_last_batch(True)
            # if self.group_size > 1 and self.async_grad_reduce and i == self.num_microbatches - 1:
            #     exit_no_sync_context(self.model_cur_stage)
            if version_major > 1:
                if version_minor > 0:
                    for m in model.modules():
                        if isinstance(m, FSDP):
                            if hasattr(m, "_handle"):
                                if m._handle != None:
                                    m._handle._needs_pre_backward_unshard = True
            input_tensor = self.input_tensors.pop(0)
            output_tensor = self.output_tensors.pop(0)

            recv_tensor_shapes = self.stage_input_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes = self.stage_output_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype
            output_tensor_grad = self.recv_backward_multi(tensor_shapes=send_tensor_shapes, dtypes=send_tensor_dtypes)

            # if not self.async_grad_reduce:
            #     raise NotImplementedError('Already use uniform implementations for sync/async gradient reduction. Please set async_grad_reduce=True.')
            #     pre_pipeline_backward(self.num_microbatches, i, self.model_cur_stage, self.checkpoint_flags_stage)

            input_tensor_grad = self.backward_step(
                input_tensor,
                output_tensor,
                output_tensor_grad,
                # recv_tensor_shapes,
                # recv_tensor_dtypes
            )

            self.send_backward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes, dtypes=recv_tensor_dtypes)
        if self.info:
            print('rank %d'%self.global_rank, 'finish backward')
        
        if self.num_microbatches > 1 and self.async_grad_reduce: 
            model = self.model_cur_stage
            exit_no_sync_context(model)
            fsdp_reduce_gradients(model)
        
        if self.finalize_wte_grads:
            torch.distributed.barrier()
            self.finalize_wte_grads_func()
        
    def to_list(self, tensor):
        if isinstance(tensor, list):
            return tensor
        elif isinstance(tensor, tuple):
            return list(tensor)
        else:
            return [tensor]

    # forward & backward step
    # ---------------------------------------
    def forward_step(self, forward_step_func, batch, model, input_tensor, losses_reduced, loss_stage=False):
        """Forward step for passed-in model.

        If first stage, input tensor is obtained from data_iterator, otherwise
        passed-in input_tensor is used.

        Returns output tensor."""

        input_tensor = self.to_list(input_tensor)

        for x in input_tensor:
            if x is not None and x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                x.requires_grad = True

        if input_tensor[0] is None:
            output_tensor, loss_func = forward_step_func(batch[0], model)
        else:
            output_tensor, loss_func = forward_step_func(input_tensor, model)

        if self.is_pipeline_last_stage():
            output_tensor = self.to_list(output_tensor)
            if self.require_loss:
                output_tensor, loss_reduced = loss_func(batch[1], output_tensor)
            loss = output_tensor
            if self.require_loss:
                output_tensor = loss / self.real_chunks
            losses_reduced.append(loss_reduced)
            return output_tensor

        output_tensor = self.to_list(output_tensor)
        return output_tensor

    def check_finish_backward(self, require_grad_param_num):
        self.finish_backward_param_num += 1
        return self.finish_backward_param_num == require_grad_param_num
    
    def backward_step(self, input_tensor, output_tensor, output_tensor_grad):
        """Backward step through passed-in output tensor.

        If last stage, output_tensor_grad is None, otherwise gradient of loss
        with respect to stage's output tensor.

        Returns gradient of loss with respect to input tensor (None if first
        stage)."""

        # Retain the grad on the input_tensor.
        unwrap_input_tensor_grad = not isinstance(input_tensor, list)
        if unwrap_input_tensor_grad:
            input_tensor = [input_tensor]
        input_tensor = [None if t is None or not t.requires_grad else t for t in input_tensor]
        for x in input_tensor:
            if x is not None:
                x.retain_grad()

        if not isinstance(output_tensor, list):
            output_tensor = [output_tensor]
        if not isinstance(output_tensor_grad, list):
            output_tensor_grad = [output_tensor_grad]

        # Backward pass.
        output_tensor_, output_tensor_grad_ = [], []
        for t, g in zip(output_tensor, output_tensor_grad):
            if t is not None and t.requires_grad:
                output_tensor_.append(t)
                output_tensor_grad_.append(g)
        torch.autograd.backward(output_tensor_, grad_tensors=output_tensor_grad_)

        # Collect the grad of the input_tensor.
        input_tensor_grad = [None]
        if input_tensor is not None:
            input_tensor_grad = []
            for x in input_tensor:
                input_tensor_grad.append(None if x is None else x.grad)

        return input_tensor_grad[0] if unwrap_input_tensor_grad else input_tensor_grad
    
    # def backward_step(self, input_tensor, output_tensor, output_tensor_grad, recv_tensor_shapes, recv_tensor_dtypes, recv_tensor_shapes_fwd = None, last_iteration = None):
    #     """Backward step through passed-in output tensor.

    #     If last stage, output_tensor_grad is None, otherwise gradient of loss
    #     with respect to stage's output tensor.

    #     Returns gradient of loss with respect to input tensor (None if first
    #     stage)."""

    #     # Retain the grad on the input_tensor.        
    #     unwrap_input_tensor_grad = not isinstance(input_tensor, list)
    #     if unwrap_input_tensor_grad:
    #         input_tensor = [input_tensor]
    #     input_tensor = [None if t is None or not t.requires_grad else t for t in input_tensor]
    #     require_grad_param_num = 0
    #     position = 0
    #     self.finish_backward_param_num = 0
    #     for x in input_tensor:
    #         if x is not None:
    #             require_grad_param_num += 1
    #     input_tensor_grad = [None for t in input_tensor]
    #     hook_list = []
    #     for x in input_tensor:
    #         if x is not None:
    #             x.retain_grad()
    #             h = x.register_hook(
    #                 functools.partial(_send_backward_hook, input_tensor_grad, position, 
    #                                   functools.partial(self.send_backward_multi, tensor_shapes=recv_tensor_shapes, dtypes=recv_tensor_dtypes),
    #                                   functools.partial(self.check_finish_backward,require_grad_param_num),
    #                                   functools.partial(self.send_backward_recv_forward_multi(tensor_shapes=recv_tensor_shapes_fwd, dtypes=recv_tensor_dtypes, tensor_shapes_send=recv_tensor_shapes),
    #                                   last_iteration)
    #             )
    #             hook_list.append(h)
    #             position += 1

        # if not isinstance(output_tensor, list):
        #     output_tensor = [output_tensor]
        # if not isinstance(output_tensor_grad, list):
        #     output_tensor_grad = [output_tensor_grad]

        # # Backward pass.
        # output_tensor_, output_tensor_grad_ = [], []
        # for t, g in zip(output_tensor, output_tensor_grad):
        #     if t is not None and t.requires_grad:
        #         output_tensor_.append(t)
        #         output_tensor_grad_.append(g)
        # torch.autograd.backward(output_tensor_, grad_tensors=output_tensor_grad_)

        # for h in hook_list:
        #     h.remove()
        
        # Collect the grad of the input_tensor.
        # input_tensor_grad = [None]
        # if input_tensor is not None:
        #     input_tensor_grad = []
        #     for x in input_tensor:
        #         input_tensor_grad.append(None if x is None else x.grad)

        # return input_tensor_grad[0] if unwrap_input_tensor_grad else input_tensor_grad
    
    def finalize_wte_grads_func(self):
        if self.group_size == 1:
            _allreduce_word_embedding_grads_no_pipeline(self.model_cur_stage[0], self.tied_wte_attr_names[0], self.model_cur_stage[-1], self.tied_wte_attr_names[-1])
        else:
            if self.is_pipeline_first_stage():
                _allreduce_word_embedding_grads(self.model_cur_stage[0], self.tied_wte_attr_names[0], self.embedding_group.group)
            elif self.is_pipeline_last_stage():
                _allreduce_word_embedding_grads(self.model_cur_stage[-1], self.tied_wte_attr_names[-1], self.embedding_group.group)

    # pipeline rank utils
    # ---------------------------------------
    def get_pipeline_model_parallel_first_rank(self):
        return self.pp_global_ranks[0]


    def get_pipeline_model_parallel_last_rank(self):
        last_rank_local = self.group_size - 1
        return self.pp_global_ranks[last_rank_local]


    def get_pipeline_model_parallel_next_rank(self):
        rank_in_pipeline = self.group_rank
        world_size = self.group_size
        return self.pp_global_ranks[(rank_in_pipeline + 1) % world_size]


    def get_pipeline_model_parallel_prev_rank(self):
        rank_in_pipeline = self.group_rank
        world_size = self.group_size
        return self.pp_global_ranks[(rank_in_pipeline - 1) % world_size]

    def is_pipeline_first_stage(self):
        """Return True if in the first pipeline model-parallel stage, False otherwise."""
        return self.group_rank == 0


    def is_pipeline_last_stage(self):
        """Return True if in the last pipeline model-parallel stage, False otherwise."""
        return self.group_rank == (self.group_size - 1)

    # ---------------------------------------


    # p2p communication
    # ---------------------------------------
    def _run_p2pops(
            self,
            tensor_send_prev: Union[torch.Tensor, None],
            tensor_send_next: Union[torch.Tensor, None],
            tensor_recv_prev: Union[torch.Tensor, None],
            tensor_recv_next: Union[torch.Tensor, None],
    ):
        ops = []
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_send_prev,
                self.get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(send_prev_op)
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv_prev,
                self.get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(recv_prev_op)
        if tensor_send_next is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_send_next,
                self.get_pipeline_model_parallel_next_rank(),
            )
            ops.append(send_next_op)
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv_next,
                self.get_pipeline_model_parallel_next_rank(),
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            if tensor_send_next is not None and tensor_recv_next is not None:
                reqs = torch.distributed.batch_isend_irecv([ops[0]])
                for req in reqs:
                    req.wait()
                reqs = torch.distributed.batch_isend_irecv([ops[1]])
                for req in reqs:
                    req.wait()
            elif tensor_send_prev is not None and tensor_recv_prev is not None:
                reqs = torch.distributed.batch_isend_irecv([ops[1]])
                for req in reqs:
                    req.wait()
                reqs = torch.distributed.batch_isend_irecv([ops[0]])
                for req in reqs:
                    req.wait()
            else:
                assert len(ops)==1
                reqs = torch.distributed.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

    def _communicate(
        self,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Optional[Shape] = None,
        override_scatter_gather_tensors_in_pipeline: bool = False,
        dtype_: Optional[torch.dtype] = None,
        *,
        scatter_gather_tensors_in_pipeline: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        fp32_residual_connection: bool = False,
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """Base function for communication of tensors between stages.

        dtype logic: If none of ``dtype_``, ``params_dtype``, ``fp32_residual_connection`` is specified,
        torch.float32 is used.

        See https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/arguments.py#L145-L159
        for the details of arguments of ``dtype_``, ``params_dtype``, ``fp32_residual_connection``.

        Args:
            tensor_send_next: tensor to send to next rank (no tensor sent if set to None).
            tensor_send_prev: tensor to send to prev rank (no tensor sent if set to None).
            recv_prev: boolean for whether tensor should be received from previous rank.
            recv_next: boolean for whether tensor should be received from next rank.
            tensor_shape: optional, use when the input sequence contains less tokens than the default sequence length
            override_scatter_gather_tensors_in_pipeline:
                optional, this is used when tensor_shape is provided to override scatter gather tensors
            dtype_: This is used when tensor_shape is provided and what is the type of tensor_shape

        Keyword args:
            scatter_gather_tensors_in_pipeline: Optional. If :obj:`True`, use scatter/gather to optimize communication of tensors.
            params_dtype: Optional and legacy. Defaults to torch.float. If you manually call `.half()` or `.bfloat16()` on
                your model deliberately, pass this argument.
            fp32_residual_connection: Optional. If :obj:`True`, move residual connections to fp32.

        Returns:
            tuple containing

            - tensor_recv_prev: `torch.Tensor` if `recv_prev` is :obj:`True`, `None` otherwise.
            - tensor_recv_next: `torch.Tensor` if `recv_next` is :obj:`True`, `None` otherwise.
        """
        # Create placeholder tensors for receive in forward and backward directions if needed.
        tensor_recv_prev = None
        tensor_recv_next = None
        if tensor_shape is None:
            # In megatron, `tensor_shape` is set to `(args.seq_length, args.micro_batch_size, args.hidden_size)`
            raise RuntimeError(
                "`tensor_shape` must be specified. Common `tensor_shape` is `(seq_length, micro_batch_size, hidden_size)`")
        if not override_scatter_gather_tensors_in_pipeline and scatter_gather_tensors_in_pipeline:
            tensor_chunk_shape = (reduce(operator.mul, tensor_shape, 1) // parallel_state.get_tensor_model_parallel_world_size(),)
        else:
            tensor_chunk_shape = tensor_shape

        # The dtype logic below is copied from NVIDIA/Megatron-LM repo:
        # https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/p2p_communication.py#L74-L81
        # NOTE (mkozuki): Currently NeMo is implementing APEX AMP O2 style using PyTorch. In O2 style, forcing p2p comm to
        # use FP32 will be a perf killer so that I decided to reanimate `dtype_` argument with the default value of `None`.
        # NOTE (mkozuki): In PyTorch AMP, i.e. `torch.cuda.amp.autocast` context, activation tensors can be either FP32,
        # FP16, or BF16 and there's no way to tell the dtypes of tensors on different devices in general.
        # It might be possible if we restrict model architecture.
        dtype = params_dtype or torch.float
        if fp32_residual_connection:
            dtype = torch.float
        requires_grad = True
        if dtype_ is not None:
            dtype = dtype_
            requires_grad = False

        if recv_prev:
            tensor_recv_prev = torch.empty(
                tensor_chunk_shape,
                requires_grad=requires_grad,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
        if recv_next:
            tensor_recv_next = torch.empty(
                tensor_chunk_shape,
                requires_grad=requires_grad,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )

        # Split tensor into smaller chunks if using scatter-gather optimization.
        if not override_scatter_gather_tensors_in_pipeline and scatter_gather_tensors_in_pipeline:
            if tensor_send_next is not None:
                tensor_send_next = split_tensor_into_1d_equal_chunks(tensor_send_next)

            if tensor_send_prev is not None:
                tensor_send_prev = split_tensor_into_1d_equal_chunks(tensor_send_prev)

        def p2p_type(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next):
            commtype = ''
            if tensor_send_prev is not None:
                commtype += 'send_prev '
            if tensor_send_next is not None:
                commtype += 'send_next '
            if tensor_recv_prev is not None:
                commtype += 'recv_prev '
            if tensor_recv_next is not None:
                commtype += 'recv_next '
            return commtype
            
        commtype = p2p_type(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next)
        if self.info:
            print('rank %d'%self.global_rank, 'start p2p', commtype)
        # Send tensors in both the forward and backward directions as appropriate.
        self._run_p2pops(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next)
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()

        if self.info:
            print('rank %d'%self.global_rank, 'done p2p', commtype)

        # If using scatter-gather optimization, gather smaller chunks.
        if not override_scatter_gather_tensors_in_pipeline and scatter_gather_tensors_in_pipeline:
            if recv_prev:
                tensor_recv_prev = (
                    gather_split_1d_tensor(tensor_recv_prev)
                    .view(tensor_shape)
                    .requires_grad_()
                )

            if recv_next:
                tensor_recv_next = (
                    gather_split_1d_tensor(tensor_recv_next)
                    .view(tensor_shape)
                    .requires_grad_()
                )

        return tensor_recv_prev, tensor_recv_next


    def recv_forward(
            self,
            tensor_shape: Shape,
            override_scatter_gather_tensors_in_pipeline: bool = False,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Receive tensor from previous rank in pipeline (forward receive)."""
        if self.is_pipeline_first_stage():
            return None
        input_tensor, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            override_scatter_gather_tensors_in_pipeline=override_scatter_gather_tensors_in_pipeline,
            dtype_=dtype,
        )
        return input_tensor


    def recv_backward(
            self,
            tensor_shape: Shape = None,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Receive tensor from next rank in pipeline (backward receive)."""
        if self.is_pipeline_last_stage():
            return None
        _, output_tensor_grad = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad


    def send_forward(
            self,
            output_tensor: torch.Tensor,
            override_scatter_gather_tensors_in_pipeline: bool = False,
            tensor_shape: Shape = None,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send tensor to next rank in pipeline (forward send)."""
        if self.is_pipeline_last_stage():
            return
        self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            override_scatter_gather_tensors_in_pipeline=override_scatter_gather_tensors_in_pipeline,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )


    def send_backward(
            self,
            input_tensor_grad: torch.Tensor,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send tensor to previous rank in pipeline (backward send)."""
        if self.is_pipeline_first_stage():
            return
        self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )


    def send_forward_recv_backward(
            self,
            output_tensor: torch.Tensor,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> Union[None, torch.Tensor]:
        """Batched send and recv with next rank in pipeline."""
        if self.is_pipeline_last_stage():
            return None
        _, output_tensor_grad = self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad


    def send_backward_recv_forward(
            self,
            input_tensor_grad: torch.Tensor,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> Union[None, torch.Tensor]:
        """Batched send and recv with previous rank in pipeline."""
        if self.is_pipeline_first_stage():
            return None
        input_tensor, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor


    def send_forward_recv_forward(
            self,
            output_tensor: torch.Tensor,
            recv_prev: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Batched recv from previous rank and send to next rank in pipeline."""
        input_tensor, _ = self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor


    def send_backward_recv_backward(
            self,
            input_tensor_grad: torch.Tensor,
            recv_next: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Batched recv from next rank and send to previous rank in pipeline."""
        _, output_tensor_grad = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad


    def send_forward_backward_recv_forward_backward(
            self,
            output_tensor: torch.Tensor,
            input_tensor_grad: torch.Tensor,
            recv_prev: bool,
            recv_next: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched send and recv with previous and next ranks in pipeline."""
        input_tensor, output_tensor_grad = self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor, output_tensor_grad

    # ---------------------------------------

    # p2p communication multiple tensors
    # ---------------------------------------
    def recv_forward_multi(
        self,
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        input_tensors = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                input_tensors.append(None)
            else:
                input_tensors.append(self.recv_forward(tensor_shape=tensor_shape, dtype=dtype))
                # print('recved!', input_tensors)
        return input_tensors


    def recv_backward_multi(
        self,
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        output_tensor_grads = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                output_tensor_grads.append(None)
            else:
                output_tensor_grads.append(self.recv_backward(tensor_shape=tensor_shape, dtype=dtype))
        return output_tensor_grads


    def send_forward_multi(
        self,
        output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> None:
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            output_tensor = output_tensors[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                continue
            if output_tensor is None and tensor_shape is not None:
                output_tensor = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            self.send_forward(output_tensor, tensor_shape=tensor_shape, dtype=dtype)


    def send_backward_multi(
        self,
        input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> None:
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        assert(len(tensor_shapes)==len(input_tensor_grads))
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            input_tensor_grad = input_tensor_grads[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                continue
            if input_tensor_grad is None and tensor_shape is not None:
                input_tensor_grad = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            self.send_backward(input_tensor_grad, tensor_shape=tensor_shape, dtype=dtype)


    def send_forward_recv_backward_multi(
        self,
        output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        tensor_shapes_send = None,
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        output_tensor_grads = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            output_tensor = output_tensors[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                output_tensor_grads.append(None)
                continue
            if output_tensor is None and tensor_shape is not None:
                if tensor_shapes_send is not None:
                    output_tensor = torch.zeros(tensor_shapes_send[i],dtype=dtype).cuda(self.local_rank)
                else:
                    output_tensor = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            output_tensor_grad = self.send_forward_recv_backward(output_tensor, tensor_shape=tensor_shape, dtype=dtype)
            output_tensor_grads.append(output_tensor_grad)
        return output_tensor_grads


    def send_backward_recv_forward_multi(
        self,
        input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        tensor_shapes_send = None,
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        input_tensors = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            input_tensor_grad = input_tensor_grads[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                input_tensors.append(None)
                continue
            if input_tensor_grad is None and tensor_shape is not None:
                if tensor_shapes_send is not None:
                    input_tensor_grad = torch.zeros(tensor_shapes_send[i],dtype=dtype).cuda(self.local_rank)
                else:
                    input_tensor_grad = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            input_tensor = self.send_backward_recv_forward(input_tensor_grad, tensor_shape=tensor_shape, dtype=dtype)
            input_tensors.append(input_tensor)
        return input_tensors




class PipeSequential(nn.Sequential):
    """
    Pipe variant of ``nn.Sequential`` which supports multiple inputs.
    """

    def forward(self, *inputs, **kwargs):
        for module in self:
            if isinstance(inputs, Tuple):  # type: ignore[arg-type]
                inputs = module(*inputs, **kwargs)
            else:
                # Don't expand single variables (ex: lists/Tensor)
                inputs = module(inputs, **kwargs)
        return inputs