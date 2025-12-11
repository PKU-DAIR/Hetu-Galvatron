from typing import List, Optional, Tuple

import torch
import os
import torch.distributed as dist
from torch.distributed.fsdp._runtime_utils import HandleTrainingState

from galvatron.core.runtime.moe.token_dispatcher import MoETokenDispatcher
from galvatron.core import get_args
from galvatron.core.runtime.moe.prefetch.async_linear_programming import submit_lp_optimization, get_lp_optimization_result
from galvatron.core.runtime.moe.fused_kernel import smart_routing_map_gpu, new_routing_map_vectorized_gpu, new_routing_map_with_gradients

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_utils import (
    get_capacity,
    maybe_move_tensor_to_cpu,
    permute,
    sort_chunks_by_idxs,
    unpermute,
    maybe_move_tensor_to_gpu,
)
from megatron.core.tensor_parallel import (
    all_to_all,
    all_to_all_with_event,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

class MoEAlltoAllSmartTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll-based token dispatcher.

    The workflow of AlltoAll token dispatcher is as follows:
    (1) preprocess(): calculate necessary metadata for communication and permute
    (2) token_permutation(): permute->A2A(EP)->AG(TP)->sort_chunk(if num_local_experts>1)
    (3) token_unpermutation(): sort_chunk(if num_local_experts>1)->RS(TP)->A2A(EP)->unpermute
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], global_expert_indices: torch.Tensor, global_expert_locations: torch.Tensor, inverse_expert_map: torch.Tensor, config: TransformerConfig, ep_group: dist.ProcessGroup = None, tp_of_ep_group: dist.ProcessGroup = None, tp_and_ep_group: dist.ProcessGroup = None,
        layer_number: int = None,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config=config, ep_group=ep_group, tp_of_ep_group=tp_of_ep_group, tp_and_ep_group=tp_and_ep_group)
        self.layer_number = layer_number
        self.iter = 0
        self.solver_iter = 0
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        self.expert_capacity = self.num_local_experts * self.ep_size
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        self.global_expert_indices = global_expert_indices
        self.global_expert_locations = global_expert_locations
        self.inverse_expert_map = inverse_expert_map
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continous"
        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = None
        # [ep_size]. Represents the number of tokens received by the current rank from
        # other EP ranks.
        self.output_splits = None
        # [tp_size]. Represents the number of tokens received by the current rank from
        # other TP ranks.
        self.output_splits_tp = None
        self.permute_idx_device = torch.device("cuda") if self.config.moe_permute_fusion else None
        input_chunk_idxs = torch.arange(
            self.expert_capacity * self.tp_size, device=self.permute_idx_device
        )
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Token drop and padding.
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
            self.moe_expert_capacity_factor = self.config.moe_expert_capacity_factor
        self.capacity = None

        # A cuda stream synchronization is needed in self.token_permutation() in some cases,
        # because there are several non-blocking DtoH data transfers called at
        # `self.cuda_dtoh_point`. The synchronization happens at `self.cuda_sync_point`, which is
        # decided based on the MoE and parallel settings. Valid points are "before_permutation_1",
        # "before_ep_alltoall", "before_permutation_2", "before_finish", and "no_sync".
        self.cuda_sync_point = "no_sync"
        self.cuda_sync_point_priority = {
            "before_permutation_1": 0,
            "before_ep_alltoall": 1,
            "before_permutation_2": 2,
            "before_finish": 3,
            "no_sync": 4,
        }
        self.cuda_dtoh_point = "before_permutation_1"
        self.cuda_dtoh_stream = torch.cuda.Stream()
        self.cuda_htod_stream = torch.cuda.Stream()

        self.shared_experts = None

        self.history_capacity = 5
        self.history_num_global_tokens_per_expert = []

        args = get_args()
        
        # Async Linear Programming Solver configuration
        self.async_lp_solver_config = {
            "enabled": os.environ.get("ENABLE_SOLVER", "0") == "1",
            "computation_config_path": getattr(args, 'moe_computation_config_path', './configs/computation_profiling_bf16_mixtral-8x7b.json'),
            "network_config_path": getattr(args, 'moe_network_config_path', './configs/network_config.json'),
            "hidden_size": getattr(args, 'hidden_size', 4096),
            "global_checkpoint": getattr(args, 'global_checkpoint', True),
            "expert_capacity_per_device": self.num_local_experts
        }
        self.async_lp_task_id = None
        self.need_to_sync = False

    def get_smart_routing(self, routing_map: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Get the smart routing map.
        Args:
            routing_map: [token_num, num_experts]
            probs: [token_num, num_experts]
        
        Returns:
            routing_map: [token_num, num_experts]
        """
        if (self.async_lp_solver_config["enabled"]):
            if hasattr(self, "nxt_dispatcher") and self.fsdp_handle._training_state == HandleTrainingState.FORWARD:
                self.nxt_dispatcher._async_lp_prefetch_logic()
    
        num_local_tokens_per_expert = routing_map.sum(dim=0).int()
        num_global_tokens_per_expert = (
            gather_from_sequence_parallel_region(
                num_local_tokens_per_expert, group=self.tp_ep_group
            )
            .reshape(self.ep_size, self.tp_size, self.num_experts)
            .transpose(0, 1)
        )
        if len(self.history_num_global_tokens_per_expert) < self.history_capacity:
            self.history_num_global_tokens_per_expert.append(num_global_tokens_per_expert.clone())
        else:
            self.history_num_global_tokens_per_expert.append(num_global_tokens_per_expert.clone())
            self.history_num_global_tokens_per_expert.pop(0)
        
        # Submit async linear programming optimization task
        if (self.async_lp_solver_config["enabled"]):
            with torch.no_grad():
                self.total_num_global_tokens_per_expert = torch.sum(torch.stack(self.history_num_global_tokens_per_expert)[:,self.tp_rank], dim=0)
            self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.cuda_dtoh_stream):
                # TODO: use MemcpyBatchAsync instead.
                self.total_num_global_tokens_per_expert = maybe_move_tensor_to_cpu(
                    self.total_num_global_tokens_per_expert, as_numpy = True, record_stream=True
                )
        
        num_global_tokens_per_expert = smart_routing_map_gpu(num_global_tokens_per_expert, self.global_expert_locations, self.num_local_experts)
        new_routing_map, new_probs = new_routing_map_with_gradients(num_global_tokens_per_expert, self.global_expert_locations, self.inverse_expert_map, routing_map, probs, self.tp_rank, self.ep_rank)
        return new_routing_map, new_probs
        

    def preprocess(self, routing_map: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token routing map for AlltoAll communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts. This method
        should not call any DtoH data copying due to performance consideration. The necessary DtoH
        copies are made on the `self.cuda_dtoh_stream` at `self.cuda_dtoh_point`.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        # if torch.cuda.current_device() == 0:
        #     print(self.global_expert_indices)
        if self.drop_and_pad:
            # Drop and pad the input to capacity.
            num_tokens = routing_map.size(0) * self.config.moe_router_topk
            self.capacity = get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts
            # [num_local_experts], number of tokens processed by each expert.
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.tp_size * self.ep_size,
                dtype=torch.long,
            )
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = torch.full(
                (self.num_experts * self.tp_size,),
                self.capacity,
                dtype=torch.long,
                device=self.permute_idx_device,
            )
            return num_tokens_per_local_expert

        # [num_experts], number of tokens assigned to each expert from the current rank's input.
        num_local_tokens_per_expert = routing_map.sum(dim=0).int()

        if self.config.moe_expert_capacity_factor is not None:
            # Drop tokens to capacity, no padding.
            self.num_out_tokens = num_local_tokens_per_expert.sum()

            # A synchronization is needed before the first permutation
            # to get the `num_out_tokens` CPU value.
            self._maybe_update_cuda_sync_point("before_permutation_1")
        else:
            # Dropless
            self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk

        if self.ep_size > 1 or self.tp_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall/allgather in variable size.
            # ===================================================
            # [ep_size]. Represents the number of tokens sent by the current rank to other
            # EP ranks.
            # self.input_splits = num_local_tokens_per_expert.reshape(
            #     self.ep_size, self.num_local_experts
            # ).sum(axis=1)
            # Gather the global distribution of tokens across ranks.
            # num_global_tokens_per_expert represents the number of tokens sent to each
            # expert by all ranks.
            # [tp_size, ep_size, num_experts]
            num_global_tokens_per_expert = (
                gather_from_sequence_parallel_region(
                    num_local_tokens_per_expert, group=self.tp_ep_group
                )
                .reshape(self.ep_size, self.tp_size, self.ep_size * self.num_local_experts) # self.num_experts)
                .transpose(0, 1)
            )
            new_routing_map, new_probs = routing_map, probs
            self.input_splits = num_global_tokens_per_expert[self.tp_rank, self.ep_rank].reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1)
            # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()
            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
            # [tp_size, ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank
            # from other EP rank.
            self.output_splits = num_global_tokens_per_rank[self.tp_rank]
            # [tp_size, ep_size] -> [tp_size]
            # self.output_splits_tp represents the number of tokens received by the current
            # rank from other TP rank.
            self.output_splits_tp = num_global_tokens_per_rank.sum(axis=1)
            # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

            # A synchronization is needed before expert parallel AlltoAll communication
            # to get the `input_splits` and `output_splits` CPU values.
            self._maybe_update_cuda_sync_point("before_ep_alltoall")
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert

            # A synchronization is needed before the returns
            # to get the `num_tokens_per_local_expert` CPU value.
            self._maybe_update_cuda_sync_point("before_finish")

        if self.num_local_experts > 1:
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            )
            if not self.config.moe_permute_fusion:
                # A synchronization is needed before permutation 2
                # to get the `num_global_tokens_per_local_expert` CPU value.
                self._maybe_update_cuda_sync_point("before_permutation_2")

        assert (
            self.cuda_sync_point_priority[self.cuda_dtoh_point]
            <= self.cuda_sync_point_priority[self.cuda_sync_point]
        ), "cuda_sync_point must be after cuda_dtoh_point."
        return num_tokens_per_local_expert, new_routing_map, new_probs
    
    def _notify_fsdp_layer_task_id(self, task_id, stream):
        """Notify FSDP layer about async LP task ID for prefetch result retrieval"""
        self.fsdp_handle.set_lp_task_id(task_id, stream)

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        This method performs the following steps:
        1. Preprocess the routing map to get metadata for communication and permutation.
        2. Permute input tokens for AlltoAll communication.
        3. Perform expert parallel AlltoAll communication.
        4. Sort tokens by local expert (if multiple local experts exist).

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert, self.routing_map, self.probs = self.preprocess(self.routing_map, self.probs)

        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_1", tokens_per_expert
        )
        self.hidden_shape_before_permute = hidden_states.shape
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            self.routing_map,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Perform expert parallel AlltoAll communication
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", tokens_per_expert
        )
        if hasattr(self, "nxt_dispatcher") and self.nxt_dispatcher is not None:
            # global_input_tokens = all_to_all(
            #     self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
            # )
            global_input_tokens = all_to_all_with_event(
                self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits,
                self.nxt_dispatcher.forward_event
            )
        else:
            global_input_tokens = all_to_all(
                self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
            )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

        if self.tp_size > 1:
            if self.output_splits_tp is None:
                output_split_sizes = None
            else:
                output_split_sizes = self.output_splits_tp.tolist()
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens, group=self.tp_group, output_split_sizes=output_split_sizes
            )

        # Permutation 2: Sort tokens by local expert.
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_2", tokens_per_expert
        )
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                global_input_tokens = (
                    global_input_tokens.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_tokens.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                global_input_tokens = sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    fused=self.config.moe_permute_fusion,
                )

        tokens_per_expert = self._maybe_dtoh_and_synchronize("before_finish", tokens_per_expert)

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        This method performs the following steps:
        1. Unsort tokens by local expert (if multiple local experts exist).
        2. Perform expert parallel AlltoAll communication to restore the original order.
        3. Unpermute tokens to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Unpermutation 2: Unsort tokens by local expert.
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                hidden_states = (
                    hidden_states.view(
                        self.num_local_experts,
                        self.tp_size * self.ep_size,
                        self.capacity,
                        *hidden_states.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                hidden_states = sort_chunks_by_idxs(
                    hidden_states,
                    self.num_global_tokens_per_local_expert.T.ravel(),
                    self.restore_output_by_local_experts,
                    fused=self.config.moe_permute_fusion,
                )

        if self.tp_size > 1:
            if self.output_splits_tp is None:
                input_split_sizes = None
            else:
                input_split_sizes = self.output_splits_tp.tolist()
            # The precision of TP reduce_scatter should be the same as the router_dtype
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.probs.dtype),
                group=self.tp_group,
                input_split_sizes=input_split_sizes,
            ).to(hidden_states.dtype)

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if hasattr(self, "pre_dispatcher") and self.pre_dispatcher is not None:
            # permutated_local_input_tokens = all_to_all(
            #     self.ep_group, hidden_states, self.input_splits, self.output_splits
            # )
            permutated_local_input_tokens = all_to_all_with_event(
                self.ep_group, hidden_states, self.input_splits, self.output_splits,
                None, self.pre_dispatcher.backward_event
            )
        else:
            permutated_local_input_tokens = all_to_all(
                self.ep_group, hidden_states, self.input_splits, self.output_splits
            )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            probs=self.probs,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        # Add shared experts output
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts.get_output()
            output += shared_expert_output

        return output, None
    
    def sync_lp_solver(self):
        """
        Sync LP solver after forward pass.
        """
        if (self.async_lp_solver_config["enabled"]):
            self.cuda_dtoh_stream.synchronize()
            self.solver_iter += 1
            self.async_lp_task_id = submit_lp_optimization(
                history_data=self.total_num_global_tokens_per_expert,
                layer_number=self.layer_number,
                computation_config_path=self.async_lp_solver_config["computation_config_path"],
                network_config_path=self.async_lp_solver_config["network_config_path"],
                expert_capacity_per_device=self.async_lp_solver_config["expert_capacity_per_device"],
                hidden_size=self.async_lp_solver_config["hidden_size"],
                global_checkpoint=self.async_lp_solver_config["global_checkpoint"],
                solver_iter=self.solver_iter,
                global_expert_indices_numpy=self.global_expert_indices_numpy,
            )

            if hasattr(self, "nxt_dispatcher") and self.fsdp_handle._training_state == HandleTrainingState.FORWARD:
                self.nxt_dispatcher.sync_htod()
            # Pass task ID to FSDP layer for prefetch result retrieval
            # self._notify_fsdp_layer_task_id(self.async_lp_task_id, self.cuda_htod_stream)

    def _maybe_update_cuda_sync_point(self, point: str):
        """
        Update the CUDA sync point if the priority of the new point is higher than the current
        sync point, which means the new point is reached earlier than the current sync point.
        """
        if (
            self.cuda_sync_point_priority[point]
            < self.cuda_sync_point_priority[self.cuda_sync_point]
        ):
            self.cuda_sync_point = point

    def _maybe_dtoh_and_synchronize(
        self, point: str, tokens_per_expert: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Move all possible GPU tensors to CPU and make a synchronization at the expected point.
        """
        if not self.drop_and_pad:
            if point == self.cuda_dtoh_point:
                # Move all possible GPU tensors to CPU at self.cuda_dtoh_point.
                on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
                if on_side_stream:
                    self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.cuda_dtoh_stream):
                    # TODO: use MemcpyBatchAsync instead.
                    tokens_per_expert = maybe_move_tensor_to_cpu(
                        tokens_per_expert, record_stream=on_side_stream
                    )
                    self.input_splits = maybe_move_tensor_to_cpu(
                        self.input_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits = maybe_move_tensor_to_cpu(
                        self.output_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits_tp = maybe_move_tensor_to_cpu(
                        self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                    )
                    self.num_out_tokens = maybe_move_tensor_to_cpu(
                        self.num_out_tokens, record_stream=on_side_stream
                    )
                    if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                        self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                        )

            if point == self.cuda_sync_point:
                # Synchronize with the dtoh stream at self.cuda_sync_point.
                self.cuda_dtoh_stream.synchronize()

        return tokens_per_expert
    def _async_lp_prefetch_logic(self):
        """Async linear programming prefetch logic - get results from pending tasks"""
        if self.async_lp_task_id is not None:
            result = get_lp_optimization_result(self.async_lp_task_id, timeout=1.0)
            if result is not None:
                self._process_lp_result(result)
                self.async_lp_task_id = None

    def _process_lp_result(self, result):
        """Process linear programming optimization result"""
        if result.get("status") == "success":
            # Here optimization results can be applied to routing strategy
            # Actual implementation needs to be adjusted based on specific MoE router interface
            # print(f"Linear programming optimization result: {result}")
            self.global_expert_indices_numpy = result.get("expert_placement_numpy")
            self.global_expert_indices = result.get("expert_placement")
            self.global_expert_locations = result.get("global_expert_locations")
            self.inverse_expert_map = result.get("inverse_expert_map")
            self.fsdp_handle.global_placement_cpu = self.global_expert_indices
            # self.fsdp_handle.global_expert_locations_cpu = self.global_expert_locations
            # self.fsdp_handle.inverse_expert_map_cpu = self.inverse_expert_map
            # self.cuda_htod_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.cuda_htod_stream):
                # TODO: use MemcpyBatchAsync instead.
                self.global_expert_indices = maybe_move_tensor_to_gpu(
                    self.global_expert_indices, torch.cuda.current_device(), True
                )
                self.global_expert_locations = maybe_move_tensor_to_gpu(
                    self.global_expert_locations, torch.cuda.current_device(), True
                )
                self.inverse_expert_map = maybe_move_tensor_to_gpu(
                    self.inverse_expert_map, torch.cuda.current_device(), True
                )
            self.need_to_sync = True
        else:
            assert False, f"Linear programming optimization failed {result}"
    
    def sync_htod(self):
        if self.need_to_sync:
            self.cuda_htod_stream.synchronize()
            self.fsdp_handle.global_placement = self.global_expert_indices
            self.fsdp_handle.global_expert_locations = self.global_expert_locations
            self.fsdp_handle.inverse_expert_map = self.inverse_expert_map
            self.need_to_sync = False