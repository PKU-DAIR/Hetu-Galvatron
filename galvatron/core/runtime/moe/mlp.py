# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import copy
import os
import itertools
from copy import deepcopy
from functools import partial, wraps
from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.transformer_config import TransformerConfig

from ..tensor_parallel.mlp import MLP, MLPSubmodules

class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig, tp_of_ep_group: dist.ProcessGroup = None):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            @jit_fuser
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = parallel_state.get_expert_tensor_parallel_world_size(tp_of_ep_group)
        tp_rank = parallel_state.get_expert_tensor_parallel_rank(tp_of_ep_group)

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition, self.config.hidden_size, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1, config.init_method, partition_dim=1, is_expert=True
                )
                _initialize_affine_weight_gpu(
                    self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
                )

    def forward(self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        """Forward step of the GroupedMLP."""
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )

            intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output, None

class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    def __init__(
        self, 
        num_local_experts, 
        config: TransformerConfig, 
        submodules: MLPSubmodules, 
        tp_of_ep_group: dist.ProcessGroup = None,
        tp_and_ep_group: dist.ProcessGroup = None,
    ):

        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            super().__init__(config=config)
        else:
            # Local SequentialMLP can still be used here by overriding the ffn_hidden_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.ffn_hidden_size = config.moe_ffn_hidden_size
            super().__init__(config=sequential_mlp_config)

        self.add_bias = config.add_bias_linear
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()

        for _ in range(self.num_local_experts):
            expert = MLP(self.config, submodules, is_expert=True, tp_group = tp_of_ep_group, tp_and_ep_group = tp_and_ep_group)
            self.local_experts.append(expert)
        
        if os.getenv("TOKEN_COUNTS_ITER", "0") != "0":
            self.token_num_log = []
            self.global_rank = dist.get_rank()
            self.iter_num = int(os.getenv("TOKEN_COUNTS_ITER"))
            self.chunks = int(os.getenv("CHUNK"))
        else:
            self.iter_num = 0

    def _pad_tensor_for_fp8(self, hidden):
        """Padding tensor shape to multiples of 16."""
        actual_num_tokens = hidden.shape[0]
        divisor = 16
        padded_num_tokens = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if padded_num_tokens > 0:
            pad_tensor = torch.zeros(
                padded_num_tokens, hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            hidden = torch.cat((hidden, pad_tensor), dim=0)
        return hidden

    def forward(self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        """Forward step of the SequentialMLP."""
        if self.iter_num > 0:
            self.token_num_log.append(sum(tokens_per_expert))
            if len(self.token_num_log) == 2 * self.iter_num * self.chunks:
                with open(f"training_log/token_counts/{os.getenv('METHOD')}_token_counts_{os.getenv('MODEL_SIZE')}_{os.getenv('DATA')}_batch{os.getenv('BATCH_SIZE')}_seq{os.getenv('SEQUENCE_LENGTH')}_aux{os.getenv('AUX')}_layer{self.layer_number}_{self.global_rank}.txt", "w") as f:
                    for token_num in self.token_num_log:
                        f.write(str(token_num) + "\n")
            elif len(self.token_num_log) > 2 * self.iter_num * self.chunks:
                print("Finish logging token number!")
                exit(0)
        with torch.profiler.record_function("expert_computation_time"):
            if self.num_local_experts == 1:
                if self.config.fp8:
                    hidden = self._pad_tensor_for_fp8(permuted_local_hidden_states)
                    output, output_bias = self.local_experts[0](hidden)
                    output = output[: permuted_local_hidden_states.shape[0]]
                else:
                    output, output_bias = self.local_experts[0](permuted_local_hidden_states)

                return output, output_bias
            else:
                tokens_per_expert = tokens_per_expert.tolist()
                tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)

                output_local_list = []
                output_bias_list = []

                for expert, tokens in zip(self.local_experts, tokens_list):
                    if self.config.fp8:
                        hidden = self._pad_tensor_for_fp8(tokens)
                        output, output_bias = expert(hidden)
                        output = output[: tokens.shape[0]]
                    else:
                        output, output_bias = expert(tokens)
                    output_local_list.append(output)
                    if self.add_bias:
                        output_bias_list.append(output_bias.expand_as(output))

                output_local = torch.cat(output_local_list, dim=0)
                if self.add_bias:
                    output_bias_local = torch.cat(output_bias_list, dim=0)
                else:
                    output_bias_local = None

                return output_local, output_bias_local