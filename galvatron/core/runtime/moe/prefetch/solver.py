#!/usr/bin/env python3
"""
MoE Load Balancing Optimization Solver
=====================================

Objective: minimize T = 4 * Σᵢ T_comm[i] + 3 * max_i T_comp[i]
Variables: A[j,i] ∈ {0,1}, S[i,j,k] ≥ 0
Constraints: expert placement, token conservation, capacity limits

Implements: Linear Programming + Heuristic Algorithms
Supports GPU tensor output for efficient integration with CUDA kernels
"""
import numpy as np
import json
import copy
from typing import Tuple, List
import greedy_balancer as gb

class MoEOptimizer:
    """MoE communication optimization solver with multiple algorithms"""
    
    def __init__(self, computation_config_path: str, network_config_path: str, hidden_size: int, global_checkpoint: bool):
        self.load_configs(computation_config_path, network_config_path)
        self.hidden_size = hidden_size
        self.global_checkpoint = global_checkpoint
        
    def load_configs(self, computation_config_path: str, network_config_path: str):
        """Load configuration files"""
        if computation_config_path != "":
            with open(computation_config_path, 'r') as f:
                self.computation_config = json.load(f)
            self.v_comp = self._extract_computation_speed()
        if network_config_path != "":
            with open(network_config_path, 'r') as f:
                self.network_config = json.load(f)
            self.V_inter = self.network_config["inter_node"]  # GB/s
            self.V_intra = self.network_config["intra_node"]  # GB/s
        
    def _extract_computation_speed(self) -> float:
        """Extract computation speed from config"""
        base_key = "layertype_0_bsz1_seq4096_mlp"
        if base_key in self.computation_config:
            return self.computation_config[base_key] / 4096  # ms/token
        return 0.001  # default: 1ms per token
    
    def default_placement(self,
                        n_device,
                        n_expert,
                        E,
                        C_e,
                        ) -> Tuple:
        """
        Default placement method
        """
        ep_size = n_expert // C_e
        A_res = []
        for j in range(n_device):
            tmp = []
            for i in range(C_e):
                tmp.append(i * ep_size + (j % ep_size))
            A_res.append(tmp)

        return 0, 0, A_res

    def greedy_load_balancing_heuristic(self,
                                        n_device,
                                        n_expert,
                                        E,
                                        C_e, 
                                        no_even = False,
                                        no_pq = False,
                                        ) -> Tuple:
        """Greedy load balancing heuristic with expert replication"""
        if no_even or no_pq:
            return gb.greedy_load_balancing_heuristic(n_device, n_expert, E, C_e, no_even, no_pq)
        else:
            return gb.greedy_load_balancing_heuristic_complete(n_device, n_expert, E, C_e, self.hidden_size * 2, 2, self.v_comp, self.V_intra, self.V_inter, self.global_checkpoint)

    def flexmoe_method(self,
                       E: List[List[int]],
                       n_device: int,
                       n_expert: int,
                       C_e: int,
                       global_expert_indices_numpy: np.ndarray = None,
                       max_iteration: int = 10) -> Tuple:
        """
        FlexMoE method implementing dynamic scheduling and load balancing
        Based on Algorithm 1: Scheduler and Algorithm 2: MakeSchedulingPlan
        
        Args:
            E: Token demand matrix [n_device][n_expert]
            n_device: Total number of devices
            n_expert: Total number of experts
            C_e: Maximum experts per device
            default_A: np.ndarray = None,
            
        Returns:
            Tuple: (A, S, total_time) where:
                A: Expert assignment matrix
                S: Token routing matrix
                total_time: Total computation time
        """
        expert_loads = [sum(E[i][j] for i in range(n_device)) for j in range(n_expert)]
        default_A = np.zeros((n_expert, n_device))

        for i in range(n_device):
            for j in range(C_e):
                default_A[global_expert_indices_numpy[i, j], i] += 1
        
        if default_A is None:
            default_A, P = self.get_default_A(E, n_device, n_expert, C_e)
        else:
            A = copy.deepcopy(default_A)
            P = np.zeros((n_device, C_e))
            length = [0] * n_device

            for expert in range(n_expert):
                for device in range(n_device):
                    while A[expert, device] > 0:
                        P[device, length[device]] = expert
                        length[device] += 1
                        A[expert, device] -= 1

        balance_ratio, now_S, now_cost = self._calculate_balance_ratio(E, default_A, n_device, n_expert)
            
        A = default_A
        # Main optimization loop (Algorithm 1)
        while balance_ratio > 1 and max_iteration > 0:
            max_iteration -= 1
            # Generate scheduling plan (Algorithm 2)
            new_A = copy.deepcopy(A)
            new_A = self._make_scheduling_plan(expert_loads, A, now_S, n_device, n_expert, C_e)

            new_balance_ratio, new_S, new_cost = self._calculate_balance_ratio(E, new_A, n_device, n_expert)

            # print(new_cost, now_cost)

            if new_cost < now_cost or new_balance_ratio < balance_ratio:
                now_cost = new_cost
                now_S = new_S
                A = new_A
                balance_ratio = new_balance_ratio
            else:
                break

        # Generate routing matrix S
        A_res = []
        for j in range(n_device):
            tmp = []
            for i in range(n_expert):
                while abs(A[i, j]) > 1e-6:
                    tmp.append(i)
                    A[i, j] -= 1
            A_res.append(tmp)
        return 0, 0, A_res
    
    def _calculate_balance_ratio(self, E: List[List[int]], A: np.ndarray, 
                                n_device: int, n_expert: int) -> float: 
        """Calculate balance ratio (Equation 6)"""

        S = self._generate_smart_routing(n_device, n_expert, E, A)

        device_loads = S.sum(axis=(0,1))
        # Calculate balance ratio (max load / min load)
        max_load = max(device_loads)
        min_load = min(device_loads)

        cost = self._calculate_total_time(n_device, n_expert, S)
        
        return max_load / min_load, S, cost
    
    def _make_scheduling_plan(self, expert_loads: List[float], A: np.ndarray, 
                              now_S: np.ndarray, n_device: int, n_expert: int, C_e: int) -> List[Tuple]:
        """
        Make scheduling plan (Algorithm 2: MakeSchedulingPlan)
        Returns list of (operation, expert_id) tuples
        """
        expert_replicas = A.sum(axis=1)
        expert_loads = [expert_loads[i] // expert_replicas[i] for i in range(n_expert)]
        device_loads = now_S.sum(axis=(0,1))

        sorted_expert = sorted(range(n_expert), key=lambda x: expert_loads[x], reverse=True)
        argmax_expert = sorted_expert[0]
        node = n_device // 8
        idx = n_expert - 1
        while expert_replicas[sorted_expert[idx]] == node:
            idx -= 1
        argmin_expert = sorted_expert[idx]
        if argmax_expert == argmin_expert:
            return A

        new_A = copy.deepcopy(A)
        
        for i in range(node):
            argmax_device = []
            argmin_device = []
            for j in range(8):
                if new_A[argmax_expert, i * 8 + j] > 0:
                    argmax_device.append(i * 8 + j)
                if new_A[argmin_expert, i * 8 + j] > 0:
                    argmin_device.append(i * 8 + j)

            final_argmin_device = min(argmin_device, key=lambda x: device_loads[x])
            
            final_argmax_device = -1
            for device in argmax_device:
                if new_A[argmax_expert, device] < C_e:
                    final_argmax_device = device
                    break
            
            if final_argmax_device == -1:
                new_A[argmax_expert, final_argmin_device] += 1
                new_A[argmin_expert, final_argmin_device] -= 1
            elif final_argmax_device == final_argmin_device:
                new_A[argmax_expert, final_argmax_device] += 1
                new_A[argmin_expert, final_argmin_device] -= 1
            else:
                new_expert = []
                for expert in range(n_expert):
                    if expert != argmax_expert and new_A[expert, final_argmax_device] > 0:
                        new_expert.append(expert)

                new_expert = max(new_expert, key=lambda x: expert_loads[x])
                new_A[argmax_expert, final_argmax_device] += 1
                new_A[argmin_expert, final_argmin_device] -= 1
                new_A[new_expert, final_argmax_device] -= 1
                new_A[new_expert, final_argmin_device] += 1

        return new_A

    def _calculate_total_time(self, n_device: int, n_expert: int, S: np.ndarray) -> float:
        """Calculate total time (objective function)"""
        comp_times = []
        
        for i in range(n_device):
            comp_time = 0
            for k in range(n_device):
                for j in range(n_expert):
                    if S[k, j, i] > 0:
                        comp_time += S[k, j, i]
            comp_times.append(comp_time)
        
        return max(comp_times)

    def _generate_smart_routing(self, n_device: int, n_expert: int, E: List[List[int]], 
                                               A: np.ndarray) -> np.ndarray:
        """Generate robust routing strategy using pre-allocated replica capacities with smart routing logic
        Based on fused_kernel.py routing logic"""
        S = np.zeros((n_device, n_expert, n_device))
        
        gpus_per_node = 8  # Assuming 8 GPUs per node
        
        def get_node_id(device):
            return device // gpus_per_node
        
        A = copy.deepcopy(A)
        # Build expert locations for each expert
        expert_locations = {}  # expert_id -> [device_ids]
        expert_weights = {}
        for expert in range(n_expert):
            expert_locations[expert] = []
            expert_weights[expert] = []
            for device in range(n_device):
                if A[expert, device] > 0:
                    expert_locations[expert].append(device)
                    expert_weights[expert].append(A[expert, device])
        
        # Process each source device and expert
        for src_device in range(n_device):
            src_node = get_node_id(src_device)
            
            for expert in range(n_expert):
                tokens_for_expert = E[src_device][expert]
                if tokens_for_expert == 0:
                    continue
                
                if expert not in expert_locations or not expert_locations[expert]:
                    continue
                
                # Separate intra-node and inter-node locations
                intra_locations = []
                intra_weights = []
                inter_locations = []
                inter_weights = []
                
                for location, weight in zip(expert_locations[expert], expert_weights[expert]):
                    target_node = get_node_id(location)
                    if target_node == src_node:
                        intra_locations.append(location)
                        intra_weights.append(weight)
                    else:
                        inter_locations.append(location)
                        inter_weights.append(weight)
                
                remaining_tokens = tokens_for_expert
                
                # Phase 1: Intra-node routing (evenly distribute)
                if intra_locations:
                    intra_count = sum(intra_weights)
                    tokens_per_location = remaining_tokens // intra_count
                    extra_tokens = remaining_tokens % intra_count

                    for location, weight in zip(intra_locations, intra_weights):
                        tokens_to_assign = tokens_per_location
                        
                        S[src_device, expert, location] += tokens_to_assign * weight + min(weight, extra_tokens)
                        extra_tokens -= min(weight, extra_tokens)
                    
                    remaining_tokens = 0
                
                # Phase 2: Inter-node routing (evenly distribute remaining tokens)
                if remaining_tokens > 0 and inter_locations:
                    inter_count = sum(inter_weights)
                    tokens_per_location = remaining_tokens // inter_count
                    extra_tokens = remaining_tokens % inter_count
                    
                    for location, weight in zip(inter_locations, inter_weights):
                        tokens_to_assign = tokens_per_location
                        
                        S[src_device, expert, location] += tokens_to_assign * weight + min(weight, extra_tokens)
                        extra_tokens -= min(weight, extra_tokens)
        
        return S