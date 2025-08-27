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

import json
import numpy as np
import torch
import random
import time
from typing import Dict, List, Tuple, Callable, Optional, Union
from dataclasses import dataclass
import heapq
import greedy_balancer as gb
import copy

try:
    from pyscipopt import Model, quicksum
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False

class MoEOptimizer:
    """MoE communication optimization solver with multiple algorithms"""
    
    def __init__(self, computation_config_path: str, network_config_path: str):
        self.load_configs(computation_config_path, network_config_path)
        
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
    
    def bandwidth_function(self, i: int, j: int) -> float:
        """Bandwidth function: returns bandwidth between devices i and j"""
        if i == j:
            return 1e9  # infinite for self-communication
        elif self.node_func(i) != self.node_func(j):
            return self.V_inter  # inter-node
        else:
            return self.V_intra  # intra-node
    
    def node_func(self, device_id):
        return device_id // 8
    
    def linear_programming_solve(self, 
                                E: List[List[int]], 
                                n_device: int,
                                n_expert: int,
                                C_e: int, 
                                M_token: float,
                                time_limit: int = 300,
                                gap_limit: float = 0.05,
                                random_seed: int = 42) -> Tuple:
        """
        Linear programming solver using SCIP
        
        Args:
            gap_limit: Stop when relative gap <= this value (e.g., 0.05 = 5%)
        
        Returns: (A, S, total_time, model_info)
        """
        if not SCIP_AVAILABLE:
            raise RuntimeError("SCIP solver not available")
        
        model = Model("moe_optimization")
        
        # Variables
        A = {}  # A[j,i]: expert j on device i
        for j in range(n_expert):
            for i in range(n_device):
                A[j, i] = model.addVar(vtype="BINARY", name=f"A_{j}_{i}")
        
        S = {}  # S[i,j,k]: tokens from device i expert j to device k
        for i in range(n_device):
            for j in range(n_expert):
                for k in range(n_device):
                    S[i, j, k] = model.addVar(vtype="INTEGER", name=f"S_{i}_{j}_{k}")
        
        T_comm = {}
        T_comp = {}
        for i in range(n_device):
            T_comm[i] = model.addVar(vtype="CONTINUOUS", name=f"T_comm_{i}")
            T_comp[i] = model.addVar(vtype="CONTINUOUS", name=f"T_comp_{i}")
        
        T_max_comp = model.addVar(vtype="CONTINUOUS", name="T_max_comp")
        
        # Constraints
        # 1. Device expert capacity: each device can host at most C_e experts
        for i in range(n_device):
            model.addCons(quicksum(A[j, i] for j in range(n_expert)) == C_e) # TODO: modify to <= ?

        # 2. Token conservation
        for i in range(n_device):
            for j in range(n_expert):
                model.addCons(quicksum(S[i, j, k] for k in range(n_device)) == E[i][j])
        
        # 3. Expert availability
        for i in range(n_device):
            for j in range(n_expert):
                for k in range(n_device):
                    model.addCons(S[i, j, k] <= E[i][j] * A[j, k])
        
        # 4. Device token capacity: each device receives at most C_t tokens
        # for k in range(n_device):
        #     model.addCons(quicksum(S[i, j, k] for i in range(n_device) for j in range(n_expert)) <= C_t)
        
        # 5. Communication time
        for i in range(n_device):
            comm_time = quicksum(
                M_token * S[i, j, k] / self.bandwidth_function(i, k) * 1e-9
                for j in range(n_expert) 
                for k in range(n_device)
            )
            model.addCons(T_comm[i] == comm_time)
        
        # 6. Computation time
        for i in range(n_device):
            comp_time = quicksum(
                S[k, j, i] * self.v_comp / 1000
                for k in range(n_device)
                for j in range(n_expert)
            )
            model.addCons(T_comp[i] == comp_time)
        
        # 7. Max computation time
        for i in range(n_device):
            model.addCons(T_max_comp >= T_comp[i])
        
        # Objective function
        total_time = 4 * quicksum(T_comm[i] for i in range(n_device)) + 3 * T_max_comp
        model.setObjective(total_time, "minimize")
        
        # Solver settings
        model.setParam('limits/time', time_limit)
        model.setParam('limits/gap', gap_limit)      # Stop when gap <= gap_limit  
        model.setParam('display/verblevel', 1)       # Show detailed solving log
        model.setParam('display/freq', 1)            # Display frequency
        
        # Set random seed for reproducible results
        model.setParam('randomization/randomseedshift', random_seed)  # Set random seed
        model.setParam('randomization/permutationseed', random_seed)   # Set permutation seed
        model.setParam('randomization/lpseed', random_seed)           # Set LP solver seed
        
        # print(f"Linear Programming Settings:")
        # print(f"  Time limit: {time_limit}s")
        # print(f"  Gap limit: {gap_limit*100:.1f}%")
        # print("Starting optimization...")
        
        # Solve
        model.optimize()
        # print(model.getStatus())
        # Extract results
        if model.getStatus() in ["optimal", "bestsollimit", "gaplimit"]:
            # A_res = np.zeros((n_expert, n_device))
            S_res = np.zeros((n_device, n_expert, n_device))
            
            # for j in range(n_expert):
            #     for i in range(n_device):
            #         A_res[j, i] = model.getVal(A[j, i])
            
            for i in range(n_device):
                for j in range(n_expert):
                    for k in range(n_device):
                        S_res[i, j, k] = model.getVal(S[i, j, k])
            
            A_res = []
            for j in range(n_device):
                tmp = []
                for i in range(n_expert):
                    if abs(model.getVal(A[i, j])-1) < 1e-6:
                        tmp.append(i)
                A_res.append(tmp)

            max_load = np.max(np.sum(S_res, axis=(0,1)))

            t_max_comp = model.getVal(T_max_comp)
            # print(t_max_comp * 1000 / self.v_comp)
            return max_load, model.getObjVal(), A_res
        else:
            # TLE
            A_res = []
            x = 0
            for j in range(n_device):
                tmp = []
                for i in range(C_e):
                    tmp.append(x)
                    x += 1
                    if x == n_expert:
                        x = 0
                A_res.append(tmp)
            return 0, 0, A_res
    
    def default_cost(self, E, n_device, n_expert, C_e, M_token):

        placement = np.zeros((n_device, n_expert))
        expert_load = np.sum(E, axis=0)
        for i in range(n_device):
            for j in range(C_e * i, C_e * (i + 1)):
                placement[i, j] = 1

        max_load = 0
        for j in range(n_device):
            max_load = max(max_load, (sum(expert_load[k] for k in range(n_expert) if placement[j, k] == 1)))

        communication_cost = 0
        for i in range(n_device):
            for j in range(n_expert):
                if placement[i, j] == 1:
                    communication_cost += sum( M_token * E[k][j] / self.bandwidth_function(i, k) / 1e9 for k in range(n_device))
        
        return max_load, 4 * communication_cost + 3 * self.v_comp / 1000 * max_load

    def new_allocate_expert_replicas_precise(self, expert_loads: List[float], device_num: int, capacity: int) -> List[int]:
        total_capacity = device_num * capacity
        expert_num = len(expert_loads)
        node_num = device_num // 8
        class HeapItem:
            def __init__(self, item) -> None:
                self.item = item
            def __lt__(self, other):
                return self.item[0] / self.item[1] > other.item[0] / other.item[1]
        max_heap = []
        if node_num * expert_num < total_capacity: # each node has at least one expert
            for i in range(expert_num):
                heapq.heappush(max_heap, HeapItem((expert_loads[i], node_num, i)))
            now_capacity = node_num * expert_num
        else:
            for i in range(expert_num):
                heapq.heappush(max_heap, HeapItem((expert_loads[i], 1, i)))
            now_capacity = expert_num

        result = [0] * expert_num
        while now_capacity < total_capacity:
            item = heapq.heappop(max_heap)
            if item.item[1] >= node_num:
                if now_capacity + node_num <= total_capacity:
                    heapq.heappush(max_heap, HeapItem((item.item[0], item.item[1] + node_num, item.item[2])))
                    now_capacity += node_num
                else:
                    result[item.item[2]] = item.item[1]
            else:
                heapq.heappush(max_heap, HeapItem((item.item[0], item.item[1] + 1, item.item[2])))
                now_capacity += 1

        for item in max_heap:
            result[item.item[2]] = item.item[1]
        return result
    
    def get_greedy_placement(self, expert_replicas, expert_loads, n_device, n_expert, C_e):
        A = np.zeros((n_expert, n_device))
        device_expert_count = [0] * n_device
        device_loads = [0.0] * n_device
        
        # Round-robin assignment: assign one replica per expert per round
        expert_replica_indices = [0] * n_expert  # Track how many replicas assigned per expert
        replica_loads_per_expert = {}  # Precompute replica loads for each expert
        expert_list = []
        # Precompute replica load distributions
        for expert in range(n_expert):
            expert_load = expert_loads[expert]
            replicas_needed = expert_replicas[expert]
            replica_loads_per_expert[expert] = self._distribute_expert_load_precise(expert_load, replicas_needed)
            expert_list.extend((expert, load) for load in replica_loads_per_expert[expert])
        
        expert_list.sort(key=lambda x: x[1], reverse=True) # sort by load
        # Round-robin assignment
        node_num = n_device // 8
        # max_replicas = max(expert_replicas)
        # for round_num in range(max_replicas):
            # Sort experts by load (descending) for this round
            # experts_by_load = sorted(range(n_expert), key=lambda x: expert_loads[x])
        # if True:  
        for expert, load in expert_list:
            # Skip if this expert has already assigned all its replicas
            # if expert_replica_indices[expert] >= expert_replicas[expert]:
            #     continue
            
            # Find available devices
            available_devices = [i for i in range(n_device) if device_expert_count[i] < C_e]
            
            if available_devices:
                # Priority strategy: distribute replicas across different nodes first
                # Get nodes that already have this expert
                existing_nodes = [0] * node_num
                for dev in range(n_device):
                    if A[expert, dev] == 1:
                        existing_nodes[self.node_func(dev)] += 1
            
                min_load = min(existing_nodes)
                # Find available devices in nodes that don't have this expert yet
                new_node_devices = [i for i in available_devices 
                                    if existing_nodes[self.node_func(i)] == min_load]

                if new_node_devices:
                    # Choose device with minimum load from new nodes
                    best_device = min(new_node_devices, key=lambda x: device_loads[x])
                else:
                    # All nodes have this expert, choose by minimum load
                    best_device = min(available_devices, key=lambda x: device_loads[x])
                
                A[expert, best_device] += 1
                
                # Record the planned capacity for this replica
                # replica_index = expert_replica_indices[expert]
                # replica_load = replica_loads_per_expert[expert][replica_index]
                replica_load = load

                device_loads[best_device] += replica_load
                device_expert_count[best_device] += 1
                expert_replica_indices[expert] += 1
            else:
                assert False, "No more capacity"
        return A, max(device_loads)
        
    def greedy_load_balancing_heuristic(self,
                                        n_device,
                                        n_expert,
                                        E,
                                        C_e, ) -> Tuple:
        """Greedy load balancing heuristic with expert replication"""
        return gb.greedy_load_balancing_heuristic(n_device, n_expert, E, C_e)
        # Step 1: Determine expert replication strategy
        total_capacity = n_device * C_e
        expert_loads = [sum(E[i][j] for i in range(n_device)) for j in range(n_expert)]

        final_A = None
        final_max_load = None

        # Average
        if (C_e * n_device) % n_expert == 0:
            expert_replicas = [C_e * n_device // n_expert] * n_expert
            A, max_load = self.get_greedy_placement(expert_replicas, expert_loads, n_device, n_expert, C_e)
            if final_max_load is None or max_load < final_max_load:
                final_max_load = max_load
                final_A = A

        expert_replicas = self.new_allocate_expert_replicas_precise(expert_loads, n_device, C_e)
        
        A, max_load = self.get_greedy_placement(expert_replicas, expert_loads, n_device, n_expert, C_e)
        
        if final_max_load is None or max_load < final_max_load:
            final_max_load = max_load
            final_A = A
        # S = self._generate_smart_routing_with_capacities(n_device, n_expert, E, A, expert_replica_capacities)
        # total_time = self._calculate_total_time(n_device, n_expert, S, M_token)

        A_res = []
        for j in range(n_device):
            tmp = []
            for i in range(n_expert):
                while abs(final_A[i, j]) > 1e-6:
                    tmp.append(i)
                    A[i, j] -= 1
            A_res.append(tmp)
        
        return 0, 0, A_res # A, S, total_time

    def smartmoe_method(self,
                        n_device,
                        n_expert,
                        E,
                        C_e,
                        ) -> Tuple:
        """
        Smart MoE method implementing greedy expert placement algorithm
        Based on Algorithm 1: Greedy Expert Placement from the paper
        
        Args:
            n_device: Total number of devices
            n_expert: Total number of experts
            E: Token demand matrix [n_device][n_expert]
            C_e: Maximum experts per device
            M_token: Token size in bytes
            ep_size: Expert placement size (default 4 devices per unit)
            
        Returns:
            Tuple: (A, S, total_time) where:
                A: Expert assignment matrix
                S: Token routing matrix
                total_time: Total computation time
        """
        ep_size = n_expert // C_e
        # Initialize arrays as per Algorithm 1
        samples = [0] * (ep_size)  # Current samples per device
        experts = [0] * (ep_size)   # Current experts per device
        P = [[] for _ in range(n_expert)]  # Placement of experts
        
        # Calculate expert loads (total tokens per expert across all devices)
        expert_loads = [sum(E[i][j] for i in range(n_device)) for j in range(n_expert)]
        
        # Sort experts by load in descending order (C in Algorithm 1)
        sorted_experts = sorted(range(n_expert), key=lambda x: expert_loads[x], reverse=True)
        
        # Process each expert in descending order of load
        for expert_idx in sorted_experts:
            expert_load = expert_loads[expert_idx]
            
            # Find the best device for this expert
            Tmin = float('inf')
            best_device = -1
            
            # Check all devices for placement
            for device in range(ep_size):
                # Check capacity constraints: experts[device] < E/N and samples[device] < Tmin
                if (experts[device] < C_e and 
                    samples[device] < Tmin):
                    Tmin = samples[device]
                    best_device = device
            
            # If we found a suitable device, place the expert
            assert best_device != -1, "No suitable device found for expert"
            P[expert_idx].append(best_device)
            samples[best_device] += expert_load
            experts[best_device] += 1
        
        # Convert placement to assignment matrix A
        A = np.zeros((n_expert, n_device))
        for expert in range(n_expert):
            for device in P[expert]:
                for k in range(n_device//ep_size):
                    A[expert, k * ep_size + device] = 1
        # print(A, samples)
        # print(expert_loads)
        # S = self._generate_smart_routing(n_device, n_expert, E, A)
        # total_time = self._calculate_total_time(n_device, n_expert, S, M_token)
        
        # Convert A to the expected format
        A_res = []
        for j in range(n_device):
            tmp = []
            for i in range(n_expert):
                if abs(A[i, j] - 1) < 1e-6:
                    tmp.append(i)
            A_res.append(tmp)
        
        # print(A)
        return 0, 0, A_res

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

    def keep_previous(self,
                        global_expert_indices_numpy: np.ndarray,
                        ) -> Tuple:
        """
        Keep previous placement
        """
        n_device = global_expert_indices_numpy.shape[0]
        C_e = global_expert_indices_numpy.shape[1]
        A_res = []
        for i in range(n_device):
            tmp = []
            for j in range(C_e):
                    tmp.append(global_expert_indices_numpy[i, j])
            A_res.append(tmp)
        return 0, 0, A_res

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
        # print(default_A)
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
        # S = self._generate_smart_routing(n_device, n_expert, E, A)
        # total_time = self._calculate_total_time(n_device, n_expert, S)
        # print(A)
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
        # print("chosses:", expert_replicas, argmin_expert)
        if argmax_expert == argmin_expert:
            return A

        # print(argmax_expert, argmin_expert, expert_loads)

        new_A = copy.deepcopy(A)
        
        for i in range(node):
            argmax_device = []
            argmin_device = []
            for j in range(8):
                if new_A[argmax_expert, i * 8 + j] > 0:
                    argmax_device.append(i * 8 + j)
                if new_A[argmin_expert, i * 8 + j] > 0:
                    argmin_device.append(i * 8 + j)

            # final_argmax_device = min(argmax_device, key=lambda x: device_loads[x])
            final_argmin_device = min(argmin_device, key=lambda x: device_loads[x])
            
            # new_A[argmax_expert, final_argmin_device] += 1
            # new_A[argmin_expert, final_argmin_device] -= 1

            final_argmax_device = -1 # max(argmax_device, key=lambda x: device_loads[x])
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
        # comm_times = []
        comp_times = []
        
        for i in range(n_device):
            # Communication time
            # comm_time = 0
            # for j in range(n_expert):
            #     for k in range(n_device):
            #         if S[i, j, k] > 0:
            #             bw = self.bandwidth_function(i, k)
            #             comm_time += M_token * S[i, j, k] / bw * 1e-9
            
            # Computation time
            comp_time = 0
            for k in range(n_device):
                for j in range(n_expert):
                    if S[k, j, i] > 0:
                        comp_time += S[k, j, i] #  * self.v_comp / 1000
            
            # comm_times.append(comm_time)
            comp_times.append(comp_time)
        
        # print(sum(comm_times), max(comp_times))
        # print("final", 4 * sum(comm_times) + 3 * max(comp_times))
        return max(comp_times) # 4 * sum(comm_times) + 3 * max(comp_times)
    
    def new_allocate_expert_replicas_precise(self, expert_loads: List[float], device_num: int, capacity: int, capacity_factor: int = 1) -> List[int]:
        total_capacity = device_num * capacity * capacity_factor
        expert_num = len(expert_loads)
        node_num = device_num // 8
        
        class HeapItem:
            def __init__(self, item) -> None:
                self.item = item
            def __lt__(self, other):
                return self.item[0] / self.item[1] > other.item[0] / other.item[1]
        max_heap = []
        if node_num * expert_num < total_capacity: # each node has at least one expert
            for i in range(expert_num):
                heapq.heappush(max_heap, HeapItem((expert_loads[i], node_num, i)))
            now_capacity = node_num * expert_num
        else:
            for i in range(expert_num):
                heapq.heappush(max_heap, HeapItem((expert_loads[i], 1, i)))
            now_capacity = expert_num

        result = [0] * expert_num
        while now_capacity < total_capacity:
            item = heapq.heappop(max_heap)
            if item.item[1] >= node_num:
                if now_capacity + node_num <= total_capacity:
                    heapq.heappush(max_heap, HeapItem((item.item[0], item.item[1] + node_num, item.item[2])))
                    now_capacity += node_num
                else:
                    result[item.item[2]] = item.item[1]
            else:
                heapq.heappush(max_heap, HeapItem((item.item[0], item.item[1] + 1, item.item[2])))
                now_capacity += 1

        for item in max_heap:
            result[item.item[2]] = item.item[1]
        return result

    def _distribute_expert_load_precise(self, expert_load: int, replicas_needed: int) -> List[int]:
        """
        Precisely distribute expert load among replicas, handling remainder
        Uses largest remainder method for fairness
        """
        if replicas_needed <= 0:
            return []
        
        # Base load per replica
        base_load = expert_load // replicas_needed
        remainder = expert_load % replicas_needed
        
        # Distribute loads
        loads = [base_load] * replicas_needed
        
        # Distribute remainder to first 'remainder' replicas
        for i in range(remainder):
            loads[i] += 1
        
        return loads

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