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
        with open(computation_config_path, 'r') as f:
            self.computation_config = json.load(f)
        with open(network_config_path, 'r') as f:
            self.network_config = json.load(f)
            
        self.V_inter = self.network_config["inter_node"]  # GB/s
        self.V_intra = self.network_config["intra_node"]  # GB/s
        self.v_comp = self._extract_computation_speed()
        
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
    
    def greedy_load_balancing_heuristic(self,
                                        n_device,
                                        n_expert,
                                        E,
                                        C_e, ) -> Tuple:
        """Greedy load balancing heuristic with expert replication"""
        
        # Step 1: Determine expert replication strategy
        total_capacity = n_device * C_e
        expert_loads = [sum(E[i][j] for i in range(n_device)) for j in range(n_expert)]
        
        # Use precise allocation to fully utilize capacity
        # expert_replicas = self._allocate_expert_replicas_precise(expert_loads, total_capacity)
        expert_replicas = self.new_allocate_expert_replicas_precise(expert_loads, n_device, C_e)
        
        # Step 2: Create expanded expert assignment matrix and track replica capacities
        A = np.zeros((n_expert, n_device))
        device_expert_count = [0] * n_device
        device_loads = [0.0] * n_device
        expert_replica_capacities = {}  # {expert: {device: capacity}}
        
        # Round-robin assignment: assign one replica per expert per round
        expert_replica_indices = [0] * n_expert  # Track how many replicas assigned per expert
        replica_loads_per_expert = {}  # Precompute replica loads for each expert
        
        # Precompute replica load distributions
        for expert in range(n_expert):
            expert_load = expert_loads[expert]
            replicas_needed = expert_replicas[expert]
            replica_loads_per_expert[expert] = self._distribute_expert_load_precise(expert_load, replicas_needed)
            expert_replica_capacities[expert] = {}
        
        # Round-robin assignment
        node_num = n_device // 8
        max_replicas = max(expert_replicas)
        for round_num in range(max_replicas):
            # Sort experts by load (descending) for this round
            experts_by_load = sorted(range(n_expert), key=lambda x: expert_loads[x], reverse=True)
            
            for expert in experts_by_load:
                # Skip if this expert has already assigned all its replicas
                if expert_replica_indices[expert] >= expert_replicas[expert]:
                    continue
                
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
                    replica_index = expert_replica_indices[expert]
                    replica_load = replica_loads_per_expert[expert][replica_index]
                    expert_replica_capacities[expert][best_device] = replica_load
                    
                    device_loads[best_device] += replica_load
                    device_expert_count[best_device] += 1
                    expert_replica_indices[expert] += 1
                else:
                    break  # No more capacity
        
        # S = self._generate_smart_routing_with_capacities(n_device, n_expert, E, A, expert_replica_capacities)
        # total_time = self._calculate_total_time(n_device, n_expert, S, M_token)

        A_res = []
        for j in range(n_device):
            tmp = []
            for i in range(n_expert):
                while abs(A[i, j]) > 1e-6:
                    tmp.append(i)
                    A[i, j] -= 1
            A_res.append(tmp)
        
        return 0, 0, A_res # A, S, total_time

    def _generate_smart_routing(self, n_device: int, n_expert: int, E: List[List[int]], 
                               A: np.ndarray) -> np.ndarray:
        """Backward compatibility wrapper for the old routing function"""
        # Create empty capacities dict for backward compatibility
        expert_replica_capacities = {}
        for expert in range(n_expert):
            expert_devices = [i for i in range(n_device) if A[expert, i] == 1]
            expert_replica_capacities[expert] = {device: float('inf') for device in expert_devices}
        
        return self._generate_smart_routing_with_capacities(n_device, n_expert, E, A, expert_replica_capacities)
    
    def _calculate_total_time(self, n_device: int, n_expert: int, S: np.ndarray, 
                             M_token: float) -> float:
        """Calculate total time (objective function)"""
        comm_times = []
        comp_times = []
        
        for i in range(n_device):
            # Communication time
            comm_time = 0
            for j in range(n_expert):
                for k in range(n_device):
                    if S[i, j, k] > 0:
                        bw = self.bandwidth_function(i, k)
                        comm_time += M_token * S[i, j, k] / bw * 1e-9
            
            # Computation time
            comp_time = 0
            for k in range(n_device):
                for j in range(n_expert):
                    if S[k, j, i] > 0:
                        comp_time += S[k, j, i] * self.v_comp / 1000
            
            comm_times.append(comm_time)
            comp_times.append(comp_time)
        
        return 4 * sum(comm_times) + 3 * max(comp_times)

    def _allocate_expert_replicas_precise(self, expert_loads: List[float], total_capacity: int) -> List[int]:
        """
        Precisely allocate expert replicas using Largest Remainder Method
        Ensures total_replicas = total_capacity with maximum fairness
        """
        n_expert = len(expert_loads)
        total_load = sum(expert_loads)
        
        # Calculate exact proportional allocation
        exact_allocations = [load / total_load * total_capacity for load in expert_loads]
        
        # Get integer parts and remainders
        integer_parts = [int(alloc) for alloc in exact_allocations]
        remainders = [exact_allocations[i] - integer_parts[i] for i in range(n_expert)]
        
        # Ensure each expert gets at least 1 replica
        for i in range(n_expert):
            if integer_parts[i] == 0:
                integer_parts[i] = 1
        
        current_total = sum(integer_parts)

        # print(integer_parts, current_total, total_capacity)
        
        # If we exceed capacity, reduce from experts with smallest remainders
        while current_total > total_capacity:
            # Sort by remainder (ascending) to reduce from smallest remainders first
            sorted_indices = sorted(range(n_expert), key=lambda x: remainders[x])
            for i in range(n_expert):
                idx = sorted_indices[i]
                if integer_parts[idx] > 1:  # Don't reduce below 1
                    integer_parts[idx] -= 1
                    current_total -= 1
                    if current_total == total_capacity:
                        break
        
        # If we have remaining capacity, assign to experts with largest remainders
        while current_total < total_capacity:
            # Sort by remainder (descending) to assign to largest remainders first
            sorted_indices = sorted(range(n_expert), key=lambda x: remainders[x], reverse=True)
            for i in range(n_expert):
                idx = sorted_indices[i]  # Cycle if needed
                integer_parts[idx] += 1
                current_total += 1
                if current_total == total_capacity:
                    break
        
        # Validation
        assert sum(integer_parts) == total_capacity, f"Total replicas {sum(integer_parts)} != capacity {total_capacity}"
        assert all(r >= 1 for r in integer_parts), "All experts must have at least 1 replica"
        
        return integer_parts
    
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

    def _generate_smart_routing_with_capacities(self, n_device: int, n_expert: int, E: List[List[int]], 
                                               A: np.ndarray, expert_replica_capacities: dict) -> np.ndarray:
        """Generate robust routing strategy using pre-allocated replica capacities with smart routing logic"""
        S = np.zeros((n_device, n_expert, n_device))
        
        # Track current loads for each expert replica
        expert_current_loads = {}
        for expert in range(n_expert):
            expert_current_loads[expert] = {}
            for device in range(n_device):
                if A[expert, device] == 1:
                    expert_current_loads[expert][device] = 0
        
        # Build expert location mappings (similar to get_smart_routing_map logic)
        expert_to_locations = {}  # expert_id -> [(device, replica_idx), ...]
        expert_to_nodes = {}      # expert_id -> [node_id, ...]
        node_to_experts = {}      # node_id -> set(expert_ids)
        
        gpus_per_node = 8  # Assuming 8 GPUs per node, can be made configurable
        
        def get_node_id(device):
            return device // gpus_per_node
        
        # Build expert location mappings
        for device in range(n_device):
            node_id = get_node_id(device)
            if node_id not in node_to_experts:
                node_to_experts[node_id] = set()
                
            for expert in range(n_expert):
                if A[expert, device] == 1:  # Expert is replicated on this device
                    if expert not in expert_to_locations:
                        expert_to_locations[expert] = []
                        expert_to_nodes[expert] = set()
                    
                    expert_to_locations[expert].append((device, 0))  # replica_idx is 0 for single replica
                    expert_to_nodes[expert].add(node_id)
                    node_to_experts[node_id].add(expert)
        
        # Two-phase routing strategy (similar to get_smart_routing_map)
        for mode in range(2):
            for src_device in range(n_device):
                src_node = get_node_id(src_device)
                
                for expert in range(n_expert):
                    if E[src_device][expert] == 0:
                        continue
                    
                    if expert not in expert_to_locations:
                        continue
                    
                    all_locations = expert_to_locations[expert]
                    intra_node_locations = [(device, replica_idx) for device, replica_idx in all_locations 
                                          if get_node_id(device) == src_node]
                    
                    remaining_tokens = E[src_device][expert]
                    
                    if mode == 0:  # Phase 1: Intra-node routing (evenly distribute)
                        if intra_node_locations:
                            self._distribute_tokens_evenly_intra_node(
                                S, src_device, expert, intra_node_locations, 
                                remaining_tokens, expert_current_loads, expert_replica_capacities
                            )
                    else:  # Phase 2: Inter-node routing (evenly)
                        if not intra_node_locations:
                            self._distribute_tokens_evenly(
                                S, src_device, expert, all_locations, 
                                remaining_tokens, expert_current_loads, expert_replica_capacities
                            )
        
        return S
    
    def _distribute_tokens_evenly(self, S, src_device, expert, locations, total_tokens, 
                                 expert_current_loads, expert_replica_capacities):
        """Generic even token allocation for both intra-node and inter-node"""
        if not locations:
            return
            
        num_locations = len(locations)
        tokens_per_location = total_tokens // num_locations
        remaining_tokens = total_tokens % num_locations
        
        # Sort locations by available capacity (prefer less loaded replicas)
        location_capacities = []
        for device, replica_idx in locations:
            planned_capacity = expert_replica_capacities[expert].get(device, 0)
            current_load = expert_current_loads[expert].get(device, 0)
            available_capacity = max(0, planned_capacity - current_load)
            location_capacities.append((available_capacity, device, replica_idx))
        
        location_capacities.sort(reverse=True)  # Sort by available capacity (descending)
        
        for i, (available_capacity, target_device, replica_idx) in enumerate(location_capacities):
            if total_tokens <= 0:
                break
                
            tokens_to_assign = tokens_per_location
            if i < remaining_tokens:
                tokens_to_assign += 1
            
            # Respect capacity constraints
            tokens_to_assign = min(tokens_to_assign, available_capacity, total_tokens)
            
            if tokens_to_assign > 0:
                S[src_device, expert, target_device] += tokens_to_assign
                expert_current_loads[expert][target_device] += tokens_to_assign
                total_tokens -= tokens_to_assign
    
    def _distribute_tokens_evenly_intra_node(self, S, src_device, expert, locations, total_tokens, 
                                           expert_current_loads, expert_replica_capacities):
        """Intra-node token average allocation (wrapper for _distribute_tokens_evenly)"""
        self._distribute_tokens_evenly(S, src_device, expert, locations, total_tokens, 
                                     expert_current_loads, expert_replica_capacities)
    
    def _distribute_tokens_greedy_inter_node(self, S, src_device, expert, locations, total_tokens,
                                           expert_current_loads, expert_replica_capacities):
        """Inter-node token greedy allocation (similar to _distribute_tokens_greedy)"""
        if not locations:
            return
            
        # Sort locations by load and communication cost
        loads_and_devices = []
        for device, replica_idx in locations:
            current_load = expert_current_loads[expert].get(device, 0)
            planned_capacity = expert_replica_capacities[expert].get(device, 0)
            available_capacity = max(0, planned_capacity - current_load)
            
            # Communication cost factor
            bandwidth = self.bandwidth_function(src_device, device)
            comm_cost = 1.0 / bandwidth if bandwidth > 0 else float('inf')
            
            # Combined score (load + communication cost)
            score = current_load + 0.1 * comm_cost  # Small weight for comm cost
            loads_and_devices.append((score, device, replica_idx, available_capacity))
        
        loads_and_devices.sort(key=lambda x: x[0])  # Sort by load (ascending)
        
        # Greedy allocation: fill least loaded replicas first
        for score, target_device, replica_idx, available_capacity in loads_and_devices:
            if total_tokens <= 0:
                break
                
            tokens_to_assign = min(total_tokens, available_capacity)
            
            if tokens_to_assign > 0:
                S[src_device, expert, target_device] += tokens_to_assign
                expert_current_loads[expert][target_device] += tokens_to_assign
                total_tokens -= tokens_to_assign