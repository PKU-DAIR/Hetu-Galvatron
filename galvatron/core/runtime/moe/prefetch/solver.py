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
        expert_replicas = self._allocate_expert_replicas_precise(expert_loads, total_capacity)
        
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
                    existing_nodes = set()
                    for dev in range(n_device):
                        if A[expert, dev] == 1:
                            existing_nodes.add(self.node_func(dev))
                    
                    # Find available devices in nodes that don't have this expert yet
                    new_node_devices = [i for i in available_devices 
                                      if self.node_func(i) not in existing_nodes]
                    
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
    
    def simulated_annealing_heuristic(self, *args, initial_temp=1000.0, cooling_rate=0.95, min_temp=1.0) -> Tuple:
        """Simulated annealing heuristic with expert replication"""
        n_device, n_expert, E, C_e, M_token = args
        
        # Determine expert replication strategy
        total_capacity = n_device * C_e
        expert_loads = [sum(E[i][j] for i in range(n_device)) for j in range(n_expert)]
        total_load = sum(expert_loads)
        
        expert_replicas = self._allocate_expert_replicas_precise(expert_loads, total_capacity)
        
        def random_neighbor(A):
            neighbor = A.copy()
            operation = random.choice(['swap_experts', 'replace_expert'])
            
            if operation == 'swap_experts':
                # Operation 1: Swap two experts between two devices
                devices_with_experts = [i for i in range(n_device) if neighbor[:, i].sum() > 0]
                
                if len(devices_with_experts) >= 2:
                    device1, device2 = random.sample(devices_with_experts, 2)
                    experts_on_device1 = [j for j in range(n_expert) if neighbor[j, device1] == 1]
                    experts_on_device2 = [j for j in range(n_expert) if neighbor[j, device2] == 1]
                    
                    if experts_on_device1 and experts_on_device2:
                        expert1 = random.choice(experts_on_device1)
                        expert2 = random.choice(experts_on_device2)
                        
                        # Perform swap
                        neighbor[expert1, device1] = 0
                        neighbor[expert2, device2] = 0
                        neighbor[expert1, device2] = 1
                        neighbor[expert2, device1] = 1
                        
            elif operation == 'replace_expert':
                # Operation 2: Replace one expert with another on same device
                devices_with_experts = [i for i in range(n_device) if neighbor[:, i].sum() > 0]
                
                if devices_with_experts:
                    device = random.choice(devices_with_experts)
                    experts_on_device = [j for j in range(n_expert) if neighbor[j, device] == 1]
                    experts_not_on_device = [j for j in range(n_expert) if neighbor[j, device] == 0]
                    
                    if experts_on_device and experts_not_on_device:
                        expert_to_remove = random.choice(experts_on_device)
                        expert_to_add = random.choice(experts_not_on_device)
                        
                        neighbor[expert_to_remove, device] = 0
                        neighbor[expert_to_add, device] = 1
            
            return neighbor
        
        def cost_function(A):
            # Rebuild expert replica capacities from assignment matrix
            expert_replica_capacities = self._rebuild_expert_capacities(A, expert_loads, expert_replicas, n_device, n_expert)
            S = self._generate_smart_routing_with_capacities(n_device, n_expert, E, A, expert_replica_capacities)
            return self._calculate_total_time(n_device, n_expert, S, M_token)
        
        # Initial solution using greedy
        current_A, _, _ = self.greedy_load_balancing_heuristic(*args)
        current_cost = cost_function(current_A)
        best_A = current_A.copy()
        best_cost = current_cost
        
        temp = initial_temp
        
        while temp > min_temp:
            neighbor_A = random_neighbor(current_A)
            neighbor_cost = cost_function(neighbor_A)
            
            if neighbor_cost < current_cost or random.random() < np.exp(-(neighbor_cost - current_cost) / temp):
                current_A = neighbor_A
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_A = current_A.copy()
                    best_cost = current_cost
            
            temp *= cooling_rate
        
        # Generate final routing with capacities
        best_expert_replica_capacities = self._rebuild_expert_capacities(best_A, expert_loads, expert_replicas, n_device, n_expert)
        S = self._generate_smart_routing_with_capacities(n_device, n_expert, E, best_A, best_expert_replica_capacities)
        
        return best_A, S, best_cost
    
    def _rebuild_expert_capacities(self, A: np.ndarray, expert_loads: List[int], expert_replicas: List[int], 
                                   n_device: int, n_expert: int) -> dict:
        """Rebuild expert replica capacities from assignment matrix"""
        expert_replica_capacities = {}
        
        for expert in range(n_expert):
            expert_replica_capacities[expert] = {}
            
            # Find devices assigned to this expert
            assigned_devices = [i for i in range(n_device) if A[expert, i] == 1]
            
            if assigned_devices:
                # Distribute expert load among assigned replicas
                expert_load = expert_loads[expert]
                replica_loads = self._distribute_expert_load_precise(expert_load, len(assigned_devices))
                
                # Assign loads to devices
                for device_idx, device in enumerate(assigned_devices):
                    expert_replica_capacities[expert][device] = replica_loads[device_idx]
        
        return expert_replica_capacities
    
    def eplb_heuristic(self, *args) -> Tuple:
        """Expert Parallel Load Balancer (EPLB) heuristic with expert replication"""
        n_device, n_expert, E, C_e, M_token = args
        
        # Step 1: Determine expert replication strategy
        total_capacity = n_device * C_e
        expert_loads = [sum(E[i][j] for i in range(n_device)) for j in range(n_expert)]
        total_load = sum(expert_loads)
        
        expert_replicas = []
        for j in range(n_expert):
            base_replicas = max(1, int(expert_loads[j] / total_load * total_capacity))
            expert_replicas.append(min(base_replicas, total_capacity // n_expert + 1))
        
        total_replicas = sum(expert_replicas)
        if total_replicas > total_capacity:
            scale = total_capacity / total_replicas
            expert_replicas = [max(1, int(r * scale)) for r in expert_replicas]
        
        # Step 2: EPLB-style assignment with replication
        A = np.zeros((n_expert, n_device))
        device_expert_count = [0] * n_device
        device_loads = [0.0] * n_device
        
        # Sort experts by load (descending)
        sorted_experts = sorted(range(n_expert), key=lambda x: expert_loads[x], reverse=True)
        
        # Assign expert replicas using EPLB principles
        for expert in sorted_experts:
            replicas_needed = expert_replicas[expert]
            expert_load = expert_loads[expert]
            
            # Get precise load distribution for this expert's replicas
            replica_loads = self._distribute_expert_load_precise(expert_load, replicas_needed)
            
            replica_index = 0
            for _ in range(replicas_needed):
                # Find devices that can still host more experts
                available_devices = [i for i in range(n_device) if device_expert_count[i] < C_e]
                
                if available_devices:
                    # Choose device with minimum load among available ones
                    best_device = min(available_devices, key=lambda d: device_loads[d])
                    A[expert, best_device] = 1
                    device_loads[best_device] += replica_loads[replica_index]
                    device_expert_count[best_device] += 1
                    replica_index += 1
                else:
                    break  # No more capacity
        
        # Generate routing strategy
        S = self._generate_smart_routing(n_device, n_expert, E, A)
        total_time = self._calculate_total_time(n_device, n_expert, S, M_token)
        
        return A, S, total_time
    
    def _balanced_packing(self, weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack n weighted objects to m packs with balanced weights"""
        num_layers, num_groups = weight.shape
        assert num_groups % num_packs == 0
        groups_per_pack = num_groups // num_packs

        if groups_per_pack == 1:
            pack_index = torch.arange(weight.size(-1), dtype=torch.int64, device=weight.device).expand(weight.shape)
            rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
            return pack_index, rank_in_pack

        indices = weight.float().sort(-1, descending=True).indices.cpu()
        pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device='cpu')
        rank_in_pack = torch.full_like(pack_index, fill_value=-1)
        
        for i in range(num_layers):
            pack_weights = [0] * num_packs
            pack_items = [0] * num_packs
            for group in indices[i]:
                pack = min((j for j in range(num_packs) if pack_items[j] < groups_per_pack), 
                          key=pack_weights.__getitem__)
                assert pack_items[pack] < groups_per_pack
                pack_index[i, group] = pack
                rank_in_pack[i, group] = pack_items[pack]
                pack_weights[pack] += weight[i, group]
                pack_items[pack] += 1
        return pack_index, rank_in_pack

    def _replicate_experts(self, weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replicate logical experts to physical replicas"""
        n, num_log = weight.shape
        num_redundant = num_phy - num_log
        assert num_redundant >= 0
        
        device = weight.device
        phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
        rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
        arangen = torch.arange(n, dtype=torch.int64, device=device)
        
        for i in range(num_log, num_phy):
            redundant_indices = (weight / logcnt).max(dim=-1).indices
            phy2log[:, i] = redundant_indices
            rank[:, i] = logcnt[arangen, redundant_indices]
            logcnt[arangen, redundant_indices] += 1
            
        return phy2log, rank, logcnt

    def _rebalance_experts_hierarchical(self, weight: torch.Tensor, num_physical_experts: int, 
                          num_groups: int, num_nodes: int, num_gpus: int):
        """Hierarchical expert rebalancing"""
        num_layers, num_logical_experts = weight.shape
        assert num_logical_experts % num_groups == 0
        group_size = num_logical_experts // num_groups 
        assert num_groups % num_nodes == 0
        groups_per_node = num_groups // num_nodes
        assert num_gpus % num_nodes == 0
        assert num_physical_experts % num_gpus == 0
        phy_experts_per_gpu = num_physical_experts // num_gpus

        def inverse(perm: torch.Tensor) -> torch.Tensor:
            inv = torch.empty_like(perm)
            inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape))
            return inv

        # Step 1: pack groups to nodes
        tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
        group_pack_index, group_rank_in_pack = self._balanced_packing(tokens_per_group, num_nodes) 
        log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) + 
                    torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
        mlog2log = inverse(log2mlog)

        # Step 2: construct redundant experts within nodes
        tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
        phy2mlog, phyrank, mlogcnt = self._replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)    

        # Step 3: pack physical_experts to GPUs
        tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
        pack_index, rank_in_pack = self._balanced_packing(tokens_per_phy, num_gpus // num_nodes)
        phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
        pphy2phy = inverse(phy2pphy)

        pphy2mlog = phy2mlog.gather(-1, pphy2phy)
        pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + 
                     torch.arange(0, num_logical_experts, num_logical_experts // num_nodes,
                                  device=group_pack_index.device).view(1, -1, 1)).flatten(-2)
        pphy2log = mlog2log.gather(-1, pphy2mlog)
        pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
        logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
        
        return pphy2log, pphyrank, logcnt

    def _rebalance_experts(self, weight: torch.Tensor, num_replicas: int, num_groups: int,
                          num_nodes: int, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Main EPLB entry point"""
        num_layers, num_logical_experts = weight.shape
        weight = weight.float().cpu()
        
        if num_groups % num_nodes == 0:
            # Use hierarchical load-balance policy
            phy2log, phyrank, logcnt = self._rebalance_experts_hierarchical(
                weight, num_replicas, num_groups, num_nodes, num_gpus
            )
        else:
            # Use global load-balance policy
            phy2log, phyrank, logcnt = self._rebalance_experts_hierarchical(
                weight, num_replicas, 1, 1, num_gpus
            )
            
        maxlogcnt = logcnt.max().item()
        log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt), 
                            -1, dtype=torch.int64, device=logcnt.device)
        log2phy.view(num_layers, -1).scatter_(-1, phy2log * maxlogcnt + phyrank, 
                torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1))
        
        return phy2log, log2phy, logcnt
    
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
            remaining = total_capacity - current_total
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

    def analyze_communication_breakdown(self, n_device: int, n_expert: int, S: np.ndarray, 
                                        M_token: float, node_func: Callable[[int], int]) -> dict:
        """Analyze communication breakdown by type: device-local, intra-node, inter-node"""
        
        comm_stats = {
            'device_local': {'tokens': 0, 'time': 0.0, 'count': 0},
            'intra_node': {'tokens': 0, 'time': 0.0, 'count': 0}, 
            'inter_node': {'tokens': 0, 'time': 0.0, 'count': 0}
        }
        
        for i in range(n_device):
            for j in range(n_expert):
                for k in range(n_device):
                    if S[i, j, k] > 0:
                        tokens = S[i, j, k]
                        bandwidth = self.bandwidth_function(i, k, node_func)
                        comm_time = M_token * tokens / bandwidth * 1e-9
                        
                        # Classify communication type
                        if i == k:
                            # Same device (local)
                            comm_stats['device_local']['tokens'] += tokens
                            comm_stats['device_local']['time'] += comm_time
                            comm_stats['device_local']['count'] += 1
                        elif node_func(i) == node_func(k):
                            # Same node, different device (intra-node)
                            comm_stats['intra_node']['tokens'] += tokens
                            comm_stats['intra_node']['time'] += comm_time
                            comm_stats['intra_node']['count'] += 1
                        else:
                            # Different node (inter-node)
                            comm_stats['inter_node']['tokens'] += tokens
                            comm_stats['inter_node']['time'] += comm_time
                            comm_stats['inter_node']['count'] += 1
        
        # Calculate statistics
        total_tokens = sum(stat['tokens'] for stat in comm_stats.values())
        total_time = sum(stat['time'] for stat in comm_stats.values())
        
        for comm_type, stats in comm_stats.items():
            if total_tokens > 0:
                stats['token_ratio'] = stats['tokens'] / total_tokens
            else:
                stats['token_ratio'] = 0.0
            
            if total_time > 0:
                stats['time_ratio'] = stats['time'] / total_time
            else:
                stats['time_ratio'] = 0.0
        
        return comm_stats

    def analyze_per_device_communication(self, n_device: int, n_expert: int, S: np.ndarray, 
                                       M_token: float, node_func: Callable[[int], int]) -> dict:
        """Analyze detailed per-device communication breakdown"""
        
        device_stats = {}
        
        for device_id in range(n_device):
            stats = {
                'device_local': {'tokens': 0, 'time': 0.0},
                'intra_node': {'tokens': 0, 'time': 0.0}, 
                'inter_node': {'tokens': 0, 'time': 0.0},
                'total_send': {'tokens': 0, 'time': 0.0},
                'total_recv': {'tokens': 0, 'time': 0.0},
                'computation_load': 0
            }
            
            # Analyze outgoing communication (device_id as source)
            for j in range(n_expert):
                for k in range(n_device):
                    if S[device_id, j, k] > 0:
                        tokens = S[device_id, j, k]
                        bandwidth = self.bandwidth_function(device_id, k, node_func)
                        comm_time = M_token * tokens / bandwidth * 1e-9
                        
                        # Add to total send
                        stats['total_send']['tokens'] += tokens
                        stats['total_send']['time'] += comm_time
                        
                        # Classify by communication type
                        if device_id == k:
                            stats['device_local']['tokens'] += tokens
                            stats['device_local']['time'] += comm_time
                        elif node_func(device_id) == node_func(k):
                            stats['intra_node']['tokens'] += tokens
                            stats['intra_node']['time'] += comm_time
                        else:
                            stats['inter_node']['tokens'] += tokens
                            stats['inter_node']['time'] += comm_time
            
            # Analyze incoming communication (device_id as target) for computation load
            for i in range(n_device):
                for j in range(n_expert):
                    if S[i, j, device_id] > 0:
                        tokens = S[i, j, device_id]
                        stats['total_recv']['tokens'] += tokens
                        stats['computation_load'] += tokens
            
            device_stats[device_id] = stats
        
        return device_stats

    def calculate_load_balance_metrics(self, device_stats: dict) -> dict:
        """Calculate load balance metrics"""
        n_device = len(device_stats)
        
        # Extract computation loads
        comp_loads = [device_stats[i]['computation_load'] for i in range(n_device)]
        send_loads = [device_stats[i]['total_send']['tokens'] for i in range(n_device)]
        recv_loads = [device_stats[i]['total_recv']['tokens'] for i in range(n_device)]
        
        def calculate_balance_stats(loads):
            if not loads:
                return {'mean': 0, 'std': 0, 'cv': 0, 'max_min_ratio': 0}
            
            mean_load = np.mean(loads)
            std_load = np.std(loads)
            cv = std_load / mean_load if mean_load > 0 else 0
            max_load = max(loads)
            min_load = min(loads)
            max_min_ratio = max_load / min_load if min_load > 0 else float('inf')
            
            return {
                'mean': mean_load,
                'std': std_load,
                'cv': cv,
                'max_min_ratio': max_min_ratio,
                'max': max_load,
                'min': min_load
            }
        
        return {
            'computation': calculate_balance_stats(comp_loads),
            'send': calculate_balance_stats(send_loads),
            'recv': calculate_balance_stats(recv_loads)
        }

    def _generate_smart_routing_with_capacities(self, n_device: int, n_expert: int, E: List[List[int]], 
                                               A: np.ndarray, expert_replica_capacities: dict, 
                                               node_func: Callable[[int], int]) -> np.ndarray:
        """Generate robust routing strategy using pre-allocated replica capacities"""
        S = np.zeros((n_device, n_expert, n_device))
        
        # Track current loads for each expert replica
        expert_current_loads = {}
        for expert in range(n_expert):
            expert_current_loads[expert] = {}
            for device in range(n_device):
                if A[expert, device] == 1:
                    expert_current_loads[expert][device] = 0
        
        # Route tokens for each device and expert
        for i in range(n_device):
            for j in range(n_expert):
                if E[i][j] == 0:
                    continue
                
                # Get available target devices for this expert
                target_devices = [k for k in range(n_device) if A[j, k] == 1]
                if not target_devices:
                    continue
                
                # Robust routing: distribute tokens across multiple replicas if needed
                remaining_tokens = E[i][j]
                
                # Sort target devices by preference (local first, then by communication cost and capacity)
                sorted_targets = self._sort_replicas_by_preference(
                    i, j, target_devices, expert_replica_capacities, expert_current_loads, node_func
                )
                
                for target_device in sorted_targets:
                    if remaining_tokens <= 0:
                        break
                    
                    # Calculate how many tokens this replica can accept
                    planned_capacity = expert_replica_capacities[j].get(target_device, 0)  # No capacity if not assigned
                    current_load = expert_current_loads[j][target_device]
                    available_capacity = max(0, planned_capacity - current_load)
                    
                    # Route as many tokens as possible to this replica
                    tokens_to_route = min(remaining_tokens, available_capacity)
                    
                    if tokens_to_route > 0:
                        S[i, j, target_device] += tokens_to_route
                        expert_current_loads[j][target_device] += tokens_to_route
                        remaining_tokens -= tokens_to_route
                
                # Handle any remaining tokens (fallback mechanism)
                if remaining_tokens > 0:
                    # If strict capacity constraints prevent routing, 
                    # route to the best available replica anyway (with overload)
                    best_target = sorted_targets[0]  # Best available target
                    S[i, j, best_target] += remaining_tokens
                    expert_current_loads[j][best_target] += remaining_tokens
                    
                    # Log warning about capacity violation
                    # print(f"Warning: Expert {j} replica on device {best_target} overloaded by {remaining_tokens} tokens")
        
        return S
    
    def _sort_replicas_by_preference(self, source_device: int, expert: int, target_devices: List[int],
                                   expert_replica_capacities: dict, expert_current_loads: dict) -> List[int]:
        """Sort replica devices by routing preference (local first, then by cost and capacity)"""
        
        def replica_preference_score(device):
            # Communication cost using bandwidth_function (primary factor)
            bandwidth = self.bandwidth_function(source_device, device)
            comm_cost = 1.0 / bandwidth if bandwidth > 0 else float('inf')
            
            # Capacity availability (secondary factor)
            planned_capacity = expert_replica_capacities[expert].get(device, 0)  # No capacity if not assigned
            current_load = expert_current_loads[expert].get(device, 0)
            available_capacity = max(0, planned_capacity - current_load)
            
            # Start with communication cost as primary score
            score = comm_cost
            
            # Penalize replicas with no available capacity
            if available_capacity <= 0:
                score += 1000.0  # Heavy penalty for full replicas
            else:
                # Small preference for replicas with more available capacity
                # TODO: Small or Large?
                capacity_bonus = 1.0 / (available_capacity + 1.0)
                score += 0.01 * capacity_bonus  # Much smaller weight than communication cost
            
            return score
        
        return sorted(target_devices, key=replica_preference_score)