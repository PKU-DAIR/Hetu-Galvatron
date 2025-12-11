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
from typing import Tuple
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
                                        C_e, ) -> Tuple:
        """Greedy load balancing heuristic with expert replication"""
        return gb.greedy_load_balancing_heuristic_complete(n_device, n_expert, E, C_e, self.hidden_size * 2, 2, self.v_comp, self.V_intra, self.V_inter, self.global_checkpoint)

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
