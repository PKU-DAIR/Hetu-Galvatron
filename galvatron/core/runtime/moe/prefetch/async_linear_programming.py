"""
MoE Async Linear Programming Solver

This module provides async functionality for calling MoEOptimizer's linear_programming_solve algorithm,
taking history_num_global_tokens_per_expert as input and solving optimization problems in background threads.
"""

import os
import time
import torch
from typing import List, Dict, Optional, Tuple, Any
from threading import Thread
import numpy as np

from galvatron.core.runtime.moe.prefetch.solver import MoEOptimizer


class AsyncLinearProgrammingSolver:
    """Async Linear Programming Solver Manager using per-task threads"""
    
    def __init__(self, 
                 computation_config_path: str,
                 network_config_path: str,
                 ep_size: int,
                 num_experts: int,
                 expert_capacity_per_device: int):
        """
        Initialize async linear programming solver
        
        Args:
            computation_config_path: Path to computation config file
            network_config_path: Path to network config file
            ep_size: Expert Parallel size
            num_experts: Total number of experts
            expert_capacity_per_device: Expert capacity per device
        """
        self.computation_config_path = computation_config_path
        self.network_config_path = network_config_path
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.expert_capacity_per_device = expert_capacity_per_device
        
        # Result cache and active threads
        self.result_cache = {}
        self.active_threads = {}
        
        # Initialize MoEOptimizer
        self.optimizer = None
        self._init_optimizer()
        self.device = torch.cuda.current_device()
    
    def _init_optimizer(self):
        """Initialize MoEOptimizer instance"""
        self.optimizer = MoEOptimizer(
            computation_config_path=self.computation_config_path,
            network_config_path=self.network_config_path
        )

    
    def submit_optimization_task(self, 
                                history_data: torch.Tensor,
                                layer_number: int,
                                task_id: Optional[str] = None) -> Optional[str]:
        """
        Submit linear programming optimization task using new thread
        
        Args:
            history_data: Historical token distribution data [ep_size, num_experts]
            layer_number: Layer number
            task_id: Task ID (optional)
            
        Returns:
            task_id: Task identifier, None if submission failed
        """
        if task_id is None:
            task_id = f"lp_layer_{layer_number}_task_{int(time.time() * 1000)}_{self.device}"
        
        # Prepare task data
        task_data = {
            "task_id": task_id,
            "history_data": history_data,  # Move to CPU
            "layer_number": layer_number,
            "ep_size": self.ep_size,
            "num_experts": self.num_experts,
            "expert_capacity_per_device": self.expert_capacity_per_device,
            "timestamp": time.time()
        }
        
        # Create new thread for this task
        thread = Thread(
            target=self._solve_task_in_thread,
            args=(task_data,),
            daemon=True
        )
        
        thread.start()
        self.active_threads[task_id] = thread
        return task_id
    
    def get_optimization_result(self, task_id: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Get linear programming optimization result
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds
            
        Returns:
            result: Optimization result, None if not completed
        """
        while task_id not in self.result_cache:
            time.sleep(0.001)
            
        if task_id in self.result_cache:
            result = self.result_cache.pop(task_id)
            # Clean up thread reference
            if task_id in self.active_threads:
                del self.active_threads[task_id]
            return result
        
        if timeout > 0:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if task_id in self.result_cache:
                    result = self.result_cache.pop(task_id)
                    if task_id in self.active_threads:
                        del self.active_threads[task_id]
                    return result
                time.sleep(0.001)  # 1ms
        
        return None
    
    def _solve_task_in_thread(self, task_data: Dict[str, Any]):
        """Solve optimization task in thread"""
        try:
            result = self._solve_linear_programming(task_data)
            self.result_cache[task_data["task_id"]] = result
                    
        except Exception:
            # Store error result
            self.result_cache[task_data["task_id"]] = {
                "task_id": task_data["task_id"],
                "status": "error",
                "error": "Thread execution failed",
                "layer_number": task_data.get("layer_number", -1),
                "timestamp": time.time()
            }
    
    def _solve_linear_programming(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call MoEOptimizer's linear_programming_solve
        
        Args:
            task_data: Task data dictionary
            
        Returns:
            result: Solve result dictionary
        """
        try:
            # Check if optimizer is available
            if self.optimizer is None:
                return {
                    "task_id": task_data["task_id"],
                    "status": "error",
                    "error": "MoEOptimizer not available",
                    "layer_number": task_data.get("layer_number", -1),
                    "timestamp": time.time()
                }
            
            # Extract parameters from task data
            history_data = task_data["history_data"]
            layer_number = task_data["layer_number"]
            ep_size = task_data["ep_size"]
            num_experts = task_data["num_experts"]
            expert_capacity_per_device = task_data["expert_capacity_per_device"]
            
            # Take latest token distribution data [ep_size, num_experts]
            latest_data = history_data
            
            # Convert data format: from [ep_size, num_experts] to E[device][expert]
            E = []
            n_device = ep_size
            
            for ep_idx in range(ep_size):
                device_tokens = []
                for expert_idx in range(num_experts):
                    tokens = latest_data[ep_idx, expert_idx].item()
                    device_tokens.append(tokens)
                E.append(device_tokens)
        
            # Set linear programming parameters
            C_e = expert_capacity_per_device  # Number of experts each device can place
            # TODO: Modify m_token automatically
            M_token = 4096 * 2  # Token size (bytes), simplified to 1
            
            # Call linear_programming_solve
            start_time = time.time()
            
            # max_load, obj_value, A_res = self.optimizer.linear_programming_solve(
            #     E=E,
            #     n_device=n_device,
            #     n_expert=num_experts,
            #     C_e=C_e,
            #     M_token=M_token,
            #     time_limit=5,  # 60 seconds timeout
            #     gap_limit=0.2  # 5% gap limit
            # )
            max_load, obj_value, A_res = self.optimizer.greedy_load_balancing_heuristic(
                E=E,
                n_device=n_device,
                n_expert=num_experts,
                C_e=C_e,
            )
            
            solve_time = time.time() - start_time

            result = {
                "task_id": task_data["task_id"],
                "status": "success",
                "layer_number": layer_number,
                "max_load": max_load,
                "objective_value": obj_value,
                "expert_placement": A_res,
                "solve_time": solve_time,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            return {
                "task_id": task_data["task_id"],
                "status": "error",
                "error": str(e),
                "layer_number": task_data.get("layer_number", -1),
                "timestamp": time.time()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get solver status"""
        return {
            "optimizer_available": self.optimizer is not None,
            "active_threads": len(self.active_threads),
            "cached_results": len(self.result_cache)
        }
    
    def cleanup(self):
        """Clean up resources"""
        # Clear caches
        self.result_cache.clear()
        self.active_threads.clear()


# Global solver instance
_global_lp_solver = None

def get_or_create_lp_solver(computation_config_path: str,
                           network_config_path: str,
                           ep_size: int,
                           num_experts: int,
                           expert_capacity_per_device: int) -> Optional[AsyncLinearProgrammingSolver]:
    """
    Get or create global linear programming solver instance
    
    Args:
        computation_config_path: Path to computation config file
        network_config_path: Path to network config file
        ep_size: Expert Parallel size
        num_experts: Total number of experts
        expert_capacity_per_device: Expert capacity per device
        
    Returns:
        solver: Async solver instance, None if unavailable
    """
    global _global_lp_solver

    # Check if already exists and parameters match
    if (_global_lp_solver is not None):
        return _global_lp_solver
    
    # Create new solver
    _global_lp_solver = AsyncLinearProgrammingSolver(
        computation_config_path=computation_config_path,
        network_config_path=network_config_path,
        ep_size=ep_size,
        num_experts=num_experts,
        expert_capacity_per_device=expert_capacity_per_device
    )
    
    return _global_lp_solver


def cleanup_global_lp_solver():
    """Clean up global linear programming solver"""
    global _global_lp_solver
    
    if _global_lp_solver is not None:
        _global_lp_solver.cleanup()
        _global_lp_solver = None


# Simplified interface functions
def submit_lp_optimization(history_data: torch.Tensor,
                          layer_number: int,
                          computation_config_path: str,
                          network_config_path: str,
                          expert_capacity_per_device: int) -> Optional[str]:
    """
    Submit linear programming optimization task (simplified interface)
    
    Args:
        history_data: Historical token distribution data
        layer_number: Layer number
        computation_config_path: Path to computation config file
        network_config_path: Path to network config file
        ep_size: Expert Parallel size
        num_experts: Total number of experts
        expert_capacity_per_device: Expert capacity per device
        
    Returns:
        task_id: Task ID, None if submission failed
    """
    ep_size, num_experts = history_data.shape
    solver = get_or_create_lp_solver(
        computation_config_path, network_config_path,
        ep_size, num_experts, expert_capacity_per_device
    )
    return solver.submit_optimization_task(history_data, layer_number)


def get_lp_optimization_result(task_id: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Get linear programming optimization result (simplified interface)
    
    Args:
        task_id: Task ID
        timeout: Timeout in seconds
        
    Returns:
        result: Optimization result, None if not completed
    """
    global _global_lp_solver
    return _global_lp_solver.get_optimization_result(task_id, timeout) 