"""
MoE Async Linear Programming Solver

This module provides async functionality for calling MoEOptimizer's linear_programming_solve algorithm,
taking history_num_global_tokens_per_expert as input and solving optimization problems in background processes.
"""

import os
import time
import torch
from typing import List, Dict, Optional, Tuple, Any
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
import threading

from galvatron.core.runtime.moe.prefetch.solver import MoEOptimizer


def _worker_process(task_queue: Queue, result_queue: Queue, 
                   computation_config_path: str, network_config_path: str, hidden_size: int,
                   global_checkpoint: bool, worker_id: int):
    """Long-running worker process that handles optimization tasks"""
    try:
        # Initialize MoEOptimizer once in worker process
        optimizer = MoEOptimizer(
            computation_config_path=computation_config_path,
            network_config_path=network_config_path,
            hidden_size = hidden_size,
            global_checkpoint=global_checkpoint,
        )
        
        print(f"Worker {worker_id} initialized successfully")
        
        # Main worker loop
        while True:
            try:
                task_data = task_queue.get()
                
                # Check for shutdown signal
                if task_data is None:
                    print(f"Worker {worker_id} received shutdown signal")
                    break
                
                result = _solve_optimization_task(task_data, optimizer)
                result_queue.put(result)
                
            except Exception as e:
                error_result = {
                    "task_id": task_data.get("task_id", "unknown") if 'task_data' in locals() else "unknown",
                    "status": "error",
                    "error": f"Worker {worker_id} error: {str(e)}",
                    "layer_number": task_data.get("layer_number", -1) if 'task_data' in locals() else -1,
                    "timestamp": time.time()
                }
                result_queue.put(error_result)
                
    except Exception as e:
        print(f"Worker {worker_id} failed to initialize: {e}")


def _solve_optimization_task(task_data: Dict[str, Any], optimizer) -> Dict[str, Any]:
    """Solve optimization task using the pre-initialized optimizer"""
    try:
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
        C_e = expert_capacity_per_device
        # Call linear_programming_solve
        start_time = time.time()

        if task_data["solver"] == "FLEX":
            max_load, obj_value, A_res = optimizer.flexmoe_method(
                E=E,
                n_device=n_device,
                n_expert=num_experts,
                C_e=C_e,
                global_expert_indices_numpy=task_data["global_expert_indices_numpy"],
            )
        elif task_data["solver"] == "LAER":
            if task_data["ablation_approach"] == "no_even":
                no_even = True
                no_pq = False
            elif task_data["ablation_approach"] == "no_pq":
                no_even = False
                no_pq = True
            else:
                no_even = False
                no_pq = False
            max_load, obj_value, A_res = optimizer.greedy_load_balancing_heuristic(
                E=E,
                n_device=n_device,
                n_expert=num_experts,
                C_e=C_e,
                no_even=no_even,
                no_pq=no_pq,
            )
        else:
            max_load, obj_value, A_res = optimizer.default_placement(
                E=E,
                n_device=n_device,
                n_expert=num_experts,
                C_e=C_e,
            )

        # Process expert placement results
        location = [0 for _ in range(num_experts)]
        for i, A in enumerate(A_res):
            for j, expert in enumerate(A):
                location[expert] += 1
        max_location = max(location)
        
        expert_placement = torch.tensor(A_res, dtype=torch.int32, device="cpu")
        global_expert_locations = torch.full((num_experts, max_location), -1, dtype=torch.int32, device="cpu")
        location = [0 for _ in range(num_experts)]
        inverse_expert_map = torch.zeros(C_e * n_device, dtype=torch.int32, device="cpu")
        
        for i, A in enumerate(A_res):
            for j, expert in enumerate(A):
                global_expert_locations[expert, location[expert]] = i * C_e + j
                location[expert] += 1
                inverse_expert_map[i * C_e + j] = expert
        
        solve_time = time.time() - start_time

        result = {
            "task_id": task_data["task_id"],
            "status": "success",
            "layer_number": layer_number,
            "max_load": max_load,
            "objective_value": obj_value,
            "expert_placement": expert_placement.numpy(),
            "inverse_expert_map": inverse_expert_map.numpy(),
            "global_expert_locations": global_expert_locations.numpy(),
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


class AsyncLinearProgrammingSolver:
    """Async Linear Programming Solver Manager using worker process pool"""
    
    def __init__(self, 
                 computation_config_path: str,
                 network_config_path: str,
                 hidden_size: int,
                 global_checkpoint: bool,
                 ep_size: int,
                 num_experts: int,
                 expert_capacity_per_device: int,
                 num_workers: int = 1):
        """Initialize async linear programming solver"""
        self.computation_config_path = computation_config_path
        self.network_config_path = network_config_path
        self.hidden_size = hidden_size
        self.global_checkpoint = global_checkpoint
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.expert_capacity_per_device = expert_capacity_per_device
        self.num_workers = num_workers
        self.solver = os.getenv("SOLVER")
        self.ablation_approach = os.getenv("ABLATION_APPROACH", "LAER")
        
        # Communication queues
        self.task_queue = Queue()
        self.result_queue = Queue()
        
        # Worker processes
        self.workers = []
        self.result_cache = {}
        self.pending_tasks = set()
        
        # Result monitoring thread
        self.result_monitor_thread = None
        self.shutdown_flag = threading.Event()
        self.device = torch.cuda.current_device()
        
        # Start worker processes and result monitor
        self._start_workers()
        self._start_result_monitor()
    
    def _start_workers(self):
        """Start worker processes"""
        for i in range(self.num_workers):
            worker = Process(
                target=_worker_process,
                args=(self.task_queue, self.result_queue, 
                      self.computation_config_path, self.network_config_path, self.hidden_size, self.global_checkpoint, i),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"Started {self.num_workers} worker processes")
    
    def _start_result_monitor(self):
        """Start result monitoring thread"""
        self.result_monitor_thread = threading.Thread(
            target=self._result_monitor_loop,
            daemon=True
        )
        self.result_monitor_thread.start()
    
    def _result_monitor_loop(self):
        """Monitor result queue and cache results"""
        while not self.shutdown_flag.is_set():
            try:
                result = self.result_queue.get(timeout=0.1)
                
                task_id = result.get("task_id")
                if task_id:
                    self.result_cache[task_id] = result
                    self.pending_tasks.discard(task_id)
                    
            except:
                continue
    
    def submit_optimization_task(self, 
                                history_data: torch.Tensor,
                                layer_number: int,
                                task_id: Optional[str] = None,
                                solver_iter: int = 0,
                                global_expert_indices_numpy: np.ndarray = None) -> Optional[str]:
        """Submit linear programming optimization task to worker pool"""
        if task_id is None:
            task_id = f"lp_layer_{layer_number}_task_{int(time.time() * 1000)}_{self.device}"
        
        # Prepare task data
        task_data = {
            "task_id": task_id,
            "history_data": history_data,
            "layer_number": layer_number,
            "ep_size": self.ep_size,
            "num_experts": self.num_experts,
            "expert_capacity_per_device": self.expert_capacity_per_device,
            "timestamp": time.time(),
            "solver": self.solver,
            "ablation_approach": self.ablation_approach,
            "solver_iter": solver_iter,
            "global_expert_indices_numpy": global_expert_indices_numpy,
        }
        
        # Add to pending tasks and submit to queue
        self.pending_tasks.add(task_id)
        self.task_queue.put(task_data)
        
        return task_id
    
    def get_optimization_result(self, task_id: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Get linear programming optimization result"""
        # Check if result is already cached
        if task_id in self.result_cache:
            result = self.result_cache.pop(task_id)
            self.pending_tasks.discard(task_id)
            
            # Convert numpy arrays back to tensors
            if result.get("status") == "success":
                result["expert_placement_numpy"] = result["expert_placement"]
                result["expert_placement"] = torch.from_numpy(result["expert_placement"])
                result["inverse_expert_map"] = torch.from_numpy(result["inverse_expert_map"])
                result["global_expert_locations"] = torch.from_numpy(result["global_expert_locations"])
            
            return result
        
        print(f"Need to wait for task {task_id}...")
        # Wait for result with timeout
        if timeout > 0 and task_id in self.pending_tasks:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if task_id in self.result_cache:
                    result = self.result_cache.pop(task_id)
                    self.pending_tasks.discard(task_id)
                    
                    # Convert numpy arrays back to tensors
                    if result.get("status") == "success":
                        result["expert_placement_numpy"] = result["expert_placement"]
                        result["expert_placement"] = torch.from_numpy(result["expert_placement"])
                        result["inverse_expert_map"] = torch.from_numpy(result["inverse_expert_map"])
                        result["global_expert_locations"] = torch.from_numpy(result["global_expert_locations"])
                    
                    return result
                time.sleep(0.001)
        
        return None
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up AsyncLinearProgrammingSolver...")
        
        # Signal shutdown
        self.shutdown_flag.set()
        
        # Send shutdown signals to workers
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.terminate()
        
        # Wait for result monitor thread
        if self.result_monitor_thread and self.result_monitor_thread.is_alive():
            self.result_monitor_thread.join(timeout=2.0)
        
        # Clear caches
        self.result_cache.clear()
        self.pending_tasks.clear()
        self.workers.clear()
        
        print("Cleanup completed")


# Global solver instance
_global_lp_solver = None

def get_or_create_lp_solver(computation_config_path: str,
                           network_config_path: str,
                           hidden_size: int,
                           global_checkpoint: bool,
                           ep_size: int,
                           num_experts: int,
                           expert_capacity_per_device: int,
                           num_workers: int = 1) -> Optional[AsyncLinearProgrammingSolver]:
    """Get or create global linear programming solver instance"""
    global _global_lp_solver

    # Check if already exists and parameters match
    if (_global_lp_solver is not None):
        return _global_lp_solver
    
    # Create new solver
    _global_lp_solver = AsyncLinearProgrammingSolver(
        computation_config_path=computation_config_path,
        network_config_path=network_config_path,
        hidden_size=hidden_size,
        global_checkpoint=global_checkpoint,
        ep_size=ep_size,
        num_experts=num_experts,
        expert_capacity_per_device=expert_capacity_per_device,
        num_workers=num_workers
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
                          expert_capacity_per_device: int,
                          hidden_size: int,
                          global_checkpoint: bool,
                          num_workers: int = 1,
                          solver_iter: int = 0,
                          global_expert_indices_numpy: np.ndarray = None) -> Optional[str]:
    """Submit linear programming optimization task (simplified interface)"""
    ep_size, num_experts = history_data.shape
    solver = get_or_create_lp_solver(
        computation_config_path, network_config_path, hidden_size, global_checkpoint,
        ep_size, num_experts, expert_capacity_per_device, num_workers
    )
    if solver is None:
        return None
    return solver.submit_optimization_task(history_data, layer_number, solver_iter=solver_iter, global_expert_indices_numpy=global_expert_indices_numpy)


def get_lp_optimization_result(task_id: str, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
    """Get linear programming optimization result (simplified interface)"""
    global _global_lp_solver
    if _global_lp_solver is None:
        return None
    return _global_lp_solver.get_optimization_result(task_id, timeout) 