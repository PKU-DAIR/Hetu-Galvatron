from flask import Flask, request, jsonify
import json
from calculate_cost import calculate_memory, calculate_time, ModelConfig, DeviceConfig, MemoryCostResult, TimeCostResult

app = Flask(__name__)

@app.route("/api/python")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/calculate_memory", methods=['POST'])
def calculate_memory_endpoint():
    request_data = request.get_json()
    
    config_data = request_data.get('config', {})
    raw_config_data = request_data.get('raw_memory_config')  # Changed from raw_time_config to raw_memory_config

    # Ensure config_data is a dictionary before unpacking
    if not isinstance(config_data, dict):
        return jsonify({"error": "Invalid config format"}), 400

    try:
        config = ModelConfig(**config_data)
    except TypeError as e:
        return jsonify({"error": f"Invalid config parameters: {str(e)}"}), 400
        
    result = calculate_memory(config, raw_config_data)
    
    return jsonify(result.__dict__)

@app.route("/api/calculate_time", methods=['POST'])
def calculate_time_endpoint():
    print("Calculate time endpoint called")
    request_data = request.get_json()
    
    config_data = request_data.get('config', {})
    
    print(f"Complete request data: {json.dumps(request_data, indent=2)}")
    
    # Ensure config_data is a dictionary before unpacking
    if not isinstance(config_data, dict):
        return jsonify({"error": "Invalid config format"}), 400

    try:
        # Create ModelConfig, filtering out extra parameters
        model_config_params = {
            'pp_size': config_data.get('pp_size', 1),
            'tp_size': config_data.get('tp_size', 8),
            'dp_size': config_data.get('dp_size', 1),
            'global_batch_size': config_data.get('global_batch_size', 8),
            'hidden_dim': config_data.get('hidden_dim', 512),
            'num_layers': config_data.get('num_layers', 12),
            'seq_length': config_data.get('seq_length', 128),
            'vocab_size': config_data.get('vocab_size', 30522),
            'sequence_parallel': config_data.get('sequence_parallel', True),
            'mixed_precision': config_data.get('mixed_precision', True),
            'micro_batch_size': config_data.get('micro_batch_size', 1),
            'attention_heads': config_data.get('attention_heads', 8),
            'ff_dim': config_data.get('ff_dim', 2048),
            'zero_stage': config_data.get('zero_stage', 0),
            'chunks': config_data.get('chunks', 8),
            'total_gpus': config_data.get('total_gpus', 8),
            'checkpoint': config_data.get('checkpoint', False),
            'use_ulysses': config_data.get('use_ulysses', False),
            'stage_idx': config_data.get('stage_idx', 0)
        }
        
        config = ModelConfig(**model_config_params)
        
        # Extract hardware parameters from request
        hardware_params = {
            'forward_computation_time': request_data.get('forward_computation_time', 10.0),
            'bct_fct_coe': request_data.get('bct_fct_coe', 2.0),
            'dp_overlap_coe': request_data.get('dp_overlap_coe', 1.0),
            'bct_overlap_coe': request_data.get('bct_overlap_coe', 1.0),
            'allreduce_bandwidth': request_data.get('allreduce_bandwidth', 100.0),
            'p2p_bandwidth': request_data.get('p2p_bandwidth', 300.0),
            'sp_space': request_data.get('sp_space', 'tp+sp'),
            'async_grad_reduce': request_data.get('async_grad_reduce', False),
            'device_count': request_data.get('device_count', 8)
        }
        
        print(f"Hardware params extracted: {hardware_params}")
        
    except TypeError as e:
        return jsonify({"error": f"Invalid config parameters: {str(e)}"}), 400
        
    try:
        result = calculate_time(config, **hardware_params)
        print(f"Time calculation result: {result}")
        return jsonify(result.__dict__)
    except Exception as e:
        print(f"Error calculating time cost: {str(e)}")
        return jsonify({"error": f"Error calculating time cost: {str(e)}"}), 500