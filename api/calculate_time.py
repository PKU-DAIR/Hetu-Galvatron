from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from calculate_cost import calculate_time, ModelConfig

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Set CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            print(f"Received request data: {list(request_data.keys())}")
            
            # Extract model config
            model_config_data = request_data.get('config', {})
            model_config = ModelConfig(
                pp_size=model_config_data.get('pp_size', 1),
                tp_size=model_config_data.get('tp_size', 8),
                dp_size=model_config_data.get('dp_size', 1),
                global_batch_size=model_config_data.get('global_batch_size', 8),
                hidden_dim=model_config_data.get('hidden_dim', 512),
                num_layers=model_config_data.get('num_layers', 12),
                seq_length=model_config_data.get('seq_length', 128),
                vocab_size=model_config_data.get('vocab_size', 30522),
                sequence_parallel=model_config_data.get('sequence_parallel', True),
                mixed_precision=model_config_data.get('mixed_precision', True),
                micro_batch_size=model_config_data.get('micro_batch_size', 1),
                attention_heads=model_config_data.get('attention_heads', 8),
                ff_dim=model_config_data.get('ff_dim', 2048),
                zero_stage=model_config_data.get('zero_stage', 0),
                chunks=model_config_data.get('chunks', 8),
                total_gpus=model_config_data.get('total_gpus', 8),
                checkpoint=model_config_data.get('checkpoint', False),
                use_ulysses=model_config_data.get('use_ulysses', False),
                stage_idx=model_config_data.get('stage_idx', 0)
            )
            
            # Extract hardware parameters with defaults
            # Try to get from top-level or from config object
            hardware_params = {
                'forward_computation_time': float(request_data.get('forward_computation_time', 
                    model_config_data.get('forward_computation_time', 10.0))),
                'bct_fct_coe': float(request_data.get('bct_fct_coe', 
                    model_config_data.get('bct_fct_coe', 2.0))),
                'dp_overlap_coe': float(request_data.get('dp_overlap_coe', 
                    model_config_data.get('dp_overlap_coe', 1.0))),
                'bct_overlap_coe': float(request_data.get('bct_overlap_coe', 
                    model_config_data.get('bct_overlap_coe', 1.0))),
                'allreduce_bandwidth': float(request_data.get('allreduce_bandwidth', 
                    model_config_data.get('allreduce_bandwidth', 100.0))),
                'p2p_bandwidth': float(request_data.get('p2p_bandwidth', 
                    model_config_data.get('p2p_bandwidth', 300.0))),
                'sp_space': request_data.get('sp_space', 
                    model_config_data.get('sp_space', 'tp+sp')),
                'async_grad_reduce': bool(request_data.get('async_grad_reduce', 
                    model_config_data.get('async_grad_reduce', False))),
                'device_count': int(request_data.get('device_count', 
                    model_config_data.get('total_gpus', 8)))
            }
            
            print(f"Hardware params: {hardware_params}")
            
            # Calculate time cost
            result = calculate_time(model_config, **hardware_params)
            
            # Convert result to dictionary
            response_data = {
                'forward_time': result.forward_time,
                'backward_time': result.backward_time,
                'dp_communication_time': result.dp_communication_time,
                'tp_communication_time': result.tp_communication_time,
                'pp_communication_time': result.pp_communication_time,
                'iteration_time': result.iteration_time,
                'samples_per_second': result.samples_per_second
            }
            
            # Send response
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            error_response = {
                'error': str(e)
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def do_OPTIONS(self):
        # Handle CORS preflight request
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()