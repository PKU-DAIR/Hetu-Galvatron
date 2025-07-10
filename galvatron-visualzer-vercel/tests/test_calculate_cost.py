import sys
import os
import unittest
from typing import Dict, Any

# Add the parent directory to the path so we can import from api
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the models and functions from api directory
from api.calculate_cost import calculate_time, ModelConfig, DeviceConfig, TimeCostResult

class TestCalculateTime(unittest.TestCase):
    """Test the calculate_time function with different device configurations"""
    
    def setUp(self):
        """Set up test data"""
        # Create a basic model config
        self.model_config = ModelConfig(
            pp_size=1,
            tp_size=8,
            dp_size=1,
            global_batch_size=8,
            hidden_dim=512,
            num_layers=12,
            seq_length=128,
            vocab_size=30522,
            sequence_parallel=True,
            mixed_precision=True,
            micro_batch_size=1,
            attention_heads=8,
            ff_dim=2048,
            zero_stage=0,
            chunks=8,
            total_gpus=8,
            checkpoint=False
        )
        
        # Create mock time profiling data
        self.raw_time_config = {
            'layertype_0': 5.0,
            'layertype_0_bsz1': 4.0,
            'layertype_0_bsz2': 8.0,
            'layertype_0_bsz4': 15.0,
            'layertype_0_bsz8': 30.0
        }
    
    def test_default_device_config(self):
        """Test with default device configuration (RTX 4090)"""
        # Calculate time with default device config
        result = calculate_time(self.model_config, self.raw_time_config)
        
        # Verify result is a TimeCostResult object
        self.assertIsInstance(result, TimeCostResult)
        
        # Store result for comparison
        self.default_forward_time = result.forward_time
        self.default_backward_time = result.backward_time
        
        # Basic validation
        self.assertGreater(result.forward_time, 0)
        self.assertGreater(result.backward_time, 0)
        self.assertGreater(result.iteration_time, 0)
    
    def test_device_factor_scaling(self):
        """Test scaling of computation time with different device factors"""
        # Create device configs with different performance factors
        a100_config = DeviceConfig(device_type="a100", device_factor=1.0)
        h100_config = DeviceConfig(device_type="h100", device_factor=1.8)
        t4_config = DeviceConfig(device_type="t4", device_factor=0.3)
        
        # Calculate time for each device
        a100_result = calculate_time(self.model_config, self.raw_time_config, a100_config)
        h100_result = calculate_time(self.model_config, self.raw_time_config, h100_config)
        t4_result = calculate_time(self.model_config, self.raw_time_config, t4_config)
        
        # Verify computation times scale inversely with device factor
        # H100 (1.8) < A100 (1.0) < T4 (0.3) in terms of computation time
        self.assertLess(h100_result.forward_time, a100_result.forward_time)
        self.assertLess(a100_result.forward_time, t4_result.forward_time)
        
        # Verify scaling ratios are approximately correct (within 1% tolerance)
        h100_a100_ratio = h100_result.forward_time / a100_result.forward_time
        a100_t4_ratio = t4_result.forward_time / a100_result.forward_time
        
        self.assertAlmostEqual(h100_a100_ratio, 1.0/1.8, delta=0.01)
        self.assertAlmostEqual(a100_t4_ratio, 1.0/0.3, delta=0.01)
    
    def test_communication_efficiency(self):
        """Test impact of communication efficiency on communication times"""
        # Create a configuration with data parallelism to test DP communication
        dp_config = ModelConfig(
            pp_size=1,
            tp_size=4,
            dp_size=2,  # Enable DP communication
            global_batch_size=8,
            hidden_dim=512,
            num_layers=12,
            seq_length=128,
            vocab_size=30522,
            sequence_parallel=True,
            mixed_precision=True,
            micro_batch_size=1,
            attention_heads=8,
            ff_dim=2048,
            zero_stage=0,
            chunks=8,
            total_gpus=8,
            checkpoint=False
        )
        
        # Create device configs with different communication efficiency
        high_eff_config = DeviceConfig(device_type="4090", device_factor=0.8, comm_efficiency=1.0)
        low_eff_config = DeviceConfig(device_type="4090", device_factor=0.8, comm_efficiency=0.5)
        
        # Calculate time for each efficiency setting
        high_eff_result = calculate_time(dp_config, self.raw_time_config, high_eff_config)
        low_eff_result = calculate_time(dp_config, self.raw_time_config, low_eff_config)
        
        # Verify communication times scale inversely with efficiency
        self.assertGreater(low_eff_result.dp_communication_time, high_eff_result.dp_communication_time)
        self.assertGreater(low_eff_result.tp_communication_time, high_eff_result.tp_communication_time)
        
        # Verify scaling ratio is approximately correct (within 1% tolerance)
        dp_comm_ratio = low_eff_result.dp_communication_time / high_eff_result.dp_communication_time
        self.assertAlmostEqual(dp_comm_ratio, 1.0/0.5, delta=0.01)
    
    def test_device_count(self):
        """Test impact of device count on communication efficiency"""
        # Create device configs with different device counts
        few_devices = DeviceConfig(device_type="4090", device_factor=0.8, device_count=2)
        many_devices = DeviceConfig(device_type="4090", device_factor=0.8, device_count=64)
        
        # Calculate time for each device count
        few_devices_result = calculate_time(self.model_config, self.raw_time_config, few_devices)
        many_devices_result = calculate_time(self.model_config, self.raw_time_config, many_devices)
        
        # Expect computation times to be the same (device count doesn't directly affect computation)
        self.assertAlmostEqual(
            few_devices_result.forward_time, 
            many_devices_result.forward_time,
            delta=0.1  # Allow small variations due to floating point
        )
        
        # When testing with DP/TP, communication would be impacted by device count
        # but our simple test config doesn't have significant communication

if __name__ == '__main__':
    unittest.main()