/**
 * Tests for TimeCostModel
 * 
 * These tests verify the functionality of the TimeCostModel class,
 * including device-specific scaling factors for computation time estimation.
 */

import TimeCostModel from '../src/models/TimeCostModel';

// Mock configuration and raw time config data
const mockConfig = {
  pp_size: 1,
  tp_size: 8,
  dp_size: 1, 
  global_batch_size: 8,
  hidden_dim: 512,
  num_layers: 12,
  seq_length: 128,
  micro_batch_size: 1,
  attention_heads: 8,
  ff_dim: 2048,
  chunks: 8,
  total_gpus: 8,
};

// Mock raw time config data
const mockRawTimeConfig = {
  'layertype_0': 5.0,
  'layertype_0_bsz1': 4.0,
  'layertype_0_bsz2': 8.0,
  'layertype_0_bsz4': 15.0,
  'layertype_0_bsz8': 30.0,
  'layernum[2]_bsz1': 10.0,
  'layernum[4]_bsz1': 20.0,
  'layernum[6]_bsz1': 30.0,
};

describe('TimeCostModel', () => {
  test('should correctly initialize with default device config', () => {
    const model = new TimeCostModel(mockConfig, null, mockRawTimeConfig);
    
    // Verify model has default device configuration
    expect(model.deviceConfig).toBeDefined();
    expect(model.deviceConfig.deviceType).toEqual('4090');
    expect(model.deviceConfig.deviceFactor).toEqual(0.8);
    expect(model.deviceConfig.deviceCount).toEqual(8);
  });
  
  test('should correctly apply device performance factor to computation time', () => {
    // Create model with default RTX 4090 (factor 0.8)
    const model4090 = new TimeCostModel(
      mockConfig, 
      { deviceType: '4090', deviceFactor: 0.8, deviceCount: 8, commEfficiency: 0.85 }, 
      mockRawTimeConfig
    );
    
    // Create model with A100 (factor 1.0)
    const modelA100 = new TimeCostModel(
      mockConfig, 
      { deviceType: 'a100', deviceFactor: 1.0, deviceCount: 8, commEfficiency: 0.85 }, 
      mockRawTimeConfig
    );
    
    // Get time cost results
    const results4090 = model4090.getTimeCost();
    const resultsA100 = modelA100.getTimeCost();
    
    // Verify that 4090 (factor 0.8) has higher computation time (slower) than A100 (factor 1.0)
    expect(results4090.forward_time).toBeGreaterThan(resultsA100.forward_time);
    // Verify the ratio is approximately 1.0 / 0.8 = 1.25
    const forwardTimeRatio = results4090.forward_time / resultsA100.forward_time;
    expect(forwardTimeRatio).toBeCloseTo(1.25, 1); // Within 10% of the expected ratio
  });
  
  test('should correctly apply communication efficiency factor', () => {
    // Create model with high communication efficiency
    const modelHighEff = new TimeCostModel(
      mockConfig, 
      { deviceType: '4090', deviceFactor: 0.8, deviceCount: 8, commEfficiency: 1.0 }, 
      mockRawTimeConfig
    );
    
    // Create model with lower communication efficiency
    const modelLowEff = new TimeCostModel(
      mockConfig, 
      { deviceType: '4090', deviceFactor: 0.8, deviceCount: 8, commEfficiency: 0.5 }, 
      mockRawTimeConfig
    );
    
    // Set TP and DP sizes to enable communication time calculation
    const configWithComm = {
      ...mockConfig,
      tp_size: 4,
      dp_size: 2,
    };
    
    // Create models with updated config
    const modelHighEffWithComm = new TimeCostModel(
      configWithComm, 
      { deviceType: '4090', deviceFactor: 0.8, deviceCount: 8, commEfficiency: 1.0 }, 
      mockRawTimeConfig
    );
    
    const modelLowEffWithComm = new TimeCostModel(
      configWithComm, 
      { deviceType: '4090', deviceFactor: 0.8, deviceCount: 8, commEfficiency: 0.5 }, 
      mockRawTimeConfig
    );
    
    // Get time cost results
    const resultsHighEff = modelHighEffWithComm.getTimeCost();
    const resultsLowEff = modelLowEffWithComm.getTimeCost();
    
    // Verify that model with lower efficiency has higher communication times
    expect(resultsLowEff.dp_communication_time).toBeGreaterThan(resultsHighEff.dp_communication_time);
    expect(resultsLowEff.tp_communication_time).toBeGreaterThan(resultsHighEff.tp_communication_time);
    
    // Verify the ratio is approximately inverse of efficiency ratio (1.0 / 0.5 = 2.0)
    const dpCommRatio = resultsLowEff.dp_communication_time / resultsHighEff.dp_communication_time;
    expect(dpCommRatio).toBeCloseTo(2.0, 1); // Within 10% of expected ratio
  });
  
  test('should handle different device types correctly', () => {
    // Create models with different device types
    const modelA100 = new TimeCostModel(
      mockConfig, 
      { deviceType: 'a100', deviceFactor: 1.0, deviceCount: 8, commEfficiency: 0.85 }, 
      mockRawTimeConfig
    );
    
    const modelH100 = new TimeCostModel(
      mockConfig, 
      { deviceType: 'h100', deviceFactor: 1.8, deviceCount: 8, commEfficiency: 0.85 }, 
      mockRawTimeConfig
    );
    
    const modelT4 = new TimeCostModel(
      mockConfig, 
      { deviceType: 't4', deviceFactor: 0.3, deviceCount: 8, commEfficiency: 0.85 }, 
      mockRawTimeConfig
    );
    
    // Get time cost results
    const resultsA100 = modelA100.getTimeCost();
    const resultsH100 = modelH100.getTimeCost();
    const resultsT4 = modelT4.getTimeCost();
    
    // Verify computation times are inversely proportional to device factors
    // H100 (1.8) < A100 (1.0) < T4 (0.3) in terms of computation time
    expect(resultsH100.forward_time).toBeLessThan(resultsA100.forward_time);
    expect(resultsA100.forward_time).toBeLessThan(resultsT4.forward_time);
    
    // Verify throughput is directly proportional to device factors
    // H100 (1.8) > A100 (1.0) > T4 (0.3) in terms of throughput
    expect(resultsH100.samples_per_second).toBeGreaterThan(resultsA100.samples_per_second);
    expect(resultsA100.samples_per_second).toBeGreaterThan(resultsT4.samples_per_second);
  });
});