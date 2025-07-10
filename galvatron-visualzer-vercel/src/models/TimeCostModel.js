/**
 * TimeCostModel: A model for estimating computation and communication time 
 * for large language model training across different hardware configurations.
 * 
 * Default device is 8× RTX 4090 GPUs with performance factor 0.8 (relative to A100).
 * This represents that RTX 4090 is approximately 80% as fast as A100 for LLM workloads.
 * The actual performance can vary significantly based on specific model architecture,
 * batch size, and sequence length.
 */

// Helper function for deep cloning objects
const deepClone = (obj) => {
  try {
    return JSON.parse(JSON.stringify(obj));
  } catch (error) {
    console.error("Deep clone failed:", error);
    return { ...obj }; // Fallback to shallow copy
  }
};

export default class TimeCostModel {
  constructor(config, deviceConfig, rawTimeConfig = null) {
    // Save raw config with deep copy
    this.rawTimeConfig = rawTimeConfig ? deepClone(rawTimeConfig) : null;
    
    // Save device configuration
    this.deviceConfig = deviceConfig || { 
      deviceType: 'a100', 
      deviceFactor: 1.0,
      deviceCount: 1,
      commEfficiency: 1.0
    };
    
    // Destructure config parameters
    const {
      pp_size = 1,
      tp_size = 8, 
      dp_size = 1,
      global_batch_size = 8,
      hidden_dim = 512,
      num_layers = 12,
      seq_length = 128,
      micro_batch_size = 1,
      attention_heads = 8,
      ff_dim = 2048,
      chunks = 8,
      total_gpus = 8,
    } = config;
    
    // Initialize basic parameters
    this.pp_size = pp_size;
    this.tp_size = tp_size;
    this.dp_size = dp_size;
    this.hidden_dim = hidden_dim;
    this.num_layers = num_layers;
    this.seq_length = seq_length;
    this.micro_batch_size = micro_batch_size;
    this.global_batch_size = global_batch_size;
    this.attention_heads = attention_heads;
    this.ff_dim = ff_dim;
    this.chunks = chunks;
    this.total_gpus = total_gpus;
    
    // Parse raw config if available
    if (this.rawTimeConfig) {
      this.parseRawTimeConfig();
    }
    
    // Initialize time calculations
    this.estimateComputationTime();
    this.estimateCommunicationTime();
    this.calculateTotalTime();
  }
  
  // Parse raw time configuration data
  parseRawTimeConfig() {
    if (!this.rawTimeConfig) {
      throw new Error("Missing required raw time configuration data. Please upload a Galvatron config file.");
    }
    
    // Find layer type keys
    const layerTypeKeys = Object.keys(this.rawTimeConfig).filter(key => key.startsWith('layertype_'));
    if (layerTypeKeys.length === 0) {
      throw new Error("Missing layertype timing data in raw configuration");
    }
    
    // Extract computation times
    const layernumKeys = Object.keys(this.rawTimeConfig).filter(key => key.startsWith('layernum['));
    if (layernumKeys.length === 0) {
      throw new Error("Missing layernum timing data in raw configuration");
    }
    
    // Get base computation time for a single layer
    const layerTypeKey = layerTypeKeys[0];
    this.baseLayerTime = this.rawTimeConfig[layerTypeKey] || 0;
    
    // Find the key with the closest batch size to our micro_batch_size
    const bszKeys = Object.keys(this.rawTimeConfig).filter(key => key.includes('_bsz'));
    
    // Extract batch sizes from keys
    const bszKeysWithValues = bszKeys.map(key => {
      const bszMatch = key.match(/_bsz(\d+)/);
      const bsz = bszMatch ? parseInt(bszMatch[1]) : 0;
      return {
        key,
        bsz
      };
    });
    
    // Find the closest batch size
    const sortedByDistance = [...bszKeysWithValues].sort((a, b) => 
      Math.abs(a.bsz - this.micro_batch_size) - Math.abs(b.bsz - this.micro_batch_size)
    );
    
    if (sortedByDistance.length > 0) {
      const closestKey = sortedByDistance[0].key;
      const closestBsz = sortedByDistance[0].bsz;
      
      // Scale time based on batch size ratio
      this.baseLayerTimePerBsz = this.rawTimeConfig[closestKey];
      this.bszScalingFactor = this.micro_batch_size / closestBsz;
    } else {
      // Fallback if no batch size keys found
      this.baseLayerTimePerBsz = this.baseLayerTime;
      this.bszScalingFactor = 1.0;
    }
    
    // Find layer num keys to get performance data for different numbers of layers
    this.layerScaling = {};
    const layernumRegex = /layernum\[(\d+)\]_bsz(\d+)/;
    
    layernumKeys.forEach(key => {
      const match = key.match(layernumRegex);
      if (match) {
        const layerNum = parseInt(match[1]);
        const bsz = parseInt(match[2]);
        
        if (!this.layerScaling[layerNum]) {
          this.layerScaling[layerNum] = {};
        }
        
        this.layerScaling[layerNum][bsz] = this.rawTimeConfig[key];
      }
    });
  }
  
  // Estimate computation time based on model size and device performance
  estimateComputationTime() {
    // Get computation time per layer
    let forwardTimePerLayer;
    
    if (this.rawTimeConfig) {
      // Use profiled data if available
      forwardTimePerLayer = this.baseLayerTimePerBsz * this.bszScalingFactor;
    } else {
      // Fallback calculation based on model parameters
      // This is a very rough estimate: hidden_dim^2 * seq_length * micro_batch_size
      const flopsPerLayer = this.hidden_dim * this.hidden_dim * 4 * this.seq_length * this.micro_batch_size;
      // Convert FLOPs to milliseconds (purely illustrative scaling)
      forwardTimePerLayer = flopsPerLayer / 1e9;
    }
    
    // Scale by device performance factor
    forwardTimePerLayer /= this.deviceConfig.deviceFactor;
    
    // Scale by tensor parallelism (approximate linear scaling)
    if (this.tp_size > 1) {
      forwardTimePerLayer /= Math.log2(this.tp_size) * 0.8; // Sub-linear scaling with TP
    }
    
    // Total forward time across all layers
    this.forward_time = forwardTimePerLayer * this.num_layers;
    
    // Backward pass is typically ~2x the forward pass
    this.backward_time = this.forward_time * 2;
  }
  
  // Estimate communication time
  estimateCommunicationTime() {
    // Data Parallel communication
    this.dp_communication_time = 0;
    if (this.dp_size > 1) {
      // Approximate gradient size: parameter_count * 4 bytes (float32)
      const parameterCount = this.hidden_dim * this.hidden_dim * 4 * this.num_layers;
      const gradientSizeMB = parameterCount * 4 / (1024 * 1024);
      
      // Simple model: communication time scales with gradient size and dp_size
      // Adjusted by communication efficiency factor
      this.dp_communication_time = gradientSizeMB * 0.01 * (this.dp_size - 1) / this.deviceConfig.commEfficiency;
    }
    
    // Tensor Parallel communication
    this.tp_communication_time = 0;
    if (this.tp_size > 1) {
      // Approximate activation size
      const activationSizeMB = this.seq_length * this.hidden_dim * this.micro_batch_size * 4 / (1024 * 1024);
      
      // All-reduce operations in TP scale with activation size and tp_size
      this.tp_communication_time = activationSizeMB * 0.005 * this.num_layers * (this.tp_size - 1) / this.deviceConfig.commEfficiency;
    }
    
    // Pipeline Parallel communication
    this.pp_communication_time = 0;
    if (this.pp_size > 1) {
      // Approximate activation size per pipeline boundary
      const activationSizeMB = this.seq_length * this.hidden_dim * this.micro_batch_size * 4 / (1024 * 1024);
      
      // Communication happens at pipeline boundaries
      this.pp_communication_time = activationSizeMB * 0.002 * (this.pp_size - 1) / this.deviceConfig.commEfficiency;
    }
  }
  
  // Calculate total time with overlap modeling
  calculateTotalTime() {
    // Total computation and communication times
    const total_computation = this.forward_time + this.backward_time;
    const total_communication = this.dp_communication_time + this.tp_communication_time + this.pp_communication_time;
    
    // Model overlap between computation and communication
    // Assume 30-50% overlap depending on device count (more devices = less efficient overlap)
    const overlapFactor = Math.max(0.3, 0.5 * this.deviceConfig.commEfficiency);
    
    // Total iteration time with overlap
    this.iteration_time = Math.max(total_computation, total_communication) + 
                          Math.min(total_computation, total_communication) * (1 - overlapFactor);
    
    // Calculate samples processed per second (throughput)
    this.samples_per_second = this.global_batch_size / (this.iteration_time / 1000);
  }
  
  // Get time cost results
  getTimeCost() {
    return {
      forward_time: this.forward_time,
      backward_time: this.backward_time,
      dp_communication_time: this.dp_communication_time,
      tp_communication_time: this.tp_communication_time,
      pp_communication_time: this.pp_communication_time,
      iteration_time: this.iteration_time,
      samples_per_second: this.samples_per_second
    };
  }
}