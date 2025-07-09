/**
 * Utility functions for parsing Galvatron JSON configuration files
 * and extracting critical hardware performance parameters
 */

/**
 * Default hardware performance parameters for different device types
 */
export const DEVICE_PRESETS = {
  'A100': {
    forward_computation_time: 8,
    bct_fct_coe: 2.0,
    dp_overlap_coe: 1.0,
    bct_overlap_coe: 1.0,
    allreduce_bandwidth: 150,
    p2p_bandwidth: 300,
    sp_space: 'tp+sp',
    async_grad_reduce: false
  },
  'H100': {
    forward_computation_time: 5,
    bct_fct_coe: 2.0,
    dp_overlap_coe: 1.0,
    bct_overlap_coe: 1.0,
    allreduce_bandwidth: 200,
    p2p_bandwidth: 400,
    sp_space: 'tp+sp',
    async_grad_reduce: false
  },
  'V100': {
    forward_computation_time: 15,
    bct_fct_coe: 2.0,
    dp_overlap_coe: 1.0,
    bct_overlap_coe: 1.0,
    allreduce_bandwidth: 100,
    p2p_bandwidth: 200,
    sp_space: 'tp+sp',
    async_grad_reduce: false
  },
  'Custom': {
    forward_computation_time: 10,
    bct_fct_coe: 2.0,
    dp_overlap_coe: 1.0,
    bct_overlap_coe: 1.0,
    allreduce_bandwidth: 100,
    p2p_bandwidth: 300,
    sp_space: 'tp+sp',
    async_grad_reduce: false
  }
};

/**
 * Extract critical hardware performance parameters from Galvatron JSON config
 * @param {Object} rawConfig - Raw Galvatron configuration JSON
 * @returns {Object} Critical hardware performance parameters
 */
export function extractHardwareParams(rawConfig) {
  if (!rawConfig || !rawConfig.computation_profiling) {
    throw new Error('Invalid config: missing computation_profiling section');
  }

  const computationConfig = rawConfig.computation_profiling;
  
  // Extract forward computation time
  let forwardTime = 10; // Default value
  
  // Look for layertype_0 or similar entries
  const layertypeKey = Object.keys(computationConfig).find(key => 
    key.startsWith('layertype_0') && !key.includes('other')
  );
  
  if (layertypeKey) {
    // If it's a direct value
    if (typeof computationConfig[layertypeKey] === 'number') {
      forwardTime = computationConfig[layertypeKey];
    }
    // If it's batch size specific, find a reasonable default
    else if (typeof computationConfig[layertypeKey] === 'object') {
      const bszKeys = Object.keys(computationConfig[layertypeKey]).filter(k => k.includes('bsz'));
      if (bszKeys.length > 0) {
        // Use the first available batch size entry
        forwardTime = computationConfig[layertypeKey][bszKeys[0]];
      }
    }
  }

  // Convert to milliseconds if needed (assuming the profiled time is in seconds)
  if (forwardTime < 1) {
    forwardTime *= 1000;
  }

  // Extract other parameters with reasonable defaults
  // These would typically come from hardware profiling, but we'll use defaults
  const hardwareParams = {
    forward_computation_time: Math.max(1, forwardTime),
    bct_fct_coe: 2.0, // Standard backward/forward ratio
    dp_overlap_coe: 1.0, // No overlap by default
    bct_overlap_coe: 1.0, // No overlap by default
    allreduce_bandwidth: 100, // GB/s - conservative default
    p2p_bandwidth: 300, // GB/s - typical NVLink bandwidth
    sp_space: 'tp+sp',
    async_grad_reduce: false
  };

  return hardwareParams;
}

/**
 * Get device count from configuration or use default
 * @param {Object} config - Current configuration
 * @returns {number} Device count
 */
export function getDeviceCount(config) {
  if (config.total_gpus && config.total_gpus > 0) {
    return config.total_gpus;
  }
  
  // Calculate from parallelism dimensions
  const tpSize = config.tp_size || 1;
  const ppSize = config.pp_size || 1;
  const dpSize = config.dp_size || 1;
  
  return tpSize * ppSize * dpSize;
}

/**
 * Apply device preset to configuration
 * @param {Object} config - Current configuration
 * @param {string} presetName - Device preset name (A100, H100, V100, Custom)
 * @returns {Object} Updated configuration with preset parameters
 */
export function applyDevicePreset(config, presetName) {
  const preset = DEVICE_PRESETS[presetName];
  if (!preset) {
    console.warn(`Unknown device preset: ${presetName}`);
    return config;
  }

  return {
    ...config,
    hardware_preset: presetName,
    ...preset
  };
}

/**
 * Default trivial parameters for backend API calls
 */
export const DEFAULT_TRIVIAL_PARAMS = {
  // Add any other parameters that the backend needs but aren't critical
  extra_overhead: 0,
  costmodel_coe: 1.0,
  mixed_precision: true,
  model_microbatch_after_dp: true,
  pipeline_type: 'gpipe'
};