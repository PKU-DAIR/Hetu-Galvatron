// Helper function for deep cloning objects
const deepClone = (obj) => {
  try {
    return JSON.parse(JSON.stringify(obj));
  } catch (error) {
    console.error("Deep clone failed:", error);
    return { ...obj }; // Fallback to shallow copy
  }
};

export default class MemoryCostModel {
    constructor(config, rawConfig = null) {
      // Save raw config with deep copy
      this.rawConfig = rawConfig ? deepClone(rawConfig) : null;
      
      // Destructure config parameters
      const {
        pp_size = 1,
        tp_size = 8, 
        dp_size = 1,
        global_batch_size = 8,
        hidden_dim = 512,
        num_layers = 12,
        seq_length = 128,
        vocab_size = 30522,
        sequence_parallel = true,
        mixed_precision = true,
        micro_batch_size = 1,
        attention_heads = 8,
        ff_dim = 2048,
        zero_stage = 0,
        chunks = 8,
        total_gpus = 8,
        checkpoint = false,
        use_ulysses = false,
        stage_idx = 0,
      } = config;
      
      // Initialize basic parameters
      this.pp_size = pp_size;
      this.tp_size = tp_size;
      this.dp_size = dp_size;
      this.hidden_dim = hidden_dim;
      this.num_layers = num_layers;
      this.seq_length = seq_length;
      this.vocab_size = vocab_size;
      this.sequence_parallel = sequence_parallel;
      this.mixed_precision = mixed_precision;
      this.micro_batch_size = micro_batch_size;
      this.global_batch_size = global_batch_size;
      this.attention_heads = attention_heads;
      this.ff_dim = ff_dim;
      this.zero_stage = zero_stage;
      this.chunks = chunks;
      this.total_gpus = total_gpus;
      this.checkpoint = checkpoint;
      this.use_ulysses = use_ulysses;
      this.stage_idx = stage_idx;
      this.config_list = [];
      
      // Parse raw config if available
      if (this.rawConfig) {
        console.log("rawConfig:", this.rawConfig);
        this.parseRawConfig();
      }
      
      // Initialize model (corresponds to cost_model.py's initialize method)
      this.initialize();
      
      // Estimate parameter size
      this.estimateParameterSize();
      
      // Estimate model states size
      this.estimateModelStatesSize();
      
      // Estimate activation size
      this.estimateActivationSize();
      
      // Estimate other memory costs
      this.estimateOtherMemoryModelStatesSize();
      this.estimateOtherMemoryActivationSize();
      this.calculateMemory();
    }
    
    // Parse raw configuration data and extract useful information
    parseRawConfig() {
      if (!this.rawConfig) {
        throw new Error("Missing required raw configuration data. Please upload a Galvatron config file.");
      }
      
      console.log("Parsing raw config data...");
      
      // Extract model config information
      if (!this.rawConfig.model_config) {
        throw new Error("Config file is missing required model_config data");
      }

      const modelConfig = this.rawConfig.model_config;
      
      // Update model parameters
      this.attention_heads = modelConfig.n_heads || this.attention_heads;
      this.hidden_dim = modelConfig.dim || this.hidden_dim;
      this.ff_dim = modelConfig.ffn_dim || this.ff_dim;
      // this.num_layers = modelConfig.n_layers || this.num_layers;
      this.vocab_size = modelConfig.vocab_size || this.vocab_size;
      
      console.log("Model parameters extracted from config:", {
        attention_heads: this.attention_heads,
        hidden_dim: this.hidden_dim,
        ff_dim: this.ff_dim,
        //num_layers: this.num_layers,
        vocab_size: this.vocab_size
      });
      
      // Extract sequence length information
      const seqLengths = [];
      for (const key in this.rawConfig) {
        if (key.match(/layertype_\d+(?:_sp)?$/)) {
          const layerKey = key;
          Object.keys(this.rawConfig[layerKey]).forEach(seq => {
            const seqNum = parseInt(seq);
            if (!seqLengths.includes(seqNum)) {
              seqLengths.push(seqNum);
            }
          });
          break;
        }
      }

      // this.seq_length = seqLengths;

      // Determine layer type based on sequence parallel setting
      const layer_type = this.sequence_parallel ? "layertype_0_sp" : "layertype_0";
      if (this.sequence_parallel) {
        if (!this.rawConfig.layertype_0_sp) {
          throw new Error("sequence_parallel is true but layertype_0_sp is not found in rawConfig");
        }
      }
      else {
        if (!this.rawConfig.layertype_0) {
          throw new Error("sequence_parallel is false but layertype_0 is not found in rawConfig");
        }
      }

      // Extract parameter and activation sizes
      this.parameter_size = this.rawConfig[layer_type][seqLengths[seqLengths.length - 1]].parameter_size; 
      this.tp_activation_per_bsz_dict = this.rawConfig[layer_type][seqLengths[seqLengths.length - 1]].tp_activation_per_bsz_dict;
      for (const key in this.tp_activation_per_bsz_dict) {
        this.tp_activation_per_bsz_dict[key] = this.tp_activation_per_bsz_dict[key] * this.seq_length / seqLengths[seqLengths.length - 1];
      }

      // Get layer types for other memory calculations
      const other_layer_type_pp_off = this.sequence_parallel ? "other_memory_pp_off_sp" : "other_memory_pp_off";
      const other_layer_type_pp_on_first = this.sequence_parallel ? "other_memory_pp_on_first_sp" : "other_memory_pp_on_first";
      const other_layer_type_pp_on_last = this.sequence_parallel ? "other_memory_pp_on_last_sp" : "other_memory_pp_on_last";
      
      // Verify required config components exist
      if (this.sequence_parallel) {
        if (!this.rawConfig.other_memory_pp_off_sp) {
          throw new Error("sequence_parallel is true but other_memory_pp_off_sp is not found in rawConfig");
        }
        if (!this.rawConfig.other_memory_pp_on_first_sp) {
          throw new Error("sequence_parallel is true but other_memory_pp_on_first_sp is not found in rawConfig");
        }
        if (!this.rawConfig.other_memory_pp_on_last_sp) {
          throw new Error("sequence_parallel is true but other_memory_pp_on_last_sp is not found in rawConfig");
        }
      }
      else {
        if (!this.rawConfig.other_memory_pp_off) {
          throw new Error("sequence_parallel is false but other_memory_pp_off is not found in rawConfig");
        }
        if (!this.rawConfig.other_memory_pp_on_first) { 
          throw new Error("sequence_parallel is false but other_memory_pp_on_first is not found in rawConfig");
        }
        if (!this.rawConfig.other_memory_pp_on_last) {
          throw new Error("sequence_parallel is false but other_memory_pp_on_last is not found in rawConfig");
        }
      }
      
      // Extract and scale parameter sizes
      // model_states_size = 4 * parameter_size
      this.other_parameter_size_pp_off = this.rawConfig[other_layer_type_pp_off][seqLengths[seqLengths.length - 1]].model_states;
      this.other_parameter_size_pp_on_first = this.rawConfig[other_layer_type_pp_on_first][seqLengths[seqLengths.length - 1]].model_states;
      this.other_parameter_size_pp_on_last = this.rawConfig[other_layer_type_pp_on_last][seqLengths[seqLengths.length - 1]].model_states;
      
      // Scale parameter sizes based on sequence length
      for (const key in this.other_parameter_size_pp_off) {
        this.other_parameter_size_pp_off[key] = this.other_parameter_size_pp_off[key] * this.seq_length / seqLengths[seqLengths.length - 1] / 4;
      }
      for (const key in this.other_parameter_size_pp_on_first) {
        this.other_parameter_size_pp_on_first[key] = this.other_parameter_size_pp_on_first[key] * this.seq_length / seqLengths[seqLengths.length - 1] / 4;
      }
      for (const key in this.other_parameter_size_pp_on_last) {
        this.other_parameter_size_pp_on_last[key] = this.other_parameter_size_pp_on_last[key] * this.seq_length / seqLengths[seqLengths.length - 1] / 4;
      }

      // Extract and scale activation sizes
      this.other_activation_size_pp_off = this.rawConfig[other_layer_type_pp_off][seqLengths[seqLengths.length - 1]].activation;
      this.other_activation_size_pp_on_first = this.rawConfig[other_layer_type_pp_on_first][seqLengths[seqLengths.length - 1]].activation;
      this.other_activation_size_pp_on_last = this.rawConfig[other_layer_type_pp_on_last][seqLengths[seqLengths.length - 1]].activation;
      
      // Scale activation sizes based on sequence length
      for (const key in this.other_activation_size_pp_off) {
        this.other_activation_size_pp_off[key] = this.other_activation_size_pp_off[key] * this.seq_length / seqLengths[seqLengths.length - 1];
      }
      for (const key in this.other_activation_size_pp_on_first) {
        this.other_activation_size_pp_on_first[key] = this.other_activation_size_pp_on_first[key] * this.seq_length / seqLengths[seqLengths.length - 1];
      }
      for (const key in this.other_activation_size_pp_on_last) {
        this.other_activation_size_pp_on_last[key] = this.other_activation_size_pp_on_last[key] * this.seq_length / seqLengths[seqLengths.length - 1];
      }

      console.log("Model parameters extracted from config:", this.parameter_size, "MiB");
      console.log("Other parameters extracted from config:", this.other_parameter_size_pp_off, "MiB");
      console.log("Other parameters extracted from config:", this.other_parameter_size_pp_on_first, "MiB");
      console.log("Other parameters extracted from config:", this.other_parameter_size_pp_on_last, "MiB");

      console.log("Activation sizes extracted from config:", this.tp_activation_per_bsz_dict, "MiB");
      console.log("Other activation sizes extracted from config:", this.other_activation_size_pp_off, "MiB");
      console.log("Other activation sizes extracted from config:", this.other_activation_size_pp_on_first, "MiB");
      console.log("Other activation sizes extracted from config:", this.other_activation_size_pp_on_last, "MiB");
    }
    
    // Initialize method (corresponds to Python cost_model.py's initialize method)
    initialize() {
      // Determine sdp_size (based on whether sequence parallel is enabled)
      if (this.use_ulysses) {
        this.sdp_size = this.tp_size * this.dp_size;
      } else {
        this.sdp_size = this.dp_size;
      }
      
      // Calculate local batch size
      this.act_1f1b_ratio = this.stage_idx;
      
      this.num_layers = Math.ceil(this.num_layers / this.pp_size);
      // Initialize ZeRO related ratios
      this.initializeZeroRatios();
    }
    
    // Initialize ZeRO optimization ratios (aligned with Python version)
    initializeZeroRatios() {
      this.zero0_ratio = 1;
      this.zero1_ratio = this.mixed_precision
        ? (d => (6/8 * (1/d) + 2/8))
        : (d => (2/4 * (1/d) + 2/4));
      this.zero2_ratio = this.mixed_precision
        ? (d => (7/8 * (1/d) + 1/8))
        : (d => (3/4 * (1/d) + 1/4));
      this.zero3_ratio = d => (1/d);
      if (this.chunks > 1 && this.mixed_precision) {
        // this.zero0_ratio = 1;
        this.zero1_ratio = (d => (6/9 * (1/d) + 3/9));
        this.zero2_ratio = (d => (9/10 * (1/d) + 1/10));
        // this.zero3_ratio = d => ((1/d) * 5/4);
        // FP32 grad accumulation
      }
    }
    
    // Estimate parameter size (corresponds to Python version's estimate_parameter_size)
    estimateParameterSize() {
      // If sequence parallel is enabled, parameter size remains unchanged
      // Otherwise, parameter size must be divided by tp_size
      if (!this.use_ulysses) {
        this.parameter_size = this.parameter_size / this.tp_size;
      }
      
      console.log(`Estimated single layer parameter size (FP32): ${this.parameter_size} MiB`);
    }
    
    // Estimate model states size (corresponds to Python version's estimate_model_states_size)
    estimateModelStatesSize() {
      // Model states size is 4 times parameter size (corresponds to Python version's estimation method)
      let model_states_size = 4 * this.parameter_size;
      if (this.chunks > 1 && this.mixed_precision) {
        if (this.zero_stage >= 2) model_states_size = model_states_size * 5/4;
        else model_states_size = model_states_size * 9/8;
      }
      
      // Apply ZeRO optimization impact
      if (this.zero_stage === 3) {
        // ZeRO-3: parameter, gradient, and optimizer states are split
        model_states_size = model_states_size * this.zero3_ratio(this.sdp_size);
      } else if (this.zero_stage === 2) {
        // ZeRO-2: gradient and optimizer states are split
        model_states_size = model_states_size * this.zero2_ratio(this.sdp_size);
      }
      else if (this.zero_stage === 1) {
        // ZeRO-1: optimizer states are split
        model_states_size = model_states_size * this.zero1_ratio(this.sdp_size);
      }
      else if (this.zero_stage === 0) {
        // ZeRO-0: not using ZeRO
        model_states_size = model_states_size * this.zero0_ratio;
      }
      
      this.model_states_size = model_states_size;
      console.log(`Estimated model states size: ${this.model_states_size} MiB`);

      if (!this.mixed_precision) {
        this.param_mem = this.parameter_size;
        this.grad_mem = this.parameter_size;
        this.optimizer_mem = 2 * this.parameter_size;
        this.grad_accumulate_mem = 0;
      }
      else {
        this.param_mem = this.parameter_size / 2;
        this.grad_mem = this.parameter_size / 2;
        this.optimizer_mem = 3 * this.parameter_size;
        this.grad_accumulate_mem = this.chunks === 1 ? 0 : this.parameter_size;
      }

      if (this.zero_stage >= 1)
        this.optimizer_mem /= this.sdp_size;
      if (this.zero_stage >= 2)
      {
        this.grad_mem /= this.sdp_size;
        this.grad_accumulate_mem /= this.sdp_size;
      }
      if (this.zero_stage >= 3)
        this.param_mem /= this.sdp_size;
    
      if (this.chunks > 1 && this.zero_stage <= 1) this.grad_mem = 0;

      console.log("debug,grad_mem",this.grad_mem,this.grad_accumulate_mem,this.optimizer_mem,this.param_mem);

      if (Math.abs(this.grad_mem + this.grad_accumulate_mem + this.optimizer_mem + this.param_mem - this.model_states_size) > 1e-6) {
        throw new Error("Memory calculation error!" + 
          "total_mem: " + (this.grad_mem + this.grad_accumulate_mem + this.optimizer_mem + this.param_mem) + 
          "\nmodel_states_size: " + this.model_states_size);
      }
    }
    
    // Estimate activation size (corresponds to Python version's estimate_activation_size)
    estimateActivationSize() {
      // Calculate basic memory requirements for each token
      if (this.checkpoint) {
        this.activation_size = this.tp_activation_per_bsz_dict['checkpoint'] * this.micro_batch_size;
        if (this.sequence_parallel) {
          this.activation_size /= this.tp_size;
        }
      }
      else {
        this.activation_size = this.tp_activation_per_bsz_dict[this.tp_size] * this.micro_batch_size;
      }

      console.log(`Estimated activation size: ${this.activation_size} MiB`);
    }
    
    estimateOtherMemoryModelStatesSize() {
      let other_memory_parameter_size = 0; // Default value
      if (this.pp_size === 1) {
        other_memory_parameter_size = this.other_parameter_size_pp_off[this.tp_size];
      }
      else if (this.stage_idx === 0) {
        other_memory_parameter_size = this.other_parameter_size_pp_on_first[this.tp_size];
      }
      else if (this.stage_idx === this.pp_size - 1) {
        other_memory_parameter_size = this.other_parameter_size_pp_on_last[this.tp_size];
      }
      // model_states_size = 4 * parameter_size
      let other_memory_model_states = other_memory_parameter_size * 4;

      if (this.chunks > 1 && this.mixed_precision) {
        if (this.zero_stage >= 2) other_memory_model_states = other_memory_model_states * 5/4;
        else other_memory_model_states = other_memory_model_states * 9/8;
      }
      
      // Apply ZeRO optimization impact
      if (this.zero_stage === 3) {
        // ZeRO-3: parameter, gradient, and optimizer states are split
        other_memory_model_states = other_memory_model_states * this.zero3_ratio(this.sdp_size);
      } else if (this.zero_stage === 2) {
        // ZeRO-2: gradient and optimizer states are split
        other_memory_model_states = other_memory_model_states * this.zero2_ratio(this.sdp_size);
      }
      else if (this.zero_stage === 1) {
        // ZeRO-1: optimizer states are split
        other_memory_model_states = other_memory_model_states * this.zero1_ratio(this.sdp_size);
      }
      else if (this.zero_stage === 0) {
        // ZeRO-0: not using ZeRO
        other_memory_model_states = other_memory_model_states * this.zero0_ratio;
      }
      
      this.other_memory_model_states = other_memory_model_states;
      console.log(`Estimated other state size: ${this.other_memory_model_states} MiB`);

      
      if (!this.mixed_precision) {
        this.other_param_mem = other_memory_parameter_size;
        this.other_grad_mem = other_memory_parameter_size;
        this.other_optimizer_mem = 2 * other_memory_parameter_size;
        this.other_grad_accumulate_mem = 0;
      }
      else {
        this.other_param_mem = other_memory_parameter_size / 2;
        this.other_grad_mem = other_memory_parameter_size / 2;
        this.other_optimizer_mem = 3 * other_memory_parameter_size;
        this.other_grad_accumulate_mem = this.chunks === 1 ? 0 : other_memory_parameter_size;
      }

      if (this.zero_stage >= 1)
        this.other_optimizer_mem /= this.sdp_size;
      if (this.zero_stage >= 2)
      {
        this.other_grad_mem /= this.sdp_size;
        this.other_grad_accumulate_mem /= this.sdp_size;
      }
      if (this.zero_stage >= 3)
        this.other_param_mem /= this.sdp_size;
    
      if (this.chunks > 1 && this.zero_stage <= 1) this.other_grad_mem = 0;

      if (Math.abs(this.other_grad_mem + this.other_grad_accumulate_mem + this.other_optimizer_mem + this.other_param_mem - this.other_memory_model_states) > 1e-6) {
        throw new Error("Other memory calculation error!" + 
          "total_mem: " + (this.other_grad_mem + this.other_grad_accumulate_mem + this.other_optimizer_mem + this.other_param_mem) + 
          "\nmodel_states_size: " + this.other_memory_model_states);
      }
    }
    // Estimate other memory costs (simplified version, because original Python version's other_memory_cost is complex)
    estimateOtherMemoryActivationSize() {
      // Simplified processing, only consider Python context memory
      // In the actual Python version, there is more complex calculation
      this.other_memory_activation = 0; // Default value
      if (this.pp_size === 1) {
        this.other_memory_activation = this.other_activation_size_pp_off[this.tp_size];
      }
      else if (this.stage_idx === 0) {
        this.other_memory_activation = this.other_activation_size_pp_on_first[this.tp_size];
      }
      else if (this.stage_idx === this.pp_size - 1) {
        this.other_memory_activation = this.other_activation_size_pp_on_last[this.tp_size];
      }
    }
    
    // Calculate total memory and memory ratios for each component
    calculateMemory() {
      // Calculate gradient memory (corresponds to part of model states)
      this.total_model_states = this.num_layers * this.model_states_size + this.other_memory_model_states;
      this.total_activation = this.num_layers * this.activation_size + this.other_memory_activation;

      this.total_activation = (this.pp_size - this.act_1f1b_ratio - 1) * this.num_layers * this.activation_size + this.total_activation;

      this.total_param_mem = this.num_layers * this.param_mem + this.other_param_mem;
      this.total_grad_mem = this.num_layers * this.grad_mem + this.other_grad_mem;
      this.total_optimizer_mem = this.num_layers * this.optimizer_mem + this.other_optimizer_mem;
      this.total_grad_accumulate_mem = this.num_layers * this.grad_accumulate_mem + this.other_grad_accumulate_mem;

      this.total_mem = this.total_model_states + this.total_activation;
                       
      // Analyze memory component ratios
      const total = this.total_mem;
      this.param_percentage = (this.total_param_mem / total) * 100;
      this.grad_percentage = (this.total_grad_mem / total) * 100;
      this.optimizer_percentage = (this.total_optimizer_mem / total) * 100;
      this.grad_accumulate_percentage = (this.total_grad_accumulate_mem / total) * 100;
      this.activation_percentage = (this.total_activation / total) * 100;
      // this.other_percentage = (this.other_memory_cost / total) * 100;
      
      console.log("Memory calculation completed:", {
        parameter: this.total_param_mem,
        gradient: this.total_grad_mem,
        grad_accumulate: this.total_grad_accumulate_mem,
        optimizer: this.total_optimizer_mem,
        activation: this.total_activation,
        total: this.total_mem
      });
    }
    
    // Get memory cost (corresponds to Python version's get_memory_cost method)
    getMemoryCost() {
      // Ensure memory calculation is done first
      this.calculateMemory();
      
      console.log("Starting memory cost calculation...");
      
      // Convert unit from MiB to MB for easier frontend display
      // 1 MiB ≈ 1.048576 MB, but here simplified to approximately equal 1
      const results = {
        num_layers: this.num_layers,
        stage_idx: this.stage_idx,
        model_states: this.total_model_states,
        parameter: this.total_param_mem,
        gradient: this.total_grad_mem,
        grad_accumulate: this.total_grad_accumulate_mem,
        optimizer: this.total_optimizer_mem,
        activation: this.total_activation,
        per_layer_parameter: this.param_mem,
        per_layer_activation: this.activation_size,
        other_memory_model_states: this.other_memory_model_states,
        other_memory_parameter: this.other_param_mem,
        other_memory_gradient: this.other_grad_mem,
        other_memory_grad_accumulate: this.other_grad_accumulate_mem,
        other_memory_optimizer: this.other_optimizer_mem,
        other_memory_activation: this.other_memory_activation,
        total: this.total_mem,
      };
      
      console.log("Basic memory cost calculation completed:", results);
      
      return results;
    }
  }