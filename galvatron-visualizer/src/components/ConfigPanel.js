import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { 
  Slider, TextField, Typography, Box, Paper, 
  Chip, Divider, Grid, Select, MenuItem, FormControl,
  Alert, Switch, FormControlLabel
} from '@mui/material';
import { useLanguage } from '../context/LanguageContext';
import { getText } from '../utils/translations';

// Default slider ranges
const DEFAULT_RANGES = {
  micro_batch_size: { min: 1, max: 64 },
  seq_length: { min: 64, max: 4096 },
  tp_size: { min: 1, max: 16 },
  pp_size: { min: 1, max: 16 },
  global_batch_size: { min: 1, max: 1024 },
  zero_stage: { min: 0, max: 3 },
  chunks: { min: 1, max: 32 },
  total_gpus: { min: 1, max: 512 },
  num_layers: { min: 1, max: 100 }
};

function ConfigPanel({ config, onConfigChange }) {
  const { language } = useLanguage();
  
  // Slider ranges state
  const [ranges, setRanges] = useState(DEFAULT_RANGES);
  
  // Use useMemo to cache the config string
  const configString = useMemo(() => JSON.stringify(config), [config]);
  
  // Update slider ranges based on current config values
  useEffect(() => {
    // Prevent unnecessary state updates
    let needsUpdate = false;
    const newRanges = { ...ranges };
    
    // Adjust max ranges if current values exceed them
    const adjustMax = (key, factor = 1.2) => {
      if (config[key] > newRanges[key].max) {
        newRanges[key].max = Math.ceil(config[key] * factor);
        needsUpdate = true;
      }
    };
    
    adjustMax('seq_length');
    adjustMax('micro_batch_size');
    adjustMax('global_batch_size');
    adjustMax('chunks');
    adjustMax('total_gpus');
    adjustMax('num_layers');
    
    // Only call setRanges if there are actual changes
    if (needsUpdate) {
      setRanges(newRanges);
    }
  }, [configString]); // Use configString instead of config and ranges
  
  // Use useMemo to cache the constraints
  const constraints = useMemo(() => {
    const { tp_size, pp_size, dp_size, total_gpus, micro_batch_size, global_batch_size, chunks } = config;
    
    const gpuConstraint = tp_size * pp_size * dp_size === total_gpus;
    const batchConstraint = micro_batch_size * dp_size * chunks === global_batch_size;
    
    return { 
      gpuConstraint, 
      batchConstraint,
      gpuProduct: tp_size * pp_size * dp_size,
      batchProduct: micro_batch_size * dp_size * chunks
    };
  }, [configString]);
  
  // Handle config parameter changes - use useCallback to optimize
  const handleChange = useCallback((key, value) => {
    const newConfig = { ...config, [key]: value };
    
    // Auto-update related parameters to satisfy constraints
    if (key === 'total_gpus') {
      // Adjust dp_size to satisfy GPU constraint
      newConfig.dp_size = Math.max(1, Math.floor(value / (config.tp_size * config.pp_size)));
      
      // Update chunks to satisfy batch constraint
      if (newConfig.dp_size > 0 && newConfig.micro_batch_size > 0) {
        newConfig.chunks = Math.max(1, Math.floor(newConfig.global_batch_size / (newConfig.dp_size * newConfig.micro_batch_size)));
      }
    } else if (key === 'tp_size' || key === 'pp_size') {
      // Adjust dp_size to satisfy GPU constraint
      const product = (key === 'tp_size' ? value : config.tp_size) * (key === 'pp_size' ? value : config.pp_size);
      if (product > 0 && config.total_gpus > 0) {
        newConfig.dp_size = Math.max(1, Math.floor(config.total_gpus / product));
        
        // Update chunks to satisfy batch constraint
        if (newConfig.dp_size > 0 && newConfig.micro_batch_size > 0) {
          newConfig.chunks = Math.max(1, Math.floor(newConfig.global_batch_size / (newConfig.dp_size * newConfig.micro_batch_size)));
        }
      }
    } else if (key === 'micro_batch_size' || key === 'global_batch_size') {
      // Adjust chunks to satisfy batch constraint
      if (newConfig.dp_size > 0 && (key === 'micro_batch_size' ? value : config.micro_batch_size) > 0) {
        newConfig.chunks = Math.max(1, Math.floor((key === 'global_batch_size' ? value : config.global_batch_size) / 
          (newConfig.dp_size * (key === 'micro_batch_size' ? value : config.micro_batch_size))));
      }
    } else if (key === 'chunks') {
      // Update global_batch_size based on chunks
      newConfig.global_batch_size = newConfig.chunks * newConfig.dp_size * newConfig.micro_batch_size;
    } else if (key === 'dp_size') {
      // Adjust chunks to satisfy batch constraint
      if (value > 0 && newConfig.micro_batch_size > 0) {
        newConfig.chunks = Math.max(1, Math.floor(newConfig.global_batch_size / (value * newConfig.micro_batch_size)));
      }
    }
    
    onConfigChange(newConfig);
  }, [config, onConfigChange]);
  
  // Use useCallback to optimize the switch handling function
  const handleSwitchChange = useCallback((key) => (event) => {
    handleChange(key, event.target.checked);
  }, [handleChange]);
  
  // Render fixed parameter chips
  const renderFixedParams = () => (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
      <Chip 
        label={`${getText('attentionHeads', language)}: ${config.attention_heads}`} 
        color="primary" 
        variant="outlined"
      />
      <Chip 
        label={`${getText('hiddenDim', language)}: ${config.hidden_dim}`} 
        color="primary" 
        variant="outlined"
      />
      <Chip 
        label={`${getText('ffnDim', language)}: ${config.ff_dim}`} 
        color="primary" 
        variant="outlined"
      />
      <Chip 
        label={`${getText('vocabSize', language)}: ${config.vocab_size}`} 
        color="primary" 
        variant="outlined"
      />
      <Chip 
        label={`${getText('mixedPrecision', language)}: ${config.mixed_precision ? getText('enabled', language) : getText('disabled', language)}`}
        color={config.mixed_precision ? "success" : "default"}
        variant="outlined"
        // onClick={() => handleChange('mixed_precision', !config.mixed_precision)}
        // sx={{ cursor: 'pointer' }}
      />
    </Box>
  );
  
  // Render switch options
  const renderSwitchOptions = () => (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 2 }}>
      <FormControlLabel
        control={
          <Switch
            checked={config.sequence_parallel || false}
            onChange={handleSwitchChange('sequence_parallel')}
            color="success"
          />
        }
        label={getText('seqParallel', language)}
      />
      
      <FormControlLabel
        control={
          <Switch
            checked={config.checkpoint || false}
            onChange={handleSwitchChange('checkpoint')}
            color="success"
          />
        }
        label={getText('activationCheckpoint', language)}
      />
    </Box>
  );
  
  // Render a slider with text input
  const renderSlider = (title, key, step = 1, marks = null) => (
    <Box sx={{ mb: 2 }}>
      <Typography gutterBottom>{title}</Typography>
      <Slider
        value={config[key] || 1}
        min={ranges[key]?.min || 1}
        max={ranges[key]?.max || 100}
        step={step}
        onChange={(_, value) => handleChange(key, value)}
        valueLabelDisplay="auto"
        marks={marks}
      />
      <TextField 
        size="small"
        value={config[key] || 1}
        onChange={e => handleChange(key, Number(e.target.value))}
        type="number"
        variant="outlined"
      />
    </Box>
  );
  
  // Prepare marks for layers slider
  const getLayerMarks = () => [
    { value: 1, label: '1' },
    { value: Math.min(12, ranges.num_layers.max), label: '12' },
    { value: Math.min(24, ranges.num_layers.max), label: '24' },
    { value: ranges.num_layers.max, label: `${ranges.num_layers.max}` }
  ];
  
  // Render PP stage slider (only when pp_size > 1)
  const renderPpStageSlider = () => {
    if (config.pp_size <= 1) return null;
    
    return (
      <Grid item xs={12} md={6}>
        <Box sx={{ mb: 2 }}>
          <Typography gutterBottom>{getText('ppStage', language)}:</Typography>
          <Slider
            value={config.stage_idx !== undefined ? config.stage_idx : 0}
            min={0}
            max={config.pp_size - 1}
            step={1}
            onChange={(_, value) => handleChange('stage_idx', value)}
            valueLabelDisplay="auto"
            marks={Array.from({length: config.pp_size}, (_, i) => ({value: i, label: `${i}`}))}
          />
          <Typography variant="caption" color="text.secondary">
            {getText('currentStage', language)}: {config.stage_idx !== undefined ? config.stage_idx : 0} / {config.pp_size - 1}
          </Typography>
        </Box>
      </Grid>
    );
  };
  
  // Render constraint warnings
  const renderConstraintWarnings = () => {
    if (constraints.gpuConstraint && constraints.batchConstraint) return null;
    
    return (
      <Box sx={{ mt: 2 }}>
        {!constraints.gpuConstraint && (
          <Alert severity="warning" sx={{ mb: 1 }}>
            <Typography variant="subtitle2">
              {getText('gpuConstraint', language)}
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <Typography variant="body2">
                {getText('current', language)}: {constraints.gpuProduct}
              </Typography>
              <Typography variant="body2">
                {getText('expected', language)}: {config.total_gpus}
              </Typography>
            </Box>
          </Alert>
        )}
        
        {!constraints.batchConstraint && (
          <Alert severity="warning" sx={{ mb: 1 }}>
            <Typography variant="subtitle2">
              {getText('batchConstraint', language)}
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <Typography variant="body2">
                {getText('current', language)}: {constraints.batchProduct}
              </Typography>
              <Typography variant="body2">
                {getText('expected', language)}: {config.global_batch_size}
              </Typography>
            </Box>
          </Alert>
        )}
      </Box>
    );
  };
  
  return (
    <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        {getText('configPanelTitle', language)}
      </Typography>
      
      <Divider sx={{ mb: 2 }} />
      
      <Typography variant="subtitle1" gutterBottom>
        {getText('basicParams', language)}
      </Typography>
      
      {renderFixedParams()}
      
      {renderSwitchOptions()}
      
      <Grid container spacing={3} sx={{ mt: 2 }}>
        {/* Model layers slider */}
        <Grid item xs={12} md={6}>
          {renderSlider(`${getText('modelLayers', language)}:`, 'num_layers', 1, getLayerMarks())}
        </Grid>
        
        {/* Sequence length slider */}
        <Grid item xs={12} md={6}>
          {renderSlider(`${getText('seqLength', language)}:`, 'seq_length', 16)}
        </Grid>
        
        {/* Micro batch size slider */}
        <Grid item xs={12} md={6}>
          {renderSlider(`${getText('microBatchSize', language)}:`, 'micro_batch_size')}
        </Grid>
        
        {/* Global batch size slider */}
        <Grid item xs={12} md={6}>
          {renderSlider(`${getText('globalBatchSize', language)}:`, 'global_batch_size', 1)}
        </Grid>

        {/* Chunks slider */}
        <Grid item xs={12} md={6}>
          {renderSlider(`${getText('chunks', language) || 'Chunks'}:`, 'chunks')}
        </Grid>
      </Grid>
      
      {renderConstraintWarnings()}
      
      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>{getText('parallelStrategies', language)}</Typography>
      
      <Grid container spacing={3}>
        {/* Total GPUs slider */}
        <Grid item xs={12} md={6}>
          {renderSlider(`${getText('totalGPUs', language)}:`, 'total_gpus')}
        </Grid>
        
        {/* DP size display (auto-calculated) */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>{getText('dataParallel', language)}:</Typography>
            <Chip label={`DP: ${config.dp_size}`} color="primary" />
            <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
              {getText('dpAutoCalc', language)}
            </Typography>
          </Box>
        </Grid>
        
        {/* TP size slider */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>{getText('tensorParallel', language)}:</Typography>
            <Slider
              value={config.tp_size}
              min={ranges.tp_size.min}
              max={ranges.tp_size.max}
              step={1}
              onChange={(_, value) => handleChange('tp_size', value)}
              valueLabelDisplay="auto"
              marks={[{value: 1, label: '1'}, {value: 8, label: '8'}]}
            />
          </Box>
        </Grid>
        
        {/* PP size slider */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>{getText('pipelineParallel', language)}:</Typography>
            <Slider
              value={config.pp_size}
              min={ranges.pp_size.min}
              max={ranges.pp_size.max}
              step={1}
              onChange={(_, value) => handleChange('pp_size', value)}
              valueLabelDisplay="auto"
              marks={[{value: 1, label: '1'}, {value: 8, label: '8'}]}
            />
          </Box>
        </Grid>

        {/* ZeRO stage selector */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>{getText('zeroStage', language)}:</Typography>
            <FormControl fullWidth size="small">
              <Select
                value={config.zero_stage || 0}
                onChange={e => handleChange('zero_stage', Number(e.target.value))}
              >
                <MenuItem value={0}>{getText('zeroOff', language)}</MenuItem>
                <MenuItem value={1}>{getText('zeroLevel', language)} 1</MenuItem>
                <MenuItem value={2}>{getText('zeroLevel', language)} 2</MenuItem>
                <MenuItem value={3}>{getText('zeroLevel', language)} 3</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Grid>
        
        {/* PP stage slider (conditional) */}
        {renderPpStageSlider()}
        
        {/* Footer notes */}
        <Grid item xs={12}>
          <Divider sx={{ my: 1 }} />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            {getText('parallelNote', language)}
          </Typography>
        </Grid>
      </Grid>
    </Paper>
  );
}

export default ConfigPanel;