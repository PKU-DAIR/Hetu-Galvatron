import React, { useEffect, useState, useMemo, useCallback } from 'react';
import {
  Slider, TextField, Typography, Box, Paper,
  Chip, Divider, Grid, Select, MenuItem, FormControl,
  Alert, Switch, FormControlLabel
} from '@mui/material';
import { useLanguage } from '../context/LanguageContext';
import { getText } from '../utils/translations';
import { applyDevicePreset, getDeviceCount } from '../utils/configParser';

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

function ConfigPanel({ config, onConfigChange, onJsonConfigLoaded }) {
  const { language } = useLanguage();

  const [ranges, setRanges] = useState(DEFAULT_RANGES);
  const configString = useMemo(() => JSON.stringify(config), [config]);

  useEffect(() => {
    let needsUpdate = false;
    const newRanges = { ...ranges };
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
    if (needsUpdate) {
      setRanges(newRanges);
    }
  }, [configString, config, ranges]);

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
  }, [configString, config]);

  const handleChange = useCallback((key, value) => {
    const newConfig = { ...config, [key]: value };
    if (key === 'total_gpus') {
      newConfig.dp_size = Math.max(1, Math.floor(value / (config.tp_size * config.pp_size)));
      if (newConfig.dp_size > 0 && newConfig.micro_batch_size > 0) {
        newConfig.chunks = Math.max(1, Math.floor(newConfig.global_batch_size / (newConfig.dp_size * newConfig.micro_batch_size)));
      }
    } else if (key === 'tp_size' || key === 'pp_size') {
      const product = (key === 'tp_size' ? value : config.tp_size) * (key === 'pp_size' ? value : config.pp_size);
      if (product > 0 && config.total_gpus > 0) {
        newConfig.dp_size = Math.max(1, Math.floor(config.total_gpus / product));
        if (newConfig.dp_size > 0 && newConfig.micro_batch_size > 0) {
          newConfig.chunks = Math.max(1, Math.floor(newConfig.global_batch_size / (newConfig.dp_size * newConfig.micro_batch_size)));
        }
      }
    } else if (key === 'micro_batch_size' || key === 'global_batch_size') {
      if (newConfig.dp_size > 0 && (key === 'micro_batch_size' ? value : config.micro_batch_size) > 0) {
        newConfig.chunks = Math.max(1, Math.floor((key === 'global_batch_size' ? value : config.global_batch_size) /
          (newConfig.dp_size * (key === 'micro_batch_size' ? value : config.micro_batch_size))));
      }
    } else if (key === 'chunks') {
      newConfig.global_batch_size = newConfig.chunks * newConfig.dp_size * newConfig.micro_batch_size;
    } else if (key === 'dp_size') {
      if (value > 0 && newConfig.micro_batch_size > 0) {
        newConfig.chunks = Math.max(1, Math.floor(newConfig.global_batch_size / (value * newConfig.micro_batch_size)));
      }
    }
    onConfigChange(newConfig);
  }, [config, onConfigChange]);

  const handleSwitchChange = useCallback((key) => (event) => {
    handleChange(key, event.target.checked);
  }, [handleChange]);

  // Handle hardware preset change
  const handleHardwarePresetChange = useCallback((presetName) => {
    const updatedConfig = applyDevicePreset(config, presetName);
    onConfigChange(updatedConfig);
  }, [config, onConfigChange]);

  const renderFixedParams = () => (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
      <Chip label={`${getText('attentionHeads', language)}: ${config.attention_heads}`} color="primary" variant="outlined" />
      <Chip label={`${getText('hiddenDim', language)}: ${config.hidden_dim}`} color="primary" variant="outlined" />
      <Chip label={`${getText('ffnDim', language)}: ${config.ff_dim}`} color="primary" variant="outlined" />
      <Chip label={`${getText('vocabSize', language)}: ${config.vocab_size}`} color="primary" variant="outlined" />
      <Chip label={`${getText('mixedPrecision', language)}: ${config.mixed_precision ? getText('enabled', language) : getText('disabled', language)}`} color={config.mixed_precision ? "success" : "default"} variant="outlined" />
    </Box>
  );

  const renderSwitchOptions = () => (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 2 }}>
      <FormControlLabel control={<Switch checked={config.sequence_parallel || false} onChange={handleSwitchChange('sequence_parallel')} color="success" />} label={getText('seqParallel', language)} />
      <FormControlLabel control={<Switch checked={config.checkpoint || false} onChange={handleSwitchChange('checkpoint')} color="success" />} label={getText('activationCheckpoint', language)} />
    </Box>
  );

  // Common sx for number TextFields to hide spinners
  const numberInputSx = {
    width: '70px', // Adjusted width
    '& input[type="number"]::-webkit-outer-spin-button, & input[type="number"]::-webkit-inner-spin-button': {
      WebkitAppearance: 'none',
      margin: 0,
    },
    '& input[type="number"]': {
      MozAppearance: 'textfield',
    },
  };

  const renderHardwarePerformance = () => (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>关键硬件性能参数 (Hardware Performance)</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>硬件类型预设:</Typography>
            <FormControl fullWidth size="small">
              <Select 
                value={config.hardware_preset || 'A100'} 
                onChange={e => handleHardwarePresetChange(e.target.value)}
              >
                <MenuItem value="A100">NVIDIA A100</MenuItem>
                <MenuItem value="H100">NVIDIA H100</MenuItem>
                <MenuItem value="V100">NVIDIA V100</MenuItem>
                <MenuItem value="Custom">自定义</MenuItem>
              </Select>
            </FormControl>
          </Box>
          
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>设备数量:</Typography>
            <FormControl fullWidth size="small">
              <Select 
                value={config.device_count || getDeviceCount(config)} 
                onChange={e => handleChange('device_count', e.target.value)}
              >
                <MenuItem value={1}>1 设备</MenuItem>
                <MenuItem value={2}>2 设备</MenuItem>
                <MenuItem value={4}>4 设备</MenuItem>
                <MenuItem value={8}>8 设备</MenuItem>
                <MenuItem value={16}>16 设备</MenuItem>
                <MenuItem value={32}>32 设备</MenuItem>
                <MenuItem value={64}>64 设备</MenuItem>
                <MenuItem value={128}>128 设备</MenuItem>
                <MenuItem value="custom">自定义</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Grid>

        {/* Forward Computation Time */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Typography variant="body2" sx={{ flexGrow: 1, mr: 1 }}>前向计算时间 (ms):</Typography>
              <TextField size="small" value={config.forward_computation_time || 10} onChange={e => handleChange('forward_computation_time', Number(e.target.value))} type="number" variant="outlined" sx={numberInputSx} />
            </Box>
            <Slider value={config.forward_computation_time || 10} min={1} max={1000} step={1} onChange={(_, value) => handleChange('forward_computation_time', value)} valueLabelDisplay="auto" />
          </Box>
        </Grid>

        {/* BCT/FCT Coefficient */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Typography variant="body2" sx={{ flexGrow: 1, mr: 1 }}>后向/前向计算比例:</Typography>
              <TextField size="small" value={config.bct_fct_coe || 2.0} onChange={e => handleChange('bct_fct_coe', Number(e.target.value))} type="number" variant="outlined" sx={numberInputSx} inputProps={{ step: 0.1 }} />
            </Box>
            <Slider value={config.bct_fct_coe || 2.0} min={1.5} max={3.0} step={0.1} onChange={(_, value) => handleChange('bct_fct_coe', value)} valueLabelDisplay="auto" marks={[{ value: 1.5, label: '1.5' }, { value: 2.0, label: '2.0' }, { value: 3.0, label: '3.0' }]} />
          </Box>
        </Grid>

        {/* DP Overlap Coefficient */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Typography variant="body2" sx={{ flexGrow: 1, mr: 1 }}>DP重叠系数:</Typography>
              <TextField size="small" value={config.dp_overlap_coe || 1.0} onChange={e => handleChange('dp_overlap_coe', Number(e.target.value))} type="number" variant="outlined" sx={numberInputSx} inputProps={{ step: 0.1 }} />
            </Box>
            <Slider value={config.dp_overlap_coe || 1.0} min={0.8} max={2.0} step={0.1} onChange={(_, value) => handleChange('dp_overlap_coe', value)} valueLabelDisplay="auto" marks={[{ value: 0.8, label: '0.8' }, { value: 1.0, label: '1.0' }, { value: 2.0, label: '2.0' }]} />
          </Box>
        </Grid>

        {/* BCT Overlap Coefficient */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Typography variant="body2" sx={{ flexGrow: 1, mr: 1 }}>BCT重叠系数:</Typography>
              <TextField size="small" value={config.bct_overlap_coe || 1.0} onChange={e => handleChange('bct_overlap_coe', Number(e.target.value))} type="number" variant="outlined" sx={numberInputSx} inputProps={{ step: 0.1 }} />
            </Box>
            <Slider value={config.bct_overlap_coe || 1.0} min={0.8} max={2.0} step={0.1} onChange={(_, value) => handleChange('bct_overlap_coe', value)} valueLabelDisplay="auto" marks={[{ value: 0.8, label: '0.8' }, { value: 1.0, label: '1.0' }, { value: 2.0, label: '2.0' }]} />
          </Box>
        </Grid>


        {/* AllReduce Bandwidth */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Typography variant="body2" sx={{ flexGrow: 1, mr: 1 }}>AllReduce带宽 (GB/s):</Typography>
              <TextField size="small" value={config.allreduce_bandwidth || 100} onChange={e => handleChange('allreduce_bandwidth', Number(e.target.value))} type="number" variant="outlined" sx={numberInputSx} />
            </Box>
            <Slider value={config.allreduce_bandwidth || 100} min={50} max={200} step={5} onChange={(_, value) => handleChange('allreduce_bandwidth', value)} valueLabelDisplay="auto" marks={[{ value: 50, label: '50' }, { value: 100, label: '100' }, { value: 200, label: '200' }]} />
          </Box>
        </Grid>

        {/* P2P Bandwidth */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Typography variant="body2" sx={{ flexGrow: 1, mr: 1 }}>P2P带宽 (GB/s):</Typography>
              <TextField size="small" value={config.p2p_bandwidth || 300} onChange={e => handleChange('p2p_bandwidth', Number(e.target.value))} type="number" variant="outlined" sx={numberInputSx} />
            </Box>
            <Slider value={config.p2p_bandwidth || 300} min={100} max={400} step={10} onChange={(_, value) => handleChange('p2p_bandwidth', value)} valueLabelDisplay="auto" marks={[{ value: 100, label: '100' }, { value: 300, label: '300' }, { value: 400, label: '400' }]} />
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>序列并行空间:</Typography>
            <FormControl fullWidth size="small">
              <Select value={config.sp_space || 'tp+sp'} onChange={e => handleChange('sp_space', e.target.value)}>
                <MenuItem value="tp+sp">TP+SP</MenuItem>
                <MenuItem value="sp+allreduce">SP+AllReduce</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <FormControlLabel control={<Switch checked={config.async_grad_reduce || false} onChange={handleSwitchChange('async_grad_reduce')} color="primary" />} label="异步梯度归约" />
          </Box>
        </Grid>
      </Grid>
    </Box>
  );

  const renderSlider = (title, key, step = 1, marks = null) => (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
        <Typography variant="body2" sx={{ flexGrow: 1, mr: 1 }}>{title}</Typography>
        <TextField
          size="small"
          value={config[key] || (ranges[key]?.min || 1) } // Ensure initial value respects min
          onChange={e => {
            let val = Number(e.target.value);
            // Basic validation against min/max if desired, though slider handles this mostly
            // if (val < (ranges[key]?.min || 1)) val = ranges[key]?.min || 1;
            // if (val > (ranges[key]?.max || 100)) val = ranges[key]?.max || 100;
            handleChange(key, val);
          }}
          type="number"
          variant="outlined"
          sx={numberInputSx} // Apply common sx here
          inputProps={{ step: step }}
        />
      </Box>
      <Slider
        value={typeof config[key] === 'number' ? config[key] : (ranges[key]?.min || 1)}
        min={ranges[key]?.min || 1}
        max={ranges[key]?.max || 100}
        step={step}
        onChange={(_, value) => handleChange(key, value)}
        valueLabelDisplay="auto"
        marks={marks}
      />
    </Box>
  );

  const getLayerMarks = () => [
    { value: 1, label: '1' },
    { value: Math.min(12, ranges.num_layers.max), label: '12' },
    { value: Math.min(24, ranges.num_layers.max), label: '24' },
    { value: ranges.num_layers.max, label: `${ranges.num_layers.max}` }
  ];

  const renderPpStageSlider = () => {
    if (config.pp_size <= 1) return null;
    return (
      <Grid item xs={12} md={6}>
        <Box sx={{ mb: 2 }}>
          <Typography gutterBottom>{getText('ppStage', language)}:</Typography>
          <Slider value={config.stage_idx !== undefined ? config.stage_idx : 0} min={0} max={config.pp_size - 1} step={1} onChange={(_, value) => handleChange('stage_idx', value)} valueLabelDisplay="auto" marks={Array.from({ length: config.pp_size }, (_, i) => ({ value: i, label: `${i}` }))} />
          <Typography variant="caption" color="text.secondary">{getText('currentStage', language)}: {config.stage_idx !== undefined ? config.stage_idx : 0} / {config.pp_size - 1}</Typography>
        </Box>
      </Grid>
    );
  };

  const renderConstraintWarnings = () => {
    if (constraints.gpuConstraint && constraints.batchConstraint) return null;
    return (
      <Box sx={{ mt: 2 }}>
        {!constraints.gpuConstraint && (
          <Alert severity="warning" sx={{ mb: 1 }}>
            <Typography variant="subtitle2">{getText('gpuConstraint', language)}</Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <Typography variant="body2">{getText('current', language)}: {constraints.gpuProduct}</Typography>
              <Typography variant="body2">{getText('expected', language)}: {config.total_gpus}</Typography>
            </Box>
          </Alert>
        )}
        {!constraints.batchConstraint && (
          <Alert severity="warning" sx={{ mb: 1 }}>
            <Typography variant="subtitle2">{getText('batchConstraint', language)}</Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <Typography variant="body2">{getText('current', language)}: {constraints.batchProduct}</Typography>
              <Typography variant="body2">{getText('expected', language)}: {config.global_batch_size}</Typography>
            </Box>
          </Alert>
        )}
      </Box>
    );
  };

  return (
    <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
      <Typography variant="h6" gutterBottom>{getText('configPanelTitle', language)}</Typography>
      <Divider sx={{ mb: 2 }} />
      <Typography variant="subtitle1" gutterBottom>{getText('basicParams', language)}</Typography>
      {renderFixedParams()}
      {renderSwitchOptions()}
      {renderHardwarePerformance()}
      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6}>{renderSlider(`${getText('modelLayers', language)}:`, 'num_layers', 1, getLayerMarks())}</Grid>
        <Grid item xs={12} md={6}>{renderSlider(`${getText('seqLength', language)}:`, 'seq_length', 16)}</Grid>
        <Grid item xs={12} md={6}>{renderSlider(`${getText('microBatchSize', language)}:`, 'micro_batch_size')}</Grid>
        <Grid item xs={12} md={6}>{renderSlider(`${getText('globalBatchSize', language)}:`, 'global_batch_size', 1)}</Grid>
        <Grid item xs={12} md={6}>{renderSlider(`${getText('chunks', language) || 'Chunks'}:`, 'chunks')}</Grid>
      </Grid>
      {renderConstraintWarnings()}
      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>{getText('parallelStrategies', language)}</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>{renderSlider(`${getText('totalGPUs', language)}:`, 'total_gpus')}</Grid>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>{getText('dataParallel', language)}:</Typography>
            <Chip label={`DP: ${config.dp_size}`} color="primary" />
            <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>{getText('dpAutoCalc', language)}</Typography>
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>{getText('tensorParallel', language)}:</Typography>
            <Slider value={config.tp_size} min={ranges.tp_size.min} max={ranges.tp_size.max} step={1} onChange={(_, value) => handleChange('tp_size', value)} valueLabelDisplay="auto" marks={[{ value: 1, label: '1' }, { value: 8, label: '8' }]} />
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>{getText('pipelineParallel', language)}:</Typography>
            <Slider value={config.pp_size} min={ranges.pp_size.min} max={ranges.pp_size.max} step={1} onChange={(_, value) => handleChange('pp_size', value)} valueLabelDisplay="auto" marks={[{ value: 1, label: '1' }, { value: 8, label: '8' }]} />
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>{getText('zeroStage', language)}:</Typography>
            <FormControl fullWidth size="small">
              <Select value={config.zero_stage || 0} onChange={e => handleChange('zero_stage', Number(e.target.value))}>
                <MenuItem value={0}>{getText('zeroOff', language)}</MenuItem>
                <MenuItem value={1}>{getText('zeroLevel', language)} 1</MenuItem>
                <MenuItem value={2}>{getText('zeroLevel', language)} 2</MenuItem>
                <MenuItem value={3}>{getText('zeroLevel', language)} 3</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Grid>
        {renderPpStageSlider()}
        <Grid item xs={12}>
          <Divider sx={{ my: 1 }} />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>{getText('parallelNote', language)}</Typography>
        </Grid>
      </Grid>
    </Paper>
  );
}

export default ConfigPanel;