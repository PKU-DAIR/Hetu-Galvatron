import React from 'react';
import { 
  Paper, Typography, FormControl, Box, Select, 
  MenuItem, Grid, Divider, Chip, Switch, FormControlLabel,
  Tooltip
} from '@mui/material';
import { useLanguage } from '../context/LanguageContext';
import { getText } from '../utils/translations';

// Pre-defined device types with relative performance factors
// These factors represent approximate relative computation speed
// Higher numbers mean faster computation (e.g. 2.0 means twice as fast as baseline)
const DEVICE_TYPES = [
  { id: '4090', name: 'NVIDIA RTX 4090', factor: 0.8 },
  { id: 'a100', name: 'NVIDIA A100', factor: 1.0 },
  { id: 'h100', name: 'NVIDIA H100', factor: 1.8 },
  { id: 'a10g', name: 'NVIDIA A10G', factor: 0.5 },
  { id: 'v100', name: 'NVIDIA V100', factor: 0.7 },
  { id: 't4', name: 'NVIDIA T4', factor: 0.3 },
  { id: 'a6000', name: 'NVIDIA A6000', factor: 0.6 },
];

// Pre-defined device counts
const DEVICE_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128];

// Communication efficiency factors based on device count
// These represent how communication efficiency changes with scale
// Values < 1.0 mean efficiency loss at scale
const COMM_EFFICIENCY = {
  1: 1.0,    // Single GPU has no communication overhead
  2: 0.95,   // 2 GPUs have 95% efficiency
  4: 0.9,    // 4 GPUs have 90% efficiency
  8: 0.85,   // etc.
  16: 0.8,
  32: 0.75,
  64: 0.7,
  128: 0.65,
};

function DeviceSelector({ deviceConfig, onDeviceConfigChange, useBackendCalc, onToggleBackendCalc }) {
  const { language } = useLanguage();
  
  // Handle device type change
  const handleDeviceTypeChange = (event) => {
    const deviceType = event.target.value;
    const deviceFactor = DEVICE_TYPES.find(d => d.id === deviceType)?.factor || 1.0;
    
    onDeviceConfigChange({
      ...deviceConfig,
      deviceType,
      deviceFactor
    });
  };
  
  // Handle device count change
  const handleDeviceCountChange = (event) => {
    const deviceCount = event.target.value;
    const commEfficiency = COMM_EFFICIENCY[deviceCount] || 1.0;
    
    onDeviceConfigChange({
      ...deviceConfig,
      deviceCount,
      commEfficiency
    });
  };
  
  // Handle calculation method toggle
  const handleCalcToggle = (event) => {
    onToggleBackendCalc(event.target.checked);
  };
  
  return (
    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        {getText('deviceSelector', language)}
      </Typography>
      
      <Divider sx={{ mb: 2 }} />
      
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <Typography variant="body2" gutterBottom>
              {getText('deviceType', language)}
            </Typography>
            <Select
              value={deviceConfig.deviceType}
              onChange={handleDeviceTypeChange}
            >
              {DEVICE_TYPES.map(device => (
                <MenuItem key={device.id} value={device.id}>
                  {device.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <Typography variant="body2" gutterBottom>
              {getText('deviceCount', language)}
            </Typography>
            <Select
              value={deviceConfig.deviceCount}
              onChange={handleDeviceCountChange}
            >
              {DEVICE_COUNTS.map(count => (
                <MenuItem key={count} value={count}>
                  {count} {count === 1 ? getText('deviceSingular', language) : getText('devicePlural', language)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
      </Grid>
      
      <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
        <Chip 
          label={`${getText('computationFactor', language)}: ${deviceConfig.deviceFactor.toFixed(2)}x`} 
          color="primary" 
          variant="outlined"
          size="small"
        />
        <Chip 
          label={`${getText('commEfficiency', language)}: ${(deviceConfig.commEfficiency * 100).toFixed(0)}%`} 
          color="secondary" 
          variant="outlined"
          size="small"
        />
      </Box>
      
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
        {getText('deviceSelectorNote', language)}
      </Typography>
      
      {deviceConfig.deviceType === '4090' && (
        <Typography variant="caption" color="info.main" sx={{ display: 'block', mt: 1 }}>
          {getText('rtx4090Note', language)}
        </Typography>
      )}
      
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
        <Tooltip title={useBackendCalc ? getText('useBackendCalc', language) : getText('useFrontendCalc', language)}>
          <FormControlLabel
            control={
              <Switch
                checked={useBackendCalc}
                onChange={handleCalcToggle}
                color="primary"
              />
            }
            label={useBackendCalc ? getText('useBackendCalc', language) : getText('useFrontendCalc', language)}
          />
        </Tooltip>
      </Box>
    </Paper>
  );
}

export default DeviceSelector;