import React, { useState } from 'react';
import { Box, FormControl, InputLabel, Select, MenuItem, Button, Typography, Paper, Alert } from '@mui/material';
import { useLanguage } from '../context/LanguageContext';
import { getText } from '../utils/translations';

/**
 * Config file selector component
 * Based on the config file reading logic from galvatron/core/search_engine/search_engine.py
 * Allows users to select different model and precision configurations
 * @param {Object} props 
 * @param {Function} props.onConfigLoaded Callback function when config is loaded
 */
function ConfigSelector({ onConfigLoaded }) {
  const { language } = useLanguage();
  
  // State variables
  const [models] = useState(['qwen2.5-32b', 'llama3-8b', 'llama2-70b', 'llama3-70b']);
  const [precisions] = useState(['bf16']);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedPrecision, setSelectedPrecision] = useState('');
  const [configContent, setConfigContent] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState('');

  // Handle model selection change
  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
    setConfigContent(null);
  };

  // Handle precision selection change
  const handlePrecisionChange = (event) => {
    setSelectedPrecision(event.target.value);
    setConfigContent(null);
  };

  // Load configuration file
  const loadConfig = () => {
    if (!selectedModel || !selectedPrecision) {
      setError(getText('selectModelPrecision', language));
      return;
    }

    setLoading(true);
    setError(null);

    // Build configuration file path (similar to memory_profiling_path in search_engine.py)
    const configName = `memory_profiling_${selectedPrecision}_${selectedModel}.json`;
    
    // Use PUBLIC_URL environment variable for correct path
    fetch(`${process.env.PUBLIC_URL}/configs/${configName}`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`${getText('configLoadFailed', language)} ${configName}`);
        }
        return response.json();
      })
      .then(data => {
        console.log("Loaded config data from server:", data);
        
        // Convert keys to integers
        const processedData = convertKeysToInt(data);
        setConfigContent(processedData);
        
        // Pass processed data to App component
        if (onConfigLoaded) {
          onConfigLoaded(processedData);
        }
      })
      .catch(err => {
        console.error('Failed to load config:', err);
        setError(`${getText('configLoadFailed', language)} ${err.message}`);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  // Convert string keys to integer keys (similar to convert_keys_to_int function in search_engine.py)
  const convertKeysToInt = (obj) => {
    if (typeof obj !== 'object' || obj === null) {
      return obj;
    }

    if (Array.isArray(obj)) {
      return obj.map(convertKeysToInt);
    }

    const result = {};
    for (const key in obj) {
      const newKey = /^\d+$/.test(key) ? parseInt(key, 10) : key;
      result[newKey] = convertKeysToInt(obj[key]);
    }

    return result;
  };

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setFileName(file.name);
    setError(null);
    setLoading(true);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = JSON.parse(e.target.result);
        console.log("File content read:", content);
        
        // Convert keys to integers
        const processedData = convertKeysToInt(content);
        setConfigContent(processedData);
        
        // Pass processed data to App component
        if (onConfigLoaded) {
          onConfigLoaded(processedData);
        }
      } catch (err) {
        console.error('Failed to parse file:', err);
        setError(`${getText('fileParseError', language)} ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    reader.onerror = () => {
      setError(getText('fileReadFailed', language));
      setLoading(false);
    };
    
    reader.readAsText(file);
  };
  
  return (
    <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        {getText('configSelectorTitle', language)}
      </Typography>
      
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2 }}>
          <FormControl fullWidth>
            <InputLabel>{getText('selectModel', language)}</InputLabel>
            <Select
              value={selectedModel}
              onChange={handleModelChange}
              label={getText('selectModel', language)}
            >
              {models.map(model => (
                <MenuItem key={model} value={model}>{model}</MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormControl fullWidth>
            <InputLabel>{getText('selectPrecision', language)}</InputLabel>
            <Select
              value={selectedPrecision}
              onChange={handlePrecisionChange}
              label={getText('selectPrecision', language)}
            >
              {precisions.map(precision => (
                <MenuItem key={precision} value={precision}>{precision}</MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <Button 
            variant="contained" 
            onClick={loadConfig}
            disabled={!selectedModel || !selectedPrecision || loading}
            sx={{ minWidth: '120px', height: '56px' }}
          >
            {loading ? getText('loading', language) : getText('loadConfig', language)}
          </Button>
        </Box>
        
        <Box sx={{ border: '1px dashed #ccc', p: 3, borderRadius: 1, textAlign: 'center' }}>
          <Typography variant="subtitle1" gutterBottom>
            {getText('uploadLocal', language)}
          </Typography>
          
          <input
            type="file"
            accept=".json"
            id="config-file-input"
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
          <label htmlFor="config-file-input">
            <Button 
              variant="outlined" 
              component="span"
              startIcon={<span>📁</span>}
              disabled={loading}
            >
              {getText('selectJsonFile', language)}
            </Button>
          </label>
          
          {fileName && (
            <Typography variant="body2" sx={{ mt: 1 }}>
              {getText('fileSelected', language)} {fileName}
            </Typography>
          )}
        </Box>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
      
      {configContent && (
        <Alert severity="success" sx={{ mt: 2 }}>
          {getText('configLoadSuccess', language)}
        </Alert>
      )}
    </Paper>
  );
}

export default ConfigSelector; 