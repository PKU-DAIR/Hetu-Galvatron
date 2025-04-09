import React, { useState, useEffect } from 'react';
import { Container, CssBaseline, Typography, Box, AppBar, Toolbar } from '@mui/material';
import ConfigPanel from './components/ConfigPanel';
import MemoryVisualization from './components/MemoryVisualization';
import MemoryCostModel from './models/MemoryCostModel';
import MemoryTreemap from './components/MemoryTreemap';
import ConfigSelector from './components/ConfigSelector';
import { LanguageProvider, useLanguage, LANGUAGES } from './context/LanguageContext';
import LanguageToggle from './components/LanguageToggle';
import LanguageInstructions from './components/LanguageInstructions';
import { getText } from './utils/translations';

// Main application component
const AppContent = () => {
  const { language } = useLanguage();
  
  // Default configuration
  const [config, setConfig] = useState({
    attention_heads: 8,
    micro_batch_size: 1,
    hidden_dim: 512,
    ff_dim: 2048,
    num_layers: 12,
    seq_length: 128,
    vocab_size: 30522,
    tp_size: 8,
    dp_size: 1,
    pp_size: 1,
    stage_idx: 0,
    sequence_parallel: true,
    mixed_precision: true,
    global_batch_size: 8,
    chunks: 8,
    zero_stage: 0,
    total_gpus: 8,
    checkpoint: false
  });
  
  // Store raw configuration data
  const [rawConfigData, setRawConfigData] = useState(null);
  
  // Memory calculation results
  const [memoryResults, setMemoryResults] = useState(null);
  // Error information
  const [memoryError, setMemoryError] = useState(null);

  // Handle config file loading
  const handleConfigLoaded = (data) => {
    // Store raw config data
    setRawConfigData(data);
    
    // Extract parameters from config file and update panel
    if (data && data.model_config) {
      const modelConfig = data.model_config;
      
      // Extract model parameters
      const newConfig = {
        ...config,
        attention_heads: modelConfig.n_heads || config.attention_heads,
        hidden_dim: modelConfig.dim || config.hidden_dim,
        ff_dim: modelConfig.ffn_dim || config.ff_dim,
        num_layers: modelConfig.n_layers || config.num_layers,
        vocab_size: modelConfig.vocab_size || config.vocab_size
      };
      
      // Extract sequence length if available
      // Find sequence lengths in config
      const seqLengths = [];
      for (const key in data) {
        if (key.match(/layertype_\d+(?:_sp)?$/)) {
          const layerKey = key;
          Object.keys(data[layerKey]).forEach(seq => {
            const seqNum = parseInt(seq);
            if (!seqLengths.includes(seqNum)) {
              seqLengths.push(seqNum);
            }
          });
          break;
        }
      }
      
      // If sequence lengths found, use the first (usually smallest) one
      if (seqLengths.length > 0) {
        seqLengths.sort((a, b) => a - b);
        newConfig.seq_length = seqLengths[0];
      }
      
      // Update config
      setConfig(newConfig);
      // console.log("Parameters extracted from config file:", newConfig);
    }
  };
  
  // Recalculate memory when config changes
  useEffect(() => {
    try {
      // console.log("Recalculating memory based on config:", config);
      // Clear previous errors
      setMemoryError(null);
      
      // Pass raw config data to MemoryCostModel as second parameter
      const memoryModel = new MemoryCostModel(config, rawConfigData);
      const results = memoryModel.getMemoryCost();
      // console.log("Memory calculation results:", results);
      
      setMemoryResults(results);
    } catch (error) {
      console.error("Memory calculation error:", error);
      setMemoryResults(null);
      setMemoryError({
        message: error.message,
        isConfigMissing: error.message.includes("not found in rawConfig"),
        needsConfigFile: !rawConfigData
      });
    }
  }, [config, rawConfigData]);
  
  const handleConfigChange = (newConfig) => {
    // console.log("Configuration changed:", newConfig);
    setConfig(newConfig);
  };
  
  return (
    <React.Fragment>
      <CssBaseline />
      <AppBar position="sticky" color="default" elevation={1}>
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Typography variant="h6" color="#000000">
            {getText('appTitle', language)}
          </Typography>
          <LanguageToggle />
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom align="center">
            {language === LANGUAGES.ZH 
              ? 'Galvatron 内存估计可视化工具' 
              : 'Galvatron Memory Estimation Tool'}
          </Typography>
          
          <Typography variant="body1" color="text.secondary">
            {language === LANGUAGES.ZH 
              ? '本工具可帮助您分析和可视化Galvatron内存分析结果。选择模型和精度配置来开始。'
              : 'This tool helps you analyze and visualize Galvatron memory analysis results. Choose a model and precision configuration to start.'}
          </Typography>
        
          <ConfigSelector onConfigLoaded={handleConfigLoaded} />
          
          <Box sx={{ 
            display: 'flex', 
            flexDirection: { xs: 'column', md: 'row' }, 
            gap: 3, 
            mt: 3 
          }}>
            <Box sx={{ width: { xs: '100%', md: '58%' } }}>
              {/* Memory treemap visualization */}
              <MemoryTreemap memoryData={memoryResults} config={config} error={memoryError} />
            </Box>
            
            <Box sx={{ width: { xs: '100%', md: '38%' } }}>
              {/* Configuration panel */}
              <ConfigPanel 
                config={config} 
                onConfigChange={handleConfigChange} 
              />
            </Box>
          </Box>
          
          {/* Bottom chart visualization */}
          <MemoryVisualization memoryResults={memoryResults} />
          
          {/* Footer */}
          <Box sx={{ mt: 4, textAlign: 'center', color: 'text.secondary' }}>
            <Typography variant="body2">
              {getText('basedOnModel', language)}
            </Typography>
            
            {/* 添加GitHub和文档链接 */}
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              gap: 3, 
              mt: 1, 
              mb: 1 
            }}>
              <Typography 
                variant="body2" 
                component="a" 
                href="https://github.com/PKU-DAIR/Hetu-Galvatron" 
                target="_blank"
                rel="noopener noreferrer"
                sx={{ 
                  color: 'primary.main', 
                  textDecoration: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                  '&:hover': { textDecoration: 'underline' }
                }}
              >
                <Box 
                  component="img" 
                  src="https://github.githubassets.com/favicons/favicon.svg" 
                  sx={{ width: 16, height: 16 }} 
                />
                {getText('githubLink', language)}
              </Typography>
              
              <Typography 
                variant="body2" 
                component="a" 
                href="https://hetu-galvatron.readthedocs.io/" 
                target="_blank"
                rel="noopener noreferrer"
                sx={{ 
                  color: 'primary.main', 
                  textDecoration: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                  '&:hover': { textDecoration: 'underline' }
                }}
              >
                <Box 
                  component="img" 
                  src="https://readthedocs.org/favicon.ico" 
                  sx={{ width: 16, height: 16 }} 
                />
                {getText('documentationLink', language)}
              </Typography>
            </Box>
            
            <Typography variant="body2">
              © {new Date().getFullYear()} Galvatron {getText('teamCopyright', language)}
            </Typography>
          </Box>
        </Box>
      </Container>
      
      {/* Language toggle instructions */}
      <LanguageInstructions />
    </React.Fragment>
  );
};

// Wrapper with Language Provider
function App() {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
}

export default App;