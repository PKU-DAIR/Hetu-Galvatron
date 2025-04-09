// src/components/MemoryTreemap.js
import React, { useEffect, useState } from 'react';
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';
import { Box, Typography, Paper, Divider } from '@mui/material';
import { useLanguage, LANGUAGES } from '../context/LanguageContext';
import { getText } from '../utils/translations';

// Memory type color mapping with distinct color schemes
const COLORS = {
  // Activation memory related - green color scheme
  'Activation Memory': '#c8e6c9', // Light green background
  'Layer': '#c8e6c9',  // Green
  'PP Activation': '#b2dfb0',  // Green
  'A': '#e8f5e9',  // Very light green - Attention
  'F': '#c1e5c1',  // Light green - Feed Forward
  'L': '#d8e8c5',  // Light yellow-green - Layer Norm
  'Other': '#aed581',  // Medium green - Context
  'P': '#c5d6d8',  // Light blue-gray - Projection
  
  // Parameter and optimizer related - blue color scheme
  'Parameter Memory': '#bbdefb', // Light blue background
  'Parameters': '#e3f2fd', // Light blue
  'Gradients': '#d2b48c', // Tan - Gradients
  'Optimizer': '#64b5f6', // Blue
  'G': '#d2b48c',  // Tan - Gradients
  'O': '#64b5f6',  // Blue - Optimizer states
};

// Create block style configuration with language support
const createBlockStyles = (language) => ({
  'Activation Memory': {
    borderColor: '#2e7d32',
    borderWidth: 4,
    borderStyle: 'solid',
    textColor: '#2e7d32',
    headerColor: '#2e7d32',
    label: getText('activationMemory', language),
    abbreviation: 'A'
  },
  'Model States': {
    borderColor: '#1565c0',
    borderWidth: 4, 
    borderStyle: 'solid',
    textColor: '#1565c0',
    headerColor: '#1565c0',
    label: getText('parameterMemory', language),
    abbreviation: 'P'
  }
});
  
// Format memory data for treemap visualization
const formatMemoryData = (memoryData, config, language) => {
  if (!memoryData) return [];
  
  // Get block styles with current language
  const BLOCK_STYLES = createBlockStyles(language);
  
  // Extract parameters
  const activation = memoryData.activation || 0;
  
  // Create root node
  const rootData = {
    name: 'Memory',
    children: []
  };
  
  // 1. Add activation memory section
  const activationNode = {
    name: 'Activation Memory',
    label: `${getText('activationMemory', language)}: ${activation.toFixed(2)} MB`,
    displayLabel: BLOCK_STYLES['Activation Memory'].label,
    value: activation,
    isMainNode: true,
    children: []
  };
  
  // Use layer count from config
  const numLayers = memoryData?.num_layers || config?.num_layers || 12;
  const displayLayers = numLayers;
  const activationPerLayer = memoryData.per_layer_activation;
  
  // Allocate memory for each layer
  for (let i = 0; i < displayLayers; i++) {
    const layerNode = {
      name: 'Layer',
      label: `L`,
      fullName: `Layer ${i+1} Activation`,
      value: activationPerLayer,
      children: []
    };
    
    // TODO: Allocate memory within each layer for attention, ffn, and layer norm
    // const attnValue = activationPerLayer * 0.5;
    // const ffnValue = activationPerLayer * 0.4;
    // const lnValue = activationPerLayer * 0.1;
    
    // // Add components to each layer
    // layerNode.children.push({ 
    //   name: 'A', 
    //   label: 'A', 
    //   fullName: 'Attention',
    //   value: attnValue
    // });
    
    // layerNode.children.push({ 
    //   name: 'F', 
    //   label: 'F', 
    //   fullName: 'FFN',
    //   value: ffnValue
    // });
    
    // layerNode.children.push({ 
    //   name: 'L', 
    //   label: 'L', 
    //   fullName: 'LayerNorm',
    //   value: lnValue
    // });
    
    activationNode.children.push(layerNode);
  }

  // Add pp activation nodes for pipeline parallel stages
  for (let i = 0; i < config.pp_size - memoryData.stage_idx - 1; i++) {
    const ppActNode = {
      name: 'PP Activation',
      label: `PP`,
      fullName: `PP Activation`,
      value: memoryData.per_layer_activation * memoryData.num_layers,
      children: []
    };
    
    activationNode.children.push(ppActNode);
  }
  
  // Add context and projection area
  activationNode.children.push({
    name: 'Other',
    label: 'O',
    fullName: 'Other Memory Activation',
    value: memoryData.other_memory_activation
  });
  
  // 2. Add parameter memory section (combine parameters, gradients, and optimizer)
  const parameterNode = {
    name: 'Model States',
    label: `M`,
    displayLabel: BLOCK_STYLES['Model States'].label,
    value: memoryData.model_states,
    isMainNode: true,
    children: [
      {
        name: 'Parameters',
        label: 'P',
        fullName: 'Parameters',
        value: memoryData.parameter
      },
      {
        name: 'Gradients (BF16)',
        label: 'G',
        fullName: 'Gradients (BF16)',
        value: memoryData.gradient
      },
      {
        name: 'Optimizer',
        label: 'O',
        fullName: 'Optimizer',
        value: memoryData.optimizer
      },
      {
        name: 'Grad Accumulate (FP32)',
        label: 'GA',
        fullName: 'Grad Accumulate (FP32)',
        value: memoryData.grad_accumulate
      }
    ]
  };
  
  // Add main nodes to root
  rootData.children.push(activationNode);
  rootData.children.push(parameterNode);
  
  // console.log('Treemap data structure:', rootData);
  return [rootData];
};

// Calculate optimal aspect ratio based on memory distribution
const calculateBestAspectRatio = (memoryData) => {
  if (!memoryData) return 4/3; // Default aspect ratio
  
  const activation = memoryData.activation || 0;
  const parameter = memoryData.parameter || 0;
  const optimizer = memoryData.optimizer || 0;
  const paramTotal = parameter * 2 + optimizer;
  
  // Calculate ratio
  const ratio = activation / paramTotal;
  
  // Set different aspect ratios based on ratio
  if (ratio > 1.5) {
    // Activation memory is notably larger, use horizontal layout
    return 1.0; // More square-like
  } else if (ratio < 0.7) {
    // Parameter memory is notably larger, use vertical layout
    return 2.0; // More horizontal rectangle
  } else {
    // Sizes are comparable, use near-square ratio
    return 1.5;
  }
};

// Helper function to adjust color brightness
const adjustColorBrightness = (hex, factor) => {
  // Convert hex to rgb
  let r = parseInt(hex.substring(1, 3), 16);
  let g = parseInt(hex.substring(3, 5), 16);
  let b = parseInt(hex.substring(5, 7), 16);
  
  // Adjust brightness (positive to lighten, negative to darken)
  if (factor > 0) {
    r = Math.min(255, Math.round(r + (255 - r) * factor));
    g = Math.min(255, Math.round(g + (255 - g) * factor));
    b = Math.min(255, Math.round(b + (255 - b) * factor));
  } else {
    r = Math.max(0, Math.round(r * (1 + factor)));
    g = Math.max(0, Math.round(g * (1 + factor)));
    b = Math.max(0, Math.round(b * (1 + factor)));
  }
  
  // Convert back to hex
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
};

// Custom content rendering with language support
const CustomizedContent = (props) => {
  const { x, y, width, height, name, depth, payload, onHover, language } = props;
  
  // Get block styles with current language
  const BLOCK_STYLES = createBlockStyles(language);
  
  // Check if it's a main node (activation memory or parameter memory)
  const isMainNode = payload?.isMainNode || false;
  
  // Only show text if block is large enough
  const shouldShowText = width > 30 && height > 20;
  
  // If main node, show full label; otherwise show appropriate label based on space
  let displayName = name;
  let fullName = '';
  if (payload) {
    if (isMainNode) {
      displayName = payload.displayLabel;
    } else {
      displayName = payload.label || name;
      fullName = payload.fullName || '';
    }
  }
  
  // Adjust color based on depth
  let fillColor = COLORS[name] || '#f5f5f5';
  
  // Make nested levels have different color depths, but with smaller adjustments
  if (depth > 1 && !isMainNode) {
    // Deeper levels get darker/lighter
    const factor = name === 'Layer' ? 0.03 : 0.05;
    const adjustFactor = depth === 1 ? 0 : (depth - 1) * factor;
    
    // Use different adjustment methods for different block types
    fillColor = adjustColorBrightness(fillColor, -adjustFactor); 
  }
  
  // Get main block style
  const blockStyle = isMainNode ? 
    BLOCK_STYLES[name] || BLOCK_STYLES['Activation Memory'] : null;
  
  // Center label style
  const centerLabelStyle = {
    fontSize: Math.min(14, width / 6, height / 6),
    fontFamily: 'Arial Black, Arial Bold, sans-serif',
    fontWeight: 'bold',
    fill: 'black',
    textShadow: 'none',
    paintOrder: 'stroke',
    stroke: 'none'
  };
  
  // Border style based on block type
  const strokeWidth = isMainNode ? (blockStyle?.borderWidth || 2) : 1;
  const strokeColor = isMainNode ? (blockStyle?.borderColor || 'rgba(0,0,0,0.3)') : 'rgba(255,255,255,0.7)';
  const strokeStyle = isMainNode ? (blockStyle?.borderStyle || 'solid') : 'solid';
  
  // Height of main node header bar
  const headerHeight = isMainNode ? 30 : 0;
  
  // Show full name only if block is large enough
  const shouldShowFullName = width > 60 && height > 40 && fullName;
  
  // Special style: bottom label bar for main blocks
  const showBottomLabel = isMainNode && width > 100 && height > 100;
  const bottomLabelHeight = showBottomLabel ? 28 : 0;
  
  return (
    <g>
      {/* Main block background */}
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        style={{
          fill: fillColor,
          stroke: strokeColor,
          strokeWidth: strokeWidth,
          strokeOpacity: 1,
          strokeDasharray: strokeStyle === 'dashed' ? '5,5' : 'none',
          transition: 'fill 0.3s, stroke 0.3s',
          cursor: 'pointer'
        }}
        onMouseEnter={() => onHover && onHover(props)}
        onMouseLeave={() => onHover && onHover(null)}
      />
      
      {/* Main block header bar */}
      {isMainNode && (
        <rect
          x={x}
          y={y}
          width={width}
          height={headerHeight}
          style={{
            fill: blockStyle?.headerColor || '#333333',
            stroke: 'none'
          }}
        />
      )}
      
      {/* Main block bottom label bar */}
      {showBottomLabel && (
        <rect
          x={x}
          y={y + height - bottomLabelHeight}
          width={width}
          height={bottomLabelHeight}
          style={{
            fill: blockStyle?.headerColor || '#333333',
            stroke: 'none',
            opacity: 0.9
          }}
        />
      )}
      
      {/* Main block title text */}
      {isMainNode && shouldShowText && (
        <>
          <text
            x={x + 10}
            y={y + headerHeight/2 + 1}
            textAnchor="start"
            dominantBaseline="middle"
            style={{
              fontSize: Math.min(16, Math.max(12, width / 40)),
              fontFamily: 'Arial Black, Arial Bold, sans-serif',
              fontWeight: 'bold',
              fill: 'white',
              textShadow: 'none',
              paintOrder: 'stroke', 
              stroke: 'none'
            }}
          >
            {displayName}
          </text>
          
          {/* Size info on right side of header */}
          <text
            x={x + width - 10}
            y={y + headerHeight/2 + 1}
            textAnchor="end"
            dominantBaseline="middle"
            style={{
              fontSize: Math.min(14, Math.max(10, width / 45)),
              fontFamily: 'Arial, sans-serif',
              fontWeight: 'bold',
              fill: 'white',
              textShadow: 'none',
              paintOrder: 'stroke', 
              stroke: 'none'
            }}
          >
            {payload?.value?.toFixed(2)} MB
          </text>
          
          {/* Bottom label text */}
          {showBottomLabel && (
            <text
              x={x + width/2}
              y={y + height - bottomLabelHeight/2}
              textAnchor="middle"
              dominantBaseline="middle"
              style={{
                fontSize: Math.min(14, Math.max(10, width / 45)),
                fontFamily: 'Arial Black, Arial Bold, sans-serif',
                fontWeight: 'bold',
                fill: 'white',
                textShadow: 'none',
                paintOrder: 'stroke', 
                stroke: 'none'
              }}
            >
              {name === 'Activation Memory' ? 
                `${getText('activationMemory', language)} (Activation)` : 
                `${getText('parameterMemory', language)} (Parameters)`}
            </text>
          )}
        </>
      )}
      
      {/* Child block center identifier */}
      {shouldShowText && !isMainNode && (
        <>
          <text
            x={x + width / 2}
            y={y + height / 2 - (shouldShowFullName ? 8 : 0)}
            textAnchor="middle"
            dominantBaseline="middle"
            style={centerLabelStyle}
          >
            {displayName}
          </text>
          
          {/* Show full name when block is large enough */}
          {shouldShowFullName && (
            <text
              x={x + width / 2}
              y={y + height / 2 + 16}
              textAnchor="middle"
              dominantBaseline="middle"
              style={{
                fontSize: Math.min(11, width / 12),
                fontFamily: 'Arial, sans-serif',
                fontWeight: 'bold',
                fill: 'rgba(0,0,0,0.7)',
                textShadow: 'none',
                paintOrder: 'stroke',
                stroke: 'none'
              }}
            >
              {fullName}
            </text>
          )}
        </>
      )}
    </g>
  );
};

// Custom tooltip content with language support
const CustomTooltip = ({ active, payload, language }) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload;
    const name = data.fullName || data.name;
    const value = data.value;
    const parentNode = payload[0].payload.parent; // Get parent node from payload
    
    return (
      <div style={{ 
        backgroundColor: '#fff', 
        padding: '10px', 
        border: '1px solid #ccc',
        borderRadius: '4px',
        boxShadow: '0 2px 5px rgba(0,0,0,0.15)'
      }}>
        <p style={{ margin: 0 }}><strong>{name}</strong></p>
        <p style={{ margin: 0 }}>{language === LANGUAGES.ZH ? '内存:' : 'Memory:'} {value.toFixed(2)} MB</p>
        {parentNode && <p style={{ margin: 0 }}>{language === LANGUAGES.ZH ? '所属:' : 'Belongs to:'} {parentNode}</p>}
      </div>
    );
  }
  return null;
};

function MemoryTreemap({ memoryData, config, error }) {
  const { language } = useLanguage();
  const [data, setData] = useState([]);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [aspectRatio, setAspectRatio] = useState(4/3);
  
  useEffect(() => {
    if (memoryData) {
      const formattedData = formatMemoryData(memoryData, config, language);
      // console.log('Formatted treemap data:', formattedData);
      setData(formattedData);
      
      // Calculate optimal aspect ratio based on memory distribution
      const bestRatio = calculateBestAspectRatio(memoryData);
      // console.log('Calculated optimal aspect ratio:', bestRatio);
      setAspectRatio(bestRatio);
    }
  }, [memoryData, config, language]);
  
  const handleHover = (node) => {
    setHoveredNode(node);
  };
  
  // Display message when no data is available or there's an error
  if (!memoryData) {
    return (
      <Paper 
        elevation={3} 
        sx={{ 
          p: 3, 
          mb: 3, 
          borderRadius: 2,
          background: '#ffffff',
          minHeight: '400px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        <Typography variant="h6" sx={{ mb: 2, color: '#757575', textAlign: 'center' }}>
          {error ? (
            error.isConfigMissing ? getText('configMissing', language) : getText('cannotVisualize', language)
          ) : getText('noDataMessage', language)}
        </Typography>
        
        <Typography variant="body1" sx={{ color: '#9e9e9e', textAlign: 'center', maxWidth: '80%', mb: 2 }}>
          {error ? (
            error.needsConfigFile ? 
              (language === LANGUAGES.ZH ? '请上传 Galvatron 配置文件以获取完整的内存分析' : 'Please upload a Galvatron config file to get complete memory analysis') : 
              (language === LANGUAGES.ZH ? '当前参数配置无法生成有效的内存分析，请调整参数或上传匹配的配置文件' : 'Current parameter settings cannot generate valid memory analysis, please adjust parameters or upload a matching config file')
          ) : getText('uploadPrompt', language)}
        </Typography>
        
        {error && (
          <Typography variant="caption" sx={{ color: '#f44336', textAlign: 'center', maxWidth: '90%', mt: 1, fontFamily: 'monospace', p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
            {error.message}
          </Typography>
        )}
      </Paper>
    );
  }
  
  // Get updated block styles for current language
  const BLOCK_STYLES = createBlockStyles(language);
  
  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 3, 
        mb: 3, 
        borderRadius: 2,
        background: '#ffffff'
      }}
    >
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ fontWeight: 'bold', color: '#333', fontFamily: 'Arial Black, Arial Bold, sans-serif' }}>
          {getText('memoryVisTitle', language)}
        </Typography>
        <Typography variant="subtitle1" sx={{ color: '#1976d2', fontWeight: 'bold', fontFamily: 'Arial Black, Arial Bold, sans-serif' }}>
          {getText('totalMemory', language)} {memoryData.total.toFixed(2)} MB
        </Typography>
      </Box>
      
      <Divider sx={{ mb: 3 }} />
      
      <Box sx={{ height: 600, mt: 2 }}>
        <ResponsiveContainer width="100%" height="100%">
          <Treemap
            data={data}
            dataKey="value"
            aspectRatio={aspectRatio} // Dynamic aspect ratio
            stroke="#fff"
            fill="#f5f5f5"
            content={<CustomizedContent onHover={handleHover} language={language} />}
            animationDuration={300} // Reduced animation time
            isAnimationActive={true}
          >
            <Tooltip content={<CustomTooltip language={language} />} />
          </Treemap>
        </ResponsiveContainer>
      </Box>
      
      {/* Legend */}
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box sx={{ 
            width: 20, 
            height: 20, 
            bgcolor: COLORS['Activation Memory'], 
            border: `3px solid ${BLOCK_STYLES['Activation Memory'].borderColor}`, 
            mr: 1 
          }} />
          <Typography sx={{ fontFamily: 'Arial, sans-serif' }}>
            {getText('activationMemory', language)} (Activation)
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box sx={{ 
            width: 20, 
            height: 20, 
            bgcolor: COLORS['Parameter Memory'], 
            border: `3px solid ${BLOCK_STYLES['Model States'].borderColor}`, 
            mr: 1 
          }} />
          <Typography sx={{ fontFamily: 'Arial, sans-serif' }}>
            {getText('parameterMemory', language)} (Parameters)
          </Typography>
        </Box>
      </Box>
      
      {/* Hover information */}
      <Box sx={{ mt: 2, textAlign: 'center', borderTop: '1px solid #e0e0e0', pt: 2 }}>
        {hoveredNode ? (
          <Typography variant="body2" sx={{ color: 'text.secondary', fontFamily: 'Arial Black, Arial Bold, sans-serif', fontWeight: 'bold' }}>
            {getText('currentSelection', language)} {hoveredNode.name && (hoveredNode.payload?.fullName || hoveredNode.payload?.displayLabel || hoveredNode.payload?.label || hoveredNode.name)}
            {hoveredNode.payload?.value && ` - ${hoveredNode.payload.value.toFixed(2)} MB`}
          </Typography>
        ) : (
          <Typography variant="body2" sx={{ color: 'text.secondary', fontStyle: 'italic', fontFamily: 'Arial Black, Arial Bold, sans-serif' }}>
            {getText('hoverInfo', language)}
          </Typography>
        )}
      </Box>
    </Paper>
  );
}

export default MemoryTreemap;