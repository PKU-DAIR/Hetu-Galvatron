import React, { useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Sector, Legend
} from 'recharts';
import { Typography, Paper, Box, Tabs, Tab } from '@mui/material';
import { useLanguage, LANGUAGES } from '../context/LanguageContext';
import { getText } from '../utils/translations';

// Color configuration
const COLORS = {
  // Primary colors
  activation: '#4CAF50',      // Green
  parameter: '#9C27B0',       // Purple
  gradient: '#FF9800',        // Orange
  optimizer: '#2196F3',       // Blue
  model_states: '#E91E63',    // Pink
  grad_accumulate: '#8BC34A', // Light green
  other: '#795548',           // Brown
  
  // Secondary colors with better differentiation
  activation_secondary: '#81C784',    // Lighter green
  parameter_secondary: '#BA68C8',     // Lighter purple
  gradient_secondary: '#FFB74D',      // Lighter orange
  optimizer_secondary: '#64B5F6',     // Lighter blue
  model_states_secondary: '#F48FB1',  // Lighter pink
  grad_accumulate_secondary: '#AED581', // Lighter light green
};

// Create name mapping for UI display based on current language
const getNameMap = (language) => ({
  activation: getText('activationMemory', language),
  parameter: getText('parameterMemory', language),
  gradient: getText('gradientMemory', language),
  optimizer: getText('optimizerMemory', language),
  model_states: getText('modelStates', language),
  grad_accumulate: getText('gradAccumulate', language),
  other_memory_activation: getText('otherActivation', language),
  other_memory_parameter: getText('otherParameter', language),
  other_memory_gradient: getText('otherGradient', language),
  other_memory_optimizer: getText('otherOptimizer', language),
  other_memory_model_states: getText('otherModelStates', language),
  other_memory_grad_accumulate: getText('otherGradAccumulate', language)
});

// Process memory data to show hierarchical relationships
const processMemoryData = (memoryResults, NAME_MAP) => {
  if (!memoryResults) return { allCategories: [], primaryCategories: [] };

  // Create primary categories
  const primaryCategories = [
    {
      key: 'activation',
      name: NAME_MAP.activation,
      value: memoryResults.activation || 0,
      color: COLORS.activation,
      isMainCategory: true
    },
    {
      key: 'model_states',
      name: NAME_MAP.model_states,
      value: memoryResults.model_states || 0,
      color: COLORS.model_states,
      isMainCategory: true,
      children: []
    }
  ];

  // Add children to model_states
  const modelStatesChildren = [
    {
      key: 'parameter',
      name: NAME_MAP.parameter,
      value: memoryResults.parameter || 0,
      color: COLORS.parameter
    },
    {
      key: 'gradient',
      name: NAME_MAP.gradient,
      value: memoryResults.gradient || 0,
      color: COLORS.gradient
    },
    {
      key: 'optimizer',
      name: NAME_MAP.optimizer,
      value: memoryResults.optimizer || 0,
      color: COLORS.optimizer
    }
  ];

  if (memoryResults.grad_accumulate > 0) {
    modelStatesChildren.push({
      key: 'grad_accumulate',
      name: NAME_MAP.grad_accumulate,
      value: memoryResults.grad_accumulate || 0,
      color: COLORS.grad_accumulate
    });
  }

  // Add children to model_states category
  primaryCategories[1].children = modelStatesChildren;

  // Process other memory categories
  const otherCategories = [];
  // [
  //   'other_memory_activation',
  //   'other_memory_parameter',
  //   'other_memory_gradient',
  //   'other_memory_optimizer',
  //   'other_memory_model_states',
  //   'other_memory_grad_accumulate'
  // ].forEach(key => {
  //   if (memoryResults[key] && memoryResults[key] > 0) {
  //     otherCategories.push({
  //       key,
  //       name: NAME_MAP[key] || key.replace('other_memory_', '').replace('_', ' '),
  //       value: memoryResults[key],
  //       parentKey: key.replace('other_memory_', '')
  //     });
  //   }
  // });

  // Create flat array for bar chart
  const allCategories = [
    ...primaryCategories,
    ...modelStatesChildren,
    ...otherCategories
  ];

  // Calculate percentage for each part
  allCategories.forEach(cat => {
    cat.percentage = (cat.value / memoryResults.total) * 100;
  });

  // Sort by size (descending)
  allCategories.sort((a, b) => b.value - a.value);

  return {
    allCategories,
    primaryCategories,
    total: memoryResults.total
  };
};

// 自定义动画效果的活跃扇区
const renderActiveShape = (props) => {
  const RADIAN = Math.PI / 180;
  const { 
    cx, cy, midAngle, innerRadius, outerRadius, startAngle, endAngle,
    fill, payload, percent, value, name
  } = props;
  
  const sin = Math.sin(-RADIAN * midAngle);
  const cos = Math.cos(-RADIAN * midAngle);
  const sx = cx + (outerRadius + 10) * cos;
  const sy = cy + (outerRadius + 10) * sin;
  const mx = cx + (outerRadius + 30) * cos;
  const my = cy + (outerRadius + 30) * sin;
  const ex = mx + (cos >= 0 ? 1 : -1) * 22;
  const ey = my;
  const textAnchor = cos >= 0 ? 'start' : 'end';

  return (
    <g>
      <text x={cx} y={cy} dy={8} textAnchor="middle" fill={fill} fontSize={14} fontWeight="bold">
        {payload.key}
      </text>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
      <Sector
        cx={cx}
        cy={cy}
        startAngle={startAngle}
        endAngle={endAngle}
        innerRadius={outerRadius + 6}
        outerRadius={outerRadius + 10}
        fill={fill}
      />
      <path d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`} stroke={fill} fill="none" />
      <circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
      <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} textAnchor={textAnchor} fill="#333" fontSize={12}>{name}</text>
      <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} dy={18} textAnchor={textAnchor} fill="#999" fontSize={12}>
        {`${value.toFixed(2)} MB (${(percent * 100).toFixed(2)}%)`}
      </text>
    </g>
  );
};

function MemoryVisualization({ memoryResults }) {
  const { language } = useLanguage();
  const NAME_MAP = getNameMap(language);
  const [visMode, setVisMode] = React.useState(0);
  const [activeIndex, setActiveIndex] = useState(null);
  const [expandedCategory, setExpandedCategory] = useState(null);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setVisMode(newValue);
    setActiveIndex(null);
    setExpandedCategory(null);
  };
  
  // 处理饼图扇区点击
  const handlePieClick = (data, index) => {
    setActiveIndex(index);
    
    // 如果点击的是主分类，切换展开/折叠状态
    if (data.isMainCategory) {
      setExpandedCategory(expandedCategory === data.key ? null : data.key);
    }
  };
  
  // Ensure we have data
  if (!memoryResults) {
    return (
      <Paper elevation={3} sx={{ p: 3, mt: 3, minHeight: '200px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          {getText('waitingForData', language)}
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {getText('tryUploadConfig', language)}
          </Typography>
        </Typography>
      </Paper>
    );
  }

  // Process data for visualizations
  const { allCategories, primaryCategories, total } = processMemoryData(memoryResults, NAME_MAP);

  // Only show top categories in the bar chart
  const chartCategories = allCategories.filter(cat => cat.percentage > 1).slice(0, 10);
  
  // 为饼图准备数据
  const pieData = expandedCategory === 'model_states' 
    ? primaryCategories.find(cat => cat.key === 'model_states')?.children || [] 
    : primaryCategories;
  
  // Function to get distinct striped pattern for primary categories
  const getPattern = (key) => {
    switch(key) {
      case 'activation':
        return { 
          bg: COLORS.activation,
          pattern: 'solid'
        };
      case 'model_states':
        return { 
          bg: COLORS.model_states,
          pattern: 'solid'
        };
      default:
        return { 
          bg: COLORS[key] || COLORS.other,
          pattern: 'solid'
        };
    }
  };
  
  return (
    <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
        {getText('memoryAnalysis', language)}
      </Typography>
      
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Box sx={{ textAlign: 'center', mb: 2 }}>
          <Typography variant="h5" fontWeight="bold" color="primary">
            {getText('totalMemory', language)} {(total || 0).toFixed(2)} MB
          </Typography>
        </Box>
        
        {/* Visualization mode tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={visMode} onChange={handleTabChange} centered>
            <Tab label={getText('barChartTab', language)} />
            <Tab label={getText('proportionTab', language)} />
            <Tab label={getText('pieChartTab', language)} />
          </Tabs>
        </Box>
        
        {/* Visualization content based on selected tab */}
        <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 2, minHeight: 400 }}>
          {/* Bar Chart Visualization */}
          {visMode === 0 && (
            <Box sx={{ height: 400 }}>
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                {getText('memoryDistribution', language)}
              </Typography>
              <ResponsiveContainer width="100%" height="90%">
                <BarChart
                  data={chartCategories}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis 
                    type="number" 
                    unit=" MB" 
                    tickFormatter={(value) => value.toFixed(1)}
                  />
                  <YAxis 
                    type="category" 
                    dataKey="name" 
                    width={120}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip 
                    formatter={(value) => [`${value.toFixed(2)} MB (${(value / total * 100).toFixed(1)}%)`, getText('memoryValue', language)]}
                    labelStyle={{ fontWeight: 'bold' }}
                  />
                  <Bar dataKey="value" name={getText('memoryUsage', language)}>
                    {chartCategories.map((entry) => (
                      <Cell 
                        key={`cell-${entry.key}`} 
                        fill={entry.color || COLORS[entry.key.replace('other_memory_', '')] || COLORS.other}
                        // Add a border for better distinction
                        stroke="#fff"
                        strokeWidth={1}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Box>
          )}
          
          {/* Proportion Visualization */}
          {visMode === 1 && (
            <Box sx={{ height: 'auto', maxHeight: 600, overflow: 'auto' }}>
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                {getText('memoryDistributionRatio', language)}
              </Typography>
              
              {/* Memory overview - Top level categories */}
              <Box sx={{ mb: 4 }}>
                <Typography variant="body2" sx={{ mb: 1, fontWeight: 'bold' }}>
                  {getText('mainMemoryCategories', language)}
                </Typography>
                <Box sx={{ 
                  height: 60, 
                  display: 'flex', 
                  borderRadius: 1,
                  overflow: 'hidden',
                  boxShadow: 'inset 0 0 5px rgba(0,0,0,0.1)'
                }}>
                  {primaryCategories.map((cat) => (
                    cat.value > 0 ? (
                      <Box 
                        key={`block-${cat.key}`}
                        sx={{
                          width: `${(cat.value / total) * 100}%`,
                          height: '100%',
                          bgcolor: cat.color || COLORS[cat.key] || COLORS.other,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: 'white',
                          flexDirection: 'column',
                          p: 1,
                          textAlign: 'center',
                          position: 'relative',
                          '&::after': getPattern(cat.key).pattern === 'striped' ? {
                            content: '""',
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0,
                            background: `repeating-linear-gradient(45deg, rgba(255,255,255,0.15), rgba(255,255,255,0.15) 10px, rgba(255,255,255,0) 10px, rgba(255,255,255,0) 20px)`,
                            zIndex: 1
                          } : {}
                        }}
                      >
                        <Typography variant="body2" sx={{ fontWeight: 'bold', zIndex: 2 }}>
                          {cat.name}
                        </Typography>
                        <Typography variant="body2" sx={{ zIndex: 2 }}>
                          {cat.value.toFixed(1)} MB ({((cat.value / total) * 100).toFixed(1)}%)
                        </Typography>
                      </Box>
                    ) : null
                  ))}
                </Box>
              </Box>
              
              {/* Model States Breakdown */}
              {primaryCategories[1]?.children?.length > 0 && primaryCategories[1].value > 0 && (
                <Box sx={{ mb: 4 }}>
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 'bold' }}>
                    {getText('modelStatesBreakdown', language)}
                  </Typography>
                  <Box sx={{ 
                    height: 60, 
                    display: 'flex', 
                    borderRadius: 1,
                    overflow: 'hidden',
                    boxShadow: 'inset 0 0 5px rgba(0,0,0,0.1)'
                  }}>
                    {primaryCategories[1].children
                      .filter(cat => cat.value > 0)
                      .map((cat) => (
                        <Box 
                          key={`states-${cat.key}`}
                          sx={{
                            width: `${(cat.value / primaryCategories[1].value) * 100}%`,
                            height: '100%',
                            bgcolor: cat.color || COLORS[cat.key] || COLORS.other,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'white',
                            flexDirection: 'column',
                            p: 1,
                            textAlign: 'center',
                            border: '1px solid rgba(255,255,255,0.3)'
                          }}
                        >
                          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                            {cat.name}
                          </Typography>
                          <Typography variant="body2">
                            {cat.value.toFixed(1)} MB ({((cat.value / primaryCategories[1].value) * 100).toFixed(1)}%)
                          </Typography>
                        </Box>
                      ))}
                  </Box>
                </Box>
              )}
              
              {/* Legend */}
              <Box sx={{ mt: 3, mb: 2 }}>
                <Typography variant="body2" sx={{ mb: 1, fontWeight: 'bold' }}>
                  {language === LANGUAGES.ZH ? '图例' : 'Legend'}
                </Typography>
                <Box sx={{ 
                  display: 'flex', 
                  flexWrap: 'wrap', 
                  gap: 2
                }}>
                  {allCategories
                    .filter(cat => cat.value > 0)
                    .map((cat) => (
                      <Box
                        key={`legend-${cat.key}`}
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          mb: 1
                        }}
                      >
                        <Box 
                          sx={{ 
                            width: 16, 
                            height: 16, 
                            bgcolor: cat.color || COLORS[cat.key.replace('other_memory_', '')] || COLORS.other,
                            mr: 1,
                            borderRadius: '3px',
                            border: '1px solid rgba(0,0,0,0.1)'
                          }} 
                        />
                        <Typography variant="caption" sx={{ fontWeight: cat.isMainCategory ? 'bold' : 'normal' }}>
                          {cat.name}
                        </Typography>
                      </Box>
                    ))}
                </Box>
              </Box>
            </Box>
          )}
          
          {/* 饼图可视化 */}
          {visMode === 2 && (
            <Box sx={{ height: 450 }}>
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                {expandedCategory === 'model_states' 
                  ? getText('modelStatesDetails', language)
                  : getText('memoryDistributionPie', language)}
              </Typography>
              
              {expandedCategory && (
                <Typography 
                  variant="body2" 
                  sx={{ mb: 2, cursor: 'pointer', color: 'primary.main', display: 'inline-block' }}
                  onClick={() => setExpandedCategory(null)}
                >
                  ← {getText('backToMainView', language)}
                </Typography>
              )}
              
              <Box sx={{ height: 'calc(100% - 40px)', position: 'relative' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart margin={{ top: 40, right: 30, left: 30, bottom: 30 }}>
                    {/* 添加图例到顶部 */}
                    <Legend 
                      verticalAlign="top" 
                      height={36} 
                      iconSize={10}
                      iconType="circle"
                      layout="horizontal"
                      wrapperStyle={{ 
                        paddingBottom: '15px',
                        fontSize: '12px'
                      }}
                    />
                    <Pie
                      activeIndex={activeIndex}
                      activeShape={renderActiveShape}
                      data={pieData.filter(item => item.value > 0)}
                      cx="50%"
                      cy="50%"
                      innerRadius={70}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      nameKey="name"
                      onMouseEnter={(data, index) => setActiveIndex(index)}
                      onMouseLeave={() => setActiveIndex(null)}
                      onClick={(data, index) => handlePieClick(data, index)}
                      label={false} // 移除外部标签，让图表更清晰
                      labelLine={false}
                    >
                      {pieData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.color || COLORS[entry.key] || COLORS.other}
                          style={{ cursor: 'pointer' }}
                        />
                      ))}
                    </Pie>
                    <Tooltip 
                      formatter={(value) => [`${value.toFixed(2)} MB (${(value / total * 100).toFixed(1)}%)`, getText('memoryValue', language)]}
                    />
                  </PieChart>
                </ResponsiveContainer>
                
                {!expandedCategory && (
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      position: 'absolute', 
                      bottom: 10, 
                      left: 0, 
                      right: 0, 
                      textAlign: 'center',
                      color: 'text.secondary',
                      px: 2,
                      fontSize: '0.85rem',
                      lineHeight: 1.5,
                      maxWidth: '80%',
                      mx: 'auto',
                      wordBreak: 'keep-all',
                      whiteSpace: 'normal',
                      borderRadius: 1,
                      py: 0.5
                    }}
                  >
                    {getText('clickForDetails', language)}
                  </Typography>
                )}
              </Box>
            </Box>
          )}
        </Box>
      </Box>
    </Paper>
  );
}

export default MemoryVisualization;