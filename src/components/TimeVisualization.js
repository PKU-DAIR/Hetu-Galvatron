import React, { useState } from 'react';
import { Card, CardContent, Typography, Grid, Divider, CircularProgress, Box, 
  Tabs, Tab, Paper, Tooltip, useTheme, IconButton, Chip } from '@mui/material';
import { useLanguage } from '../context/LanguageContext';
import { getText } from '../utils/translations';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, 
  Legend, Tooltip as RechartsTooltip, PieChart, Pie, Cell
} from 'recharts';

// Device mapping for display
const DEVICE_NAME_MAP = {
  '4090': 'NVIDIA RTX 4090',
  'a100': 'NVIDIA A100',
  'h100': 'NVIDIA H100',
  'a10g': 'NVIDIA A10G',
  'v100': 'NVIDIA V100',
  't4': 'NVIDIA T4',
  'a6000': 'NVIDIA A6000',
};

const TimeVisualization = ({ timeData, loading, deviceConfig, usingCustomDevice }) => {
  const { language } = useLanguage();
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  if (loading) {
    return (
      <Card sx={{ minWidth: 275, marginTop: 2, boxShadow: theme.shadows[4] }}>
        <CardContent>
          <Typography variant="h5" component="div" fontWeight="500">
            {getText('timeVisualizationTitle', language)}
          </Typography>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (!timeData) {
    return (
      <Card sx={{ minWidth: 275, marginTop: 2, boxShadow: theme.shadows[4] }}>
        <CardContent>
          <Typography variant="h5" component="div" fontWeight="500">
            {getText('timeVisualizationTitle', language)}
          </Typography>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              {getText('noTimeDataAvailable', language)}
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // Format numbers for display
  const formatNumber = (num) => {
    if (num < 0.01) {
      return num.toExponential(2);
    }
    return num.toFixed(2);
  };

  // Prepare data for bar chart
  const barChartData = [
    {
      name: getText('forwardTime', language),
      value: timeData.forward_time,
      fill: '#8884d8',
      category: 'computation'
    },
    {
      name: getText('backwardTime', language),
      value: timeData.backward_time,
      fill: '#8884d8',
      category: 'computation'
    },
    {
      name: getText('dpCommTime', language),
      value: timeData.dp_communication_time,
      fill: '#82ca9d',
      category: 'communication'
    },
    {
      name: getText('tpCommTime', language),
      value: timeData.tp_communication_time,
      fill: '#82ca9d',
      category: 'communication'
    },
    {
      name: getText('ppCommTime', language),
      value: timeData.pp_communication_time,
      fill: '#82ca9d',
      category: 'communication'
    }
  ];

  // Prepare data for pie chart
  const pieChartData = [
    { name: getText('computationTime', language), value: timeData.forward_time + timeData.backward_time, fill: '#8884d8' },
    { name: getText('communicationTime', language), value: timeData.dp_communication_time + timeData.tp_communication_time + timeData.pp_communication_time, fill: '#82ca9d' }
  ];

  // Calculate totals
  const totalComputationTime = timeData.forward_time + timeData.backward_time;
  const totalCommunicationTime = timeData.dp_communication_time + timeData.tp_communication_time + timeData.pp_communication_time;
  const overlapEfficiency = 100 * (1 - (totalComputationTime + totalCommunicationTime) / timeData.iteration_time);

  // COLORS
  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088FE'];

  // Custom Tooltip for bar chart
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Paper sx={{ p: 1, boxShadow: theme.shadows[3], backgroundColor: 'rgba(255, 255, 255, 0.9)' }}>
          <Typography variant="subtitle2" color="text.primary">{label}</Typography>
          <Typography variant="body2" color="text.secondary">
            {`${payload[0].value.toFixed(2)} ms`}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  return (
    <Card sx={{ 
      minWidth: 275, 
      marginTop: 2,
      boxShadow: theme.shadows[4],
      borderRadius: '12px',
      overflow: 'hidden',
      transition: 'all 0.3s ease-in-out'
    }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h5" component="div" fontWeight="500" color={theme.palette.primary.main}>
              {getText('timeVisualizationTitle', language)}
            </Typography>
            
            {/* Show device information if custom device is used */}
            {usingCustomDevice && deviceConfig && (
              <Box sx={{ display: 'flex', mt: 1, gap: 1, flexWrap: 'wrap' }}>
                <Chip 
                  label={`${DEVICE_NAME_MAP[deviceConfig.deviceType] || deviceConfig.deviceType}`}
                  color="secondary"
                  size="small"
                  variant="outlined"
                />
                <Chip 
                  label={`${deviceConfig.deviceCount} ${deviceConfig.deviceCount === 1 ? 
                    getText('deviceSingular', language) : 
                    getText('devicePlural', language)}`}
                  color="secondary" 
                  size="small"
                  variant="outlined"
                />
              </Box>
            )}
          </Box>
          
          <Box sx={{ 
            bgcolor: theme.palette.background.paper, 
            borderRadius: '20px',
            border: `1px solid ${theme.palette.divider}`
          }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              indicatorColor="primary"
              textColor="primary"
              sx={{ 
                '& .MuiTabs-indicator': { 
                  height: 3,
                  borderRadius: '3px'
                } 
              }}
            >
              <Tab label={getText('barChartTab', language)} sx={{ fontSize: '0.8rem' }} />
              <Tab label={getText('pieChartTab', language)} sx={{ fontSize: '0.8rem' }} />
              <Tab label={getText('timeDetailsTab', language)} sx={{ fontSize: '0.8rem' }} />
            </Tabs>
          </Box>
        </Box>
        
        <Divider sx={{ my: 2 }} />
        
        {/* Bar Chart */}
        <Box sx={{ height: 350, display: activeTab === 0 ? 'block' : 'none' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={barChartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="name" 
                angle={-45} 
                textAnchor="end" 
                height={70} 
                tick={{ fontSize: 12 }}
                stroke={theme.palette.text.secondary}
              />
              <YAxis 
                label={{ 
                  value: 'Time (ms)', 
                  angle: -90, 
                  position: 'insideLeft', 
                  style: { textAnchor: 'middle' } 
                }}
                tick={{ fontSize: 12 }}
                stroke={theme.palette.text.secondary}
              />
              <RechartsTooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ paddingTop: 20 }} />
              <Bar dataKey="value" name="Time (ms)" fill="#8884d8">
                {barChartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Box>
        
        {/* Pie Chart */}
        <Box sx={{ height: 350, display: activeTab === 1 ? 'block' : 'none' }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieChartData}
                cx="50%"
                cy="50%"
                labelLine={true}
                outerRadius={120}
                fill="#8884d8"
                dataKey="value"
                nameKey="name"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {pieChartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <RechartsTooltip formatter={(value) => `${value.toFixed(2)} ms`} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Box>
        
        {/* Detailed view */}
        <Box sx={{ display: activeTab === 2 ? 'block' : 'none' }}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper 
                elevation={3} 
                sx={{ 
                  p: 2, 
                  borderRadius: '12px',
                  borderLeft: `4px solid ${theme.palette.primary.main}`,
                  backgroundColor: theme.palette.background.paper
                }}
              >
                <Typography variant="h6" color="primary" sx={{ mb: 2, fontWeight: 500 }}>
                  {getText('computationTime', language)}
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={7}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>{getText('forwardTime', language)}</Typography>
                  </Grid>
                  <Grid item xs={5}>
                    <Typography variant="body2" align="right" sx={{ fontWeight: 700 }}>
                      {formatNumber(timeData.forward_time)} ms
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={7}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>{getText('backwardTime', language)}</Typography>
                  </Grid>
                  <Grid item xs={5}>
                    <Typography variant="body2" align="right" sx={{ fontWeight: 700 }}>
                      {formatNumber(timeData.backward_time)} ms
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                  </Grid>
                  
                  <Grid item xs={7}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{getText('totalComputation', language)}</Typography>
                  </Grid>
                  <Grid item xs={5}>
                    <Box sx={{ 
                      display: 'flex', 
                      justifyContent: 'flex-end', 
                      alignItems: 'center', 
                      bgcolor: `${theme.palette.primary.light}20`,
                      p: 0.5,
                      borderRadius: '4px'
                    }}>
                      <Typography variant="body2" align="right" sx={{ fontWeight: 700, color: theme.palette.primary.main }}>
                        {formatNumber(totalComputationTime)} ms
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Paper 
                elevation={3} 
                sx={{ 
                  p: 2, 
                  borderRadius: '12px',
                  borderLeft: `4px solid ${theme.palette.secondary.main}`,
                  backgroundColor: theme.palette.background.paper
                }}
              >
                <Typography variant="h6" color="secondary" sx={{ mb: 2, fontWeight: 500 }}>
                  {getText('communicationTime', language)}
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={7}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>{getText('dpCommTime', language)}</Typography>
                  </Grid>
                  <Grid item xs={5}>
                    <Typography variant="body2" align="right" sx={{ fontWeight: 700 }}>
                      {formatNumber(timeData.dp_communication_time)} ms
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={7}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>{getText('tpCommTime', language)}</Typography>
                  </Grid>
                  <Grid item xs={5}>
                    <Typography variant="body2" align="right" sx={{ fontWeight: 700 }}>
                      {formatNumber(timeData.tp_communication_time)} ms
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={7}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>{getText('ppCommTime', language)}</Typography>
                  </Grid>
                  <Grid item xs={5}>
                    <Typography variant="body2" align="right" sx={{ fontWeight: 700 }}>
                      {formatNumber(timeData.pp_communication_time)} ms
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                  </Grid>
                  
                  <Grid item xs={7}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{getText('totalCommunication', language)}</Typography>
                  </Grid>
                  <Grid item xs={5}>
                    <Box sx={{ 
                      display: 'flex', 
                      justifyContent: 'flex-end', 
                      alignItems: 'center', 
                      bgcolor: `${theme.palette.secondary.light}20`,
                      p: 0.5,
                      borderRadius: '4px'
                    }}>
                      <Typography variant="body2" align="right" sx={{ fontWeight: 700, color: theme.palette.secondary.main }}>
                        {formatNumber(totalCommunicationTime)} ms
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>
          </Grid>
          
          <Paper 
            elevation={4} 
            sx={{ 
              p: 2, 
              mt: 3, 
              borderRadius: '12px',
              borderTop: `4px solid ${theme.palette.success.main}`,
              backgroundColor: theme.palette.background.paper
            }}
          >
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  p: 1,
                  borderRadius: '8px',
                  border: `1px dashed ${theme.palette.primary.main}30`
                }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, flexGrow: 1 }}>
                    {getText('totalIterationTime', language)}
                  </Typography>
                  <Typography variant="h5" color="primary.dark" sx={{ fontWeight: 700 }}>
                    {formatNumber(timeData.iteration_time)} ms
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  p: 1,
                  borderRadius: '8px',
                  border: `1px dashed ${theme.palette.secondary.main}30`
                }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, flexGrow: 1 }}>
                    {getText('throughput', language)}
                  </Typography>
                  <Typography variant="h5" color="secondary.dark" sx={{ fontWeight: 700 }}>
                    {formatNumber(timeData.samples_per_second)} samples/s
                  </Typography>
                </Box>
              </Grid>
              
              {/* <Grid item xs={12}>
                <Divider sx={{ my: 1 }} />
                <Box sx={{ 
                  mt: 1, 
                  p: 1, 
                  borderRadius: '8px', 
                  backgroundColor: `${theme.palette.info.light}10`,
                  border: `1px solid ${theme.palette.info.light}40`
                }}>
                  <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                    {getText('overlapEfficiency', language)}: <span style={{ fontWeight: 700, color: theme.palette.info.main }}>{formatNumber(overlapEfficiency)}%</span>
                  </Typography>
                </Box>
              </Grid> */}
            </Grid>
          </Paper>
        </Box>
        
      </CardContent>
    </Card>
  );
};

export default TimeVisualization;