import { LANGUAGES } from '../context/LanguageContext';

/**
 * Translation dictionary for UI text
 * Contains translations for all UI elements in Chinese and English
 */
const translations = {
  // Common
  appTitle: {
    [LANGUAGES.ZH]: 'Galvatron 内存可视化',
    [LANGUAGES.EN]: 'Galvatron Memory Visualizer'
  },
  
  // ConfigSelector
  configSelectorTitle: {
    [LANGUAGES.ZH]: 'Galvatron 内存配置选择器',
    [LANGUAGES.EN]: 'Galvatron Memory Config Selector'
  },
  selectModel: {
    [LANGUAGES.ZH]: '选择模型',
    [LANGUAGES.EN]: 'Select Model'
  },
  selectPrecision: {
    [LANGUAGES.ZH]: '选择精度',
    [LANGUAGES.EN]: 'Select Precision'
  },
  loadConfig: {
    [LANGUAGES.ZH]: '加载配置',
    [LANGUAGES.EN]: 'Load Config'
  },
  loading: {
    [LANGUAGES.ZH]: '加载中...',
    [LANGUAGES.EN]: 'Loading...'
  },
  uploadLocal: {
    [LANGUAGES.ZH]: '或者上传本地配置文件',
    [LANGUAGES.EN]: 'Or Upload Local Config File'
  },
  selectJsonFile: {
    [LANGUAGES.ZH]: '选择JSON配置文件',
    [LANGUAGES.EN]: 'Select JSON Config File'
  },
  fileSelected: {
    [LANGUAGES.ZH]: '已选择:',
    [LANGUAGES.EN]: 'Selected:'
  },
  
  // ConfigPanel
  configPanelTitle: {
    [LANGUAGES.ZH]: '模型配置',
    [LANGUAGES.EN]: 'Model Configuration'
  },
  basicParams: {
    [LANGUAGES.ZH]: '基本参数',
    [LANGUAGES.EN]: 'Basic Parameters'
  },
  advancedParams: {
    [LANGUAGES.ZH]: '高级参数',
    [LANGUAGES.EN]: 'Advanced Parameters'
  },
  mixedPrecision: {
    [LANGUAGES.ZH]: '混合精度训练',
    [LANGUAGES.EN]: 'Mixed Precision Training'
  },
  distributedParams: {
    [LANGUAGES.ZH]: '分布式训练参数',
    [LANGUAGES.EN]: 'Distributed Training Parameters'
  },
  attentionHeads: {
    [LANGUAGES.ZH]: '注意力头数',
    [LANGUAGES.EN]: 'Attention Heads'
  },
  hiddenDim: {
    [LANGUAGES.ZH]: '隐藏层维度',
    [LANGUAGES.EN]: 'Hidden Dimension'
  },
  ffnDim: {
    [LANGUAGES.ZH]: '前馈网络维度',
    [LANGUAGES.EN]: 'FFN Dimension'
  },
  vocabSize: {
    [LANGUAGES.ZH]: '词汇表大小',
    [LANGUAGES.EN]: 'Vocabulary Size'
  },
  enabled: {
    [LANGUAGES.ZH]: '开启',
    [LANGUAGES.EN]: 'Enabled'
  },
  disabled: {
    [LANGUAGES.ZH]: '关闭',
    [LANGUAGES.EN]: 'Disabled'
  },
  seqParallel: {
    [LANGUAGES.ZH]: '序列并行',
    [LANGUAGES.EN]: 'Sequence Parallel'
  },
  activationCheckpoint: {
    [LANGUAGES.ZH]: '激活检查点 (Checkpoint)',
    [LANGUAGES.EN]: 'Activation Checkpoint'
  },
  modelLayers: {
    [LANGUAGES.ZH]: '模型层数',
    [LANGUAGES.EN]: 'Model Layers'
  },
  seqLength: {
    [LANGUAGES.ZH]: '序列长度',
    [LANGUAGES.EN]: 'Sequence Length'
  },
  microBatchSize: {
    [LANGUAGES.ZH]: '微批次大小',
    [LANGUAGES.EN]: 'Micro Batch Size'
  },
  globalBatchSize: {
    [LANGUAGES.ZH]: '全局批次大小',
    [LANGUAGES.EN]: 'Global Batch Size'
  },
  chunks: {
    [LANGUAGES.ZH]: '累积次数',
    [LANGUAGES.EN]: 'Chunks'
  },
  parallelStrategies: {
    [LANGUAGES.ZH]: '并行策略配置',
    [LANGUAGES.EN]: 'Parallel Strategy Configuration'
  },
  totalGPUs: {
    [LANGUAGES.ZH]: '总GPU数量',
    [LANGUAGES.EN]: 'Total GPUs'
  },
  dataParallel: {
    [LANGUAGES.ZH]: '数据并行度 (DP)',
    [LANGUAGES.EN]: 'Data Parallel (DP)'
  },
  dpAutoCalc: {
    [LANGUAGES.ZH]: '(自动计算: Total GPUs / (TP * PP))',
    [LANGUAGES.EN]: '(Auto-calculated: Total GPUs / (TP * PP))'
  },
  tensorParallel: {
    [LANGUAGES.ZH]: '张量并行度 (TP)',
    [LANGUAGES.EN]: 'Tensor Parallel (TP)'
  },
  pipelineParallel: {
    [LANGUAGES.ZH]: '流水线并行度 (PP)',
    [LANGUAGES.EN]: 'Pipeline Parallel (PP)'
  },
  zeroStage: {
    [LANGUAGES.ZH]: 'ZeRO优化级别',
    [LANGUAGES.EN]: 'ZeRO Optimization Level'
  },
  zeroOff: {
    [LANGUAGES.ZH]: '关闭 (0)',
    [LANGUAGES.EN]: 'Off (0)'
  },
  zeroLevel: {
    [LANGUAGES.ZH]: '级别',
    [LANGUAGES.EN]: 'Level'
  },
  ppStage: {
    [LANGUAGES.ZH]: '流水线阶段 (PP Stage)',
    [LANGUAGES.EN]: 'Pipeline Stage (PP Stage)'
  },
  currentStage: {
    [LANGUAGES.ZH]: '当前阶段',
    [LANGUAGES.EN]: 'Current Stage'
  },
  parallelNote: {
    [LANGUAGES.ZH]: '注意: 序列并行设置根据配置文件自动确定。数据并行度(DP)根据总GPU数量自动计算，块数(Chunks)确保 MBS * DP * Chunks = GBS。当启用流水线并行时(PP>1)，可以选择查看特定流水线阶段的内存分配情况。',
    [LANGUAGES.EN]: 'Note: Sequence parallel is automatically determined based on the configuration file. Data Parallel (DP) is auto-calculated from total GPUs, and Chunks ensure that MBS * DP * Chunks = GBS. When pipeline parallel is enabled (PP>1), you can select a specific pipeline stage to view memory allocation.'
  },
  gpuConstraint: {
    [LANGUAGES.ZH]: 'GPU 分配约束: TP × PP × DP = 总 GPU 数',
    [LANGUAGES.EN]: 'GPU Allocation Constraint: TP × PP × DP = Total GPUs'
  },
  batchConstraint: {
    [LANGUAGES.ZH]: '批处理约束: MBS × DP × Chunks = GBS',
    [LANGUAGES.EN]: 'Batch Constraint: MBS × DP × Chunks = GBS'
  },
  constraintSatisfied: {
    [LANGUAGES.ZH]: '约束已满足',
    [LANGUAGES.EN]: 'Constraint Satisfied'
  },
  constraintWarning: {
    [LANGUAGES.ZH]: '约束未满足',
    [LANGUAGES.EN]: 'Constraint Not Satisfied'
  },
  value: {
    [LANGUAGES.ZH]: '值',
    [LANGUAGES.EN]: 'Value'
  },
  expected: {
    [LANGUAGES.ZH]: '期望',
    [LANGUAGES.EN]: 'Expected'
  },
  current: {
    [LANGUAGES.ZH]: '当前',
    [LANGUAGES.EN]: 'Current'
  },
  
  // MemoryVisualization
  memoryVisTitle: {
    [LANGUAGES.ZH]: '内存使用详情',
    [LANGUAGES.EN]: 'Memory Usage Details'
  },
  memoryAnalysis: {
    [LANGUAGES.ZH]: '内存使用分析',
    [LANGUAGES.EN]: 'Memory Usage Analysis'
  },
  totalMemory: {
    [LANGUAGES.ZH]: '总内存:',
    [LANGUAGES.EN]: 'Total Memory:'
  },
  memoryDistribution: {
    [LANGUAGES.ZH]: '内存分布明细',
    [LANGUAGES.EN]: 'Memory Distribution Details'
  },
  memoryUsage: {
    [LANGUAGES.ZH]: '内存使用量',
    [LANGUAGES.EN]: 'Memory Usage'
  },
  memoryValue: {
    [LANGUAGES.ZH]: '内存',
    [LANGUAGES.EN]: 'Memory'
  },
  waitingForData: {
    [LANGUAGES.ZH]: '等待内存数据计算...',
    [LANGUAGES.EN]: 'Waiting for memory data calculation...'
  },
  tryUploadConfig: {
    [LANGUAGES.ZH]: '尝试上传配置文件或调整参数以生成内存分析',
    [LANGUAGES.EN]: 'Try uploading a config file or adjusting parameters to generate memory analysis'
  },
  hoverInfo: {
    [LANGUAGES.ZH]: '悬停在区块上查看详细信息',
    [LANGUAGES.EN]: 'Hover over blocks for details'
  },
  currentSelection: {
    [LANGUAGES.ZH]: '当前选中:',
    [LANGUAGES.EN]: 'Current Selection:'
  },
  // Adding new visualization translations
  barChartTab: {
    [LANGUAGES.ZH]: '柱状图',
    [LANGUAGES.EN]: 'Bar Chart'
  },
  proportionTab: {
    [LANGUAGES.ZH]: '比例图', 
    [LANGUAGES.EN]: 'Proportion'
  },
  memoryDistributionRatio: {
    [LANGUAGES.ZH]: '内存分布比例',
    [LANGUAGES.EN]: 'Memory Distribution Ratio'
  },
  mainMemoryCategories: {
    [LANGUAGES.ZH]: '主要内存类别',
    [LANGUAGES.EN]: 'Main Memory Categories'
  },
  modelStatesBreakdown: {
    [LANGUAGES.ZH]: '模型状态细分',
    [LANGUAGES.EN]: 'Model States Breakdown'
  },
  allMemoryCategories: {
    [LANGUAGES.ZH]: '所有内存类别',
    [LANGUAGES.EN]: 'All Memory Categories'
  },
  
  // Memory categories
  activationMemory: {
    [LANGUAGES.ZH]: '激活内存',
    [LANGUAGES.EN]: 'Activation Memory'
  },
  parameterMemory: {
    [LANGUAGES.ZH]: '参数内存',
    [LANGUAGES.EN]: 'Parameter Memory'
  },
  gradientMemory: {
    [LANGUAGES.ZH]: '梯度内存',
    [LANGUAGES.EN]: 'Gradient Memory'
  },
  optimizerMemory: {
    [LANGUAGES.ZH]: '优化器内存',
    [LANGUAGES.EN]: 'Optimizer Memory'
  },
  modelStates: {
    [LANGUAGES.ZH]: '模型状态',
    [LANGUAGES.EN]: 'Model States'
  },
  gradAccumulate: {
    [LANGUAGES.ZH]: '梯度累积',
    [LANGUAGES.EN]: 'Gradient Accumulation'
  },
  otherActivation: {
    [LANGUAGES.ZH]: '其他激活内存',
    [LANGUAGES.EN]: 'Other Activation Memory'
  },
  otherParameter: {
    [LANGUAGES.ZH]: '其他参数内存',
    [LANGUAGES.EN]: 'Other Parameter Memory'
  },
  otherGradient: {
    [LANGUAGES.ZH]: '其他梯度内存',
    [LANGUAGES.EN]: 'Other Gradient Memory'
  },
  otherOptimizer: {
    [LANGUAGES.ZH]: '其他优化器内存',
    [LANGUAGES.EN]: 'Other Optimizer Memory'
  },
  otherModelStates: {
    [LANGUAGES.ZH]: '其他模型状态',
    [LANGUAGES.EN]: 'Other Model States'
  },
  otherGradAccumulate: {
    [LANGUAGES.ZH]: '其他梯度累积',
    [LANGUAGES.EN]: 'Other Gradient Accumulation'
  },
  
  // MemoryTreemap
  noDataMessage: {
    [LANGUAGES.ZH]: '暂无内存数据可视化',
    [LANGUAGES.EN]: 'No memory data visualization available'
  },
  uploadPrompt: {
    [LANGUAGES.ZH]: '请上传配置文件或调整参数以生成内存分析',
    [LANGUAGES.EN]: 'Please upload a config file or adjust parameters to generate memory analysis'
  },
  configMissing: {
    [LANGUAGES.ZH]: '配置文件缺少必要参数',
    [LANGUAGES.EN]: 'Config file missing required parameters'
  },
  cannotVisualize: {
    [LANGUAGES.ZH]: '无法生成内存可视化',
    [LANGUAGES.EN]: 'Cannot generate memory visualization'
  },
  
  // Errors and messages
  selectModelPrecision: {
    [LANGUAGES.ZH]: '请选择模型和精度',
    [LANGUAGES.EN]: 'Please select model and precision'
  },
  configLoadFailed: {
    [LANGUAGES.ZH]: '加载配置失败:',
    [LANGUAGES.EN]: 'Failed to load config:'
  },
  configLoadSuccess: {
    [LANGUAGES.ZH]: '已成功加载配置并更新参数',
    [LANGUAGES.EN]: 'Successfully loaded config and updated parameters'
  },
  fileReadFailed: {
    [LANGUAGES.ZH]: '读取文件失败',
    [LANGUAGES.EN]: 'Failed to read file'
  },
  fileParseError: {
    [LANGUAGES.ZH]: '解析文件失败:',
    [LANGUAGES.EN]: 'Failed to parse file:'
  },
  // 添加饼图相关的翻译
  pieChartTab: {
    [LANGUAGES.ZH]: '饼图',
    [LANGUAGES.EN]: 'Pie Chart'
  },
  memoryDistributionPie: {
    [LANGUAGES.ZH]: '内存分布',
    [LANGUAGES.EN]: 'Memory Distribution'
  },
  modelStatesDetails: {
    [LANGUAGES.ZH]: '模型状态内存详情',
    [LANGUAGES.EN]: 'Model States Details'
  },
  backToMainView: {
    [LANGUAGES.ZH]: '返回主视图',
    [LANGUAGES.EN]: 'Back to main view'
  },
  clickForDetails: {
    [LANGUAGES.ZH]: '点击"模型状态"扇区查看详细内存分布',
    [LANGUAGES.EN]: 'Click on "Model States" segment to see detailed breakdown'
  },
  
  // 底部链接
  githubLink: {
    [LANGUAGES.ZH]: 'GitHub 仓库',
    [LANGUAGES.EN]: 'GitHub Repository'
  },
  documentationLink: {
    [LANGUAGES.ZH]: '技术文档',
    [LANGUAGES.EN]: 'Documentation'
  },
  basedOnModel: {
    [LANGUAGES.ZH]: '基于 Galvatron 内存成本模型 | 数据仅供参考',
    [LANGUAGES.EN]: 'Based on Galvatron Memory Cost Model | Data for reference only'
  },
  teamCopyright: {
    [LANGUAGES.ZH]: '团队',
    [LANGUAGES.EN]: 'Team'
  },
  showDetails: {
    [LANGUAGES.ZH]: '显示细节',
    [LANGUAGES.EN]: 'Show Details'
  },
};

/**
 * Get text in the specified language
 * @param {string} key - Translation key
 * @param {string} language - Current language
 * @returns {string} - Translated text
 */
export const getText = (key, language) => {
  if (!translations[key]) {
    console.warn(`Missing translation key: ${key}`);
    return key;
  }
  return translations[key][language] || translations[key][LANGUAGES.ZH];
};

export default translations; 