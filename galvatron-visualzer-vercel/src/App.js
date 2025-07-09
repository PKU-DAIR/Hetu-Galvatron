import React, { useState, useEffect } from "react";
import {
    Container,
    CssBaseline,
    Typography,
    Box,
    AppBar,
    Toolbar,
} from "@mui/material";
import ConfigPanel from "./components/ConfigPanel";
import MemoryVisualization from "./components/MemoryVisualization";
import TimeVisualization from "./components/TimeVisualization";
import MemoryCostModel from "./models/MemoryCostModel";
// import TimeCostModel from './models/TimeCostModel'; // This was used by the removed calculateTimeResults
import MemoryTreemap from "./components/MemoryTreemap";
import ConfigSelector from "./components/ConfigSelector";
import {
    LanguageProvider,
    useLanguage,
    LANGUAGES,
} from "./context/LanguageContext";
import LanguageToggle from "./components/LanguageToggle";
import LanguageInstructions from "./components/LanguageInstructions";
import { getText } from "./utils/translations";
// Removed applyDevicePreset as it was unused
import { extractHardwareParams, DEVICE_PRESETS } from "./utils/configParser";

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
        checkpoint: false,
    });

    // Store raw configuration data
    const [rawConfigData, setRawConfigData] = useState(null);

    // Memory calculation results
    const [memoryResults, setMemoryResults] = useState(null);
    // Error information
    const [memoryError, setMemoryError] = useState(null);

    // Time calculation results
    const [timeResults, setTimeResults] = useState(null);
    const [timeError, setTimeError] = useState(null);
    const [timeLoading, setTimeLoading] = useState(false);

    // Handle config file loading
    const handleConfigLoaded = (data) => {
        // Store raw config data
        setRawConfigData(data);

        // Extract parameters from config file and update panel
        if (data && data.model_config) {
            const modelConfig = data.model_config;

            // Extract model parameters
            let newConfig = {
                ...config,
                attention_heads: modelConfig.n_heads || config.attention_heads,
                hidden_dim: modelConfig.dim || config.hidden_dim,
                ff_dim: modelConfig.ffn_dim || config.ff_dim,
                num_layers: modelConfig.n_layers || config.num_layers,
                vocab_size: modelConfig.vocab_size || config.vocab_size,
            };

            // Extract sequence length if available
            // Find sequence lengths in config
            const seqLengths = [];
            for (const key in data) {
                if (key.match(/layertype_\d+(?:_sp)?$/)) {
                    const layerKey = key;
                    Object.keys(data[layerKey]).forEach((seq) => {
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

            // Extract hardware parameters from JSON config and set to custom mode
            try {
                const hardwareParams = extractHardwareParams(data);
                newConfig = {
                    ...newConfig,
                    ...hardwareParams,
                    hardware_preset: "Custom", // Set to custom when loading JSON
                    device_count: "custom", // Set device count to custom
                };
                console.log(
                    "Hardware parameters extracted from config file:",
                    hardwareParams
                );
            } catch (error) {
                console.warn(
                    "Could not extract hardware parameters:",
                    error.message
                );
                // Set to custom mode anyway
                newConfig = {
                    ...newConfig,
                    hardware_preset: "Custom",
                    device_count: "custom",
                    ...DEVICE_PRESETS.Custom, // Use custom defaults
                };
            }

            // Update config
            setConfig(newConfig);
            console.log("Parameters extracted from config file:", newConfig);
        }
    };

    // The calculateTimeResults function was here and has been removed as it was unused.
    // The error 'deviceConfig' is not defined was within this function.

    // Fetch time calculation results from the backend API
    const fetchTimeResults = async (configData) => {
        try {
            setTimeLoading(true);

            // Prepare hardware parameters from config
            const hardwareParams = {
                forward_computation_time:
                    configData.forward_computation_time || 10.0,
                bct_fct_coe: configData.bct_fct_coe || 2.0,
                dp_overlap_coe: configData.dp_overlap_coe || 1.0,
                bct_overlap_coe: configData.bct_overlap_coe || 1.0,
                allreduce_bandwidth: configData.allreduce_bandwidth || 100.0,
                p2p_bandwidth: configData.p2p_bandwidth || 300.0,
                sp_space: configData.sp_space || "tp+sp",
                async_grad_reduce: configData.async_grad_reduce || false,
                device_count:
                    configData.device_count === "custom"
                        ? configData.total_gpus
                        : configData.device_count || 8,
            };

            console.log("Sending hardware params to backend:", hardwareParams);

            const response = await fetch("/api/calculate_time", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    config: configData,
                    ...hardwareParams, // Send hardware parameters directly
                }),
            });

            if (!response.ok) {
                throw new Error(
                    `API request failed with status ${response.status}`
                );
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            setTimeResults(data);
            setTimeError(null);
        } catch (error) {
            console.error("Time calculation error:", error);
            setTimeResults(null);
            setTimeError({
                message: error.message,
                needsConfigFile: false,
            });
        } finally {
            setTimeLoading(false);
        }
    };

    // Recalculate memory when config changes
    useEffect(() => {
        try {
            // console.log("Recalculating memory based on config:", config);
            // Clear previous errors
            setMemoryError(null);
            // MARK
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
                isConfigMissing: error.message.includes(
                    "not found in rawConfig"
                ),
                needsConfigFile: !rawConfigData,
            });
        }
    }, [config, rawConfigData]);

    // Fetch/calculate time results when config changes
    useEffect(() => {
        // Always use backend now since it accepts hardware parameters directly
        fetchTimeResults(config);
    }, [config]);

    const handleConfigChange = (newConfig) => {
        // console.log("Configuration changed:", newConfig);
        setConfig(newConfig);
    };

    return (
        <React.Fragment>
            <CssBaseline />
            <AppBar position="sticky" color="default" elevation={1}>
                <Toolbar sx={{ justifyContent: "space-between" }}>
                    <Typography variant="h6" color="#000000">
                        {getText("appTitle", language)}
                    </Typography>
                    <LanguageToggle />
                </Toolbar>
            </AppBar>
            <Container maxWidth="lg">
                <Box sx={{ my: 4 }}>
                    <Typography
                        variant="h4"
                        component="h1"
                        gutterBottom
                        align="center"
                    >
                        {language === LANGUAGES.ZH
                            ? "Galvatron 内存估计可视化工具"
                            : "Galvatron Memory Estimation Tool"}
                    </Typography>

                    <Typography variant="body1" color="text.secondary">
                        {language === LANGUAGES.ZH
                            ? "本工具可帮助您分析和可视化Galvatron内存分析结果。选择模型和精度配置来开始。"
                            : "This tool helps you analyze and visualize Galvatron memory analysis results. Choose a model and precision configuration to start."}
                    </Typography>

                    <ConfigSelector onConfigLoaded={handleConfigLoaded} />

                    <Box
                        sx={{
                            display: "flex",
                            flexDirection: { xs: "column", md: "row" },
                            gap: 3,
                            mt: 3,
                        }}
                    >
                        <Box sx={{ width: { xs: "100%", md: "58%" } }}>
                            {/* Memory treemap visualization */}
                            <MemoryTreemap
                                memoryData={memoryResults}
                                config={config}
                                error={memoryError}
                            />
                        </Box>

                        <Box sx={{ width: { xs: "100%", md: "38%" } }}>
                            {/* Configuration panel */}
                            <ConfigPanel
                                config={config}
                                onConfigChange={handleConfigChange}
                            />
                        </Box>
                    </Box>

                    {/* Bottom chart visualizations */}
                    <Box
                        sx={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 2,
                            mt: 3,
                        }}
                    >
                        {/* Memory visualization */}
                        <MemoryVisualization memoryResults={memoryResults} />

                        {/* Time visualization */}
                        <TimeVisualization
                            timeData={timeResults}
                            loading={timeLoading}
                            error={timeError}
                        />
                    </Box>

                    {/* Footer */}
                    <Box
                        sx={{
                            mt: 4,
                            textAlign: "center",
                            color: "text.secondary",
                        }}
                    >
                        <Typography variant="body2">
                            {getText("basedOnModel", language)}
                        </Typography>

                        {/* 添加GitHub和文档链接 */}
                        <Box
                            sx={{
                                display: "flex",
                                justifyContent: "center",
                                gap: 3,
                                mt: 1,
                                mb: 1,
                            }}
                        >
                            <Typography
                                variant="body2"
                                component="a"
                                href="https://github.com/PKU-DAIR/Hetu-Galvatron"
                                target="_blank"
                                rel="noopener noreferrer"
                                sx={{
                                    color: "primary.main",
                                    textDecoration: "none",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 0.5,
                                    "&:hover": { textDecoration: "underline" },
                                }}
                            >
                                <Box
                                    component="img"
                                    src="https://github.githubassets.com/favicons/favicon.svg"
                                    sx={{ width: 16, height: 16 }}
                                />
                                {getText("githubLink", language)}
                            </Typography>

                            <Typography
                                variant="body2"
                                component="a"
                                href="https://hetu-galvatron.readthedocs.io/"
                                target="_blank"
                                rel="noopener noreferrer"
                                sx={{
                                    color: "primary.main",
                                    textDecoration: "none",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 0.5,
                                    "&:hover": { textDecoration: "underline" },
                                }}
                            >
                                <Box
                                    component="img"
                                    src="https://readthedocs.org/favicon.ico"
                                    sx={{ width: 16, height: 16 }}
                                />
                                {getText("documentationLink", language)}
                            </Typography>
                        </Box>

                        <Typography variant="body2">
                            © {new Date().getFullYear()} Galvatron{" "}
                            {getText("teamCopyright", language)}
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
