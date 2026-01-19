# LAER-MoE Artifact

> **Note**: This repository contains only the LAER-MoE code. For the complete Artifact Evaluation (AE) package including all baselines, please refer to [this repository](https://github.com/Fizzmy/LAER-MoE-AE).

This repo contains the actifact of paper "LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training", including codes and scripts for reproducing all experiments in the paper.

## Requirements

### Hardware dependencies

We conduct our experiments on a 4-node GPU cluster, with each node containing 8 NVIDIA A100 80GB GPUs. Within nodes, GPUs within a node are connected via NVLink, and nodes are interconnected via Infiniband. The peak unidirectional communication bandwidth intra-node is 450 GB/s, and inter-node is 800 Gbps.

### Software dependencies

Following toolkits are requireds: python=3.9.2, CUDA=12.1, torch=2.1.0-cu121. For baseline, cmake >= 3.21 is also required.

## Installation

> For artifact evaluation, users can use existing virtual environment directly.(conda activate AE)

First, we use conda to create a virtual environment and modify environment variables (if necessary).

```
# Create virtual environment
conda create -n laer-moe python=3.9.2
conda activate laer-moe

# Modify environment variables
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

```

Second, to install the artifact, users need to install torch, flash-attn and apex. Moreover, to be able to run the baseline, users also need to install Transformer-Engine.

```
# Install torch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn
pip install packaging
pip install ninja
pip install psutil
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.5.8
python setup.py install
cd csrc/layer_norm
pip install . --no-build-isolation
# or use wheel


# Install apex
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 312acb4
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# or use wheel



# Install Transformer Engine
pip install pybind11
pip install einops
git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git checkout 7f2afaa
git submodule update --init --recursive
NVTE_FRAMEWORK=pytorch pip install --no-build-isolation .
# or use wheel
```

Finally, install laer-moe:

```
cd LAER-MoE
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Prepare datasets

We placed the processed dataset in the `datasets/processed` directory and the raw dataset in the `datasets/raw`. The processed dataset was obtained using the following command:
```
# for wikitext
cd Megatron
mkdir -p ../datasets/processed/wikitext
python tools/preprocess_data.py \
       --input "../datasets/raw/wikitext/wikitext.json" \
       --partitions 1 \
       --output-prefix ../datasets/processed/wikitext/mixtral \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model ../tokenizers/mixtral \
       --workers 20 \
       --append-eod
# for C4
cd Megatron
mkdir -p ../datasets/processed/C4
python tools/preprocess_data.py \
       --input "../datasets/raw/C4/c4-train*" \
       --partitions 5 \
       --output-prefix ../datasets/processed/C4/mixtral \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model ../tokenizers/mixtral \
       --workers 20 \
       --append-eod
```
where `../datasets/raw/wikitext/wikitext.json` and `../datasets/raw/C4/c4-train*` are the paths to the wikitext and C4 raw datasets, respectively. `../datasets/processed/wikitext/mixtral` and `../datasets/processed/C4/mixtral` are the paths to the processed wikitext and C4 datasets, respectively. `../tokenizers/mixtral` is the path to the tokenizer.

## Expertiment workflow


### End-to-End Performance

#### Prepare checkpoints

We placed the checkpoints in the `checkpoints` directory, where `checkpoints/megatron/{name}` is in Megatron-LM format adn `checkpoints/laer/{name}` is in LAER-MoE format. To maintain the same training accuracy, LAER-MoE and the baseline start training from the same checkpoint. We use the following command to save the Megatron-LM checkpoint and convert it to LAER-MoE checkpoint:

```
cd LAER-MoE
bash scripts_ae/save_checkpoint.sh {name} {type}
```
where `{name}` is the name of the model, e.g., `mixtral-8x7b-e8k2`, `{type}` is the type of the checkpoint, which can be one of `e2e`, `convergence`. For `e2e`, the checkpoint will be saved in `checkpoints/megatron/{name}` and `checkpoints/laer/{name}`; for `convergence`, the checkpoint will be saved in `checkpoints/megatron_convergence/{name}` and `checkpoints/laer_convergence/{name}`. To ensure the best performance of the baseline, we use different parallel strategies for end-to-end analysis and convergence study, so that users need to generate the checkpoints respectively for each case.

When the command is finished, there will be a `checkpoints/megatron/{name}` and `checkpoints/laer/{name}` directory, which is the Megatron-LM checkpoint and LAER-MoE checkpoint respectively. For `convergence`, there will be a `checkpoints/megatron_convergence/{name}` and `checkpoints/laer_convergence/{name}` directory, which is the Megatron-LM checkpoint and LAER-MoE checkpoint respectively.

#### End-to-End Evaluation (Figure 8)

`e2e.sh` will start the end-to-end evaluation in §5.2. In the following command, the `approach` is the training approach to use, which can be one of `LAER, FLEX, FSDP, megatron`. `model_name` is the name of the model, which can be one of `mixtral-8x7b-e8k2`, `mixtral-8x7b-e16k4`, `mixtral-8x22b-e8k2`, `mixtral-8x22b-e16k4`, `qwen-8x7b-e8k2`, `qwen-8x7b-e16k4`. `aux_loss` is the auxiliary loss coefficient, which can be one of `0.0`, `1e-4`, and `dataset` is the dataset to evaluate on, which can be one of `wikitext`, `C4`. Each evaluation will be corresponding to a bar in Figure 8.

```
bash scripts_ae/e2e.sh <approach> <model_name> <aux_loss> <dataset>
```

We provide a script to run the end-to-end evaluation for all the models and all the approaches.
```
bash scripts_ae/e2e_all.sh
```

Users can modify the `e2e_all.sh` script to run the end-to-end evaluation for different models and different approaches.

#### Convergence study (Figure 9)

`convergence.sh` will start the convergence study in §5.2. In the following command, the `approach` is the training approach to use. `aux_loss` is the auxiliary loss coefficient. (`approach`, `aux_loss`) can be one of (`LAER, 1e-4`), (`Megatron, 1e-2`), (`Megatron, 1e-4`). We will evaluate on `wikitext` dataset. Each evaluation will be corresponding to a curve in Figure 9. For default setting, `iter` is 3000. 
> NOTE: This command will take a long time to run (16h for `Megatron, 1e-4`, 10h for `Megatron, 1e-2`, 9h for `LAER, 1e-4`), for artifact evaluation, users can set `iter` to 1500 to save time. But users still need to set `iter` to 3000 to get the best performance.
```
./scripts_ae/convergence.sh <approach> <aux_loss> <iter>
```

### Case study

#### Breakdown (Figure 10(a))

`breakdown.sh` will start the case study in §5.3. In the following command, the `approach` is the training approach to use, which can be one of `LAER, FLEX, FSDP`. `model_name` is the name of the model, which can be one of `mixtral-8x7b-e8k2`, `mixtral-8x7b-e16k4`. Each evaluation will be corresponding to a bar in Figure 10(a). For default setting, `aux_loss` is `0`, and `dataset` is `wikitext`.

```
./scripts_ae/breakdown.sh <approach> <model_name>
```

#### Maximum token counts (Figure 10(b))

`token_counts.sh` will start the case study in §5.3. In the following command, the `approach` is the training approach to use. `model_name` is the name of the model, which can be one of `mixtral-8x7b-e8k2`, `mixtral-8x7b-e16k4`. Each evaluation will be corresponding to a curve in Figure 10(b). For default setting, `aux_loss` is `0`, and `dataset` is `wikitext`.

```
./scripts_ae/token_counts.sh <approach> <model_name>
```

### Planner Performance (Figure 11)

`planner.sh` will start the planner performance in §5.4. In the following command, the `N` is the number of devices, and `C` is the capacity of experts per device. Each evaluation will be corresponding to a point in Figure 11.

```
./scripts_ae/planner.sh <N> <C>
```

Users can also run the planner.sh with no arguments to get the average time for each (N, C) pair by running the following command:
```
./scripts_ae/planner.sh
```

### Ablation study (Figure 12)

`ablation.sh` will start the ablation study in §5.5. In the following command, the `approach` is the training approach to use, which can be one of `LAER, no_even, no_pq, no_comm_opt, FSDP`. Each evaluation will be corresponding to a bar in Figure 12. For default setting, `model_name` is `mixtral-8x7b-e8k2`, `aux_loss` is `0`, and `dataset` is `wikitext`.

```
./scripts_ae/ablation.sh <approach>
```

## Plotting Figures

To analysis these data and plot figures presented in the paper, we also provide `plot.sh` script to plot the corresponding figure. For example, to plot Figure 8/9, we use the following command:

```
pip install pandas matplotlib
bash scripts_ae/plot.sh <figure_id> <type>
```

where `{figure_id}` is the id of the figure, which can be one of `8`, `9`, `10a`, `10b`, `11`, `12`. `{type}` is the type of the plot, which can be one of `default` or `new`. For `default`, the script will plot the figure with the data used in the paper. For `new`, the script will first analyze the new experiment results and then plot the figure.

## Evaluation and expected results

The absolute performance numbers may vary across different hardware and network configurations. However, we expect that the relative performance trends shown in Figures 8-12 should be reproducible, with the average speedup ratios remaining comparable to those reported in the paper.

## Experiment customization

Users can customize the evaluation scripts to test the system performance on other workloads. For example, users can use their own model architectures, datasets, or training parameters. Specifically, users can modify arguments when running \texttt{LAER-MoE/galvatron/models/moe/scripts/train.sh} to customize the experiments.

```
bash LAER-MoE/galvatron/models/moe/scripts/train.sh <model_name> <layer_num> <aux_loss> <approach> <dataset> <capacity> <batch_size> <chunk> <seq_length> <profile>
```

where the parameters are:

- `model_name`: model architecture (customizable, save configs in `configs/`)
- `layer_num`: number of layers
- `aux_loss`: auxiliary loss weight
- `approach`: training approach (`LAER`, `FLEX`, or `FSDP`)
- `dataset`: dataset name (`wikitext` or `C4`)
- `capacity`: capacity of experts per device
- `batch_size`: batch size
- `chunk`: chunk size
- `seq_length`: sequence length
- `profile`: whether to enable profiling

## Notes

- The experiments require a 4-node cluster with 32 A100 GPUs. Results may vary on different hardware configurations and bandwidth.
  
- To ensure comparable training convergence across different approaches (with relative loss error below 1e-3), we save checkpoints for each model configuration. This requires significant disk space (approximately 200 GB per model). Users can choose to train without loading checkpoints, but this may result in different routing results and loss trajectories.