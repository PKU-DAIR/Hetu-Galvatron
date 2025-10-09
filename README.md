# Galvatron-Ascend

## Overview

Galvatron-Ascend is the Ascend AI processor adaptation of Galvatron, an automatic distributed training system for Transformer models and Large Language Models (LLMs). It seamlessly integrates Galvatron with Huawei Ascend AI processors, enabling efficient automated distributed training solutions on the Ascend platform.

## Prerequisites

* Hardware: Ascend 910B series
* OS: Linux
* Software:
    * Python == 3.9.10
    * CANN == 8.0.RC1
    * PyTorch == 2.1.0 
    * torch-npu == 2.1.0.post3-20240413
    * MindSpeed (commit: 2b0edd2)

## Getting Started

Please refer to [Usage](#usage) for more details.

## Contributors

This project is jointly maintained and developed by **Peking University DAIR Lab** and **Huawei GTS (Global Technical Service)**.

We appreciate the collaborative efforts from both teams in making efficient distributed training accessible on the Ascend AI computing platform.


Below is Galvatron's original README.
---

# Galvatron-2

Galvatron is an automatic distributed training system designed for Transformer models, including Large Language Models (LLMs). It leverages advanced automatic parallelism techniques to deliver exceptional training efficiency. This repository houses the official implementation of Galvatron-2, our latest version enriched with several new features.

## Key Features
### (1) Enhanced Efficiency via Automatic Parallelism

#### Enlarged Parallelism Search Space
Incorporate multiple popular parallelism dimensions of distributed training, including DP (Data Parallelism), SDP (Sharded Data Parallelism, support both ZeRO-2 & ZeRO-3), PP (Pipeline Parallelism, support both GPipe & Pipedream-flush / 1F1B-flush), TP (Tensor Parallelism). Also incorporate CKPT (Activation Checkpointing) as a special parallelism dimension.

#### Fine-grained Hybrid Parallelism
For each Transformer layer, support flexible and fine-grained hybrid parallelism strategies, contributing to the enhanced training efficiency.

#### Efficient Automatic Parallelism Optimization
For any given Transformer model, automatically and efficiently search for the optimal parallelism strategies, which provides the optimal training efficiency.

### (2) Versatility
Suitable for a wide range of Transformer architectures, including language models, LLMs, vision models, multi-modality models, etc.

### (3) User-Friendly Interface
Easy to use, even for those new to distributed training.

## What's New in Galvatron-2
- Support CKPT (Activation Checkpointing)
- Support Mixed Precision (FP16, BF16)
- Support more pipeline schedules (GPipe and pipedream-flush / 1F1B-flush)
- Support PyTorch-2 (currently suppport 2.0.1)
- Support FlashAttention-2 for more efficient attention kernel
- Provide new Galvatron Profiler that profiles the model consumptions conveniently
- Provide new Galvatron Search Engine with enhanced efficiency of parallelism optimization
- Optimized user-friendly interfaces
- Support more Transformer models (more models are comming soon...)

## System Architecture
Galvatron is consisted of four modules, including an automatic Galvatron Profiler, a strategy cost estimator, Galvatron Search Engine that provides parallelism optimization, and Galvatron runtime framework. To train Transformer models over multiple GPUs using automatic parallelism with Galvatron, users only need to provide with hardware environment and the Transformer model configuration.

<div align=center> <img src="./figs/api.jpg" width="800" /> </div>

## Installation
Requirements:
- PyTorch 2.0.1 (we will support newer versions of pytorch soon)

To install Galvatron:

``` shell
pip install hetu-galvatron
```
Alternatively, you can install Galvatron from source with ```pip install .```

To use FlashAttention-2 features in Galvatron-2, you can either:
- Install the [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) manually and then ```pip install hetu-galvatron```.
- Alternatively, you can install Galvatron-2 with FlashAttention-2 as follows:

1. Make sure that PyTorch, `packaging` (`pip install packaging`), `ninja` is installed.
2. Install Galvatron-2 with FlashAttention-2:
```sh
GALVATRON_FLASH_ATTN_INSTALL=TRUE pip install hetu-galvatron
```


## Usage

### Profiling with Galvatron
The first step to use Galvatron is to profile the hardware environment and the model computation time. Galvatron will automatically save the profiled results into config files.

(1) Firstly, to profile the hardward environment, ```cd galvatron/profile_hardware```,  write the host address into ```hostfile```, set ```NUM_NODES, NUM_GPUS_PER_NODE, MPI_PATH``` in ```scripts/profile_hardware.sh``` and run:
``` shell
sh scripts/profile_hardware.sh
```

Galvatron will call [nccl-tests](https://github.com/NVIDIA/nccl-tests) to profile the communication bandwidth.

For Ascend platform, the script will not directly profile the bandwidth, but will generate four scripts, `profile_allreduce`, `profile_p2p`, `profile_allreduce_sp`, `profile_all2all_sp`. Users need to run these scripts on all nodes one by one to get the bandwidth of different communication modes.

(2) Secondly, to profile the model computation time and memory usage, ```cd galvatron/models/model_name``` and run:
``` shell
sh scripts/profile_computation.sh
sh scripts/profile_memory.sh
```

### Parallelism Optimizing with Galvatron
After profiling the environments, Galvatron is able to automatically optimize the parallelism strategy for the given Transformer model. Given the memory budget, Galvatron provides the fine-grained hybrid parallel strategy with maximum throughput. The optimized parallelism strategy will be saved in `galvatron/models/model_name/configs` for the training. Users can train the model with the provided optimal strategy to obtain the optimal throughput. 

To conduct parallelim optimization, ```cd galvatron/models/model_name```, customize ```NUM_NODES, NUM_GPUS_PER_NODE, MEMORY``` in ```scripts/search_dist.sh```, run:

``` shell
sh scripts/search_dist.sh
```

See more usage details of the customized parallelism optimization in [Galvatron Model Usage](galvatron/models/README.md#parallelism-optimizing-with-galvatron).

### Training with Galvatron
Galvatron provides a simple way to train Transformer models in fined-grained hybrid parallelism fashion. Users can either train Transformer models with the searched optimal parallel strategy by specifying argument ```galvatron_config_path``` to obtain the optimal throughput, or use any parallel strategies as they like. Galvatron support two hybrid parallel config modes, including JSON config mode and GLOBAL config mode. Users can specify parallel strategies by modifying only a few arguments. 

To train the model with Galvatron, ```cd galvatron/models/model_name```, set ```NUM_NODES, NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, NODE_RANK```,  and run:
``` shell
sh scripts/train_dist.sh
```

See detailed guidance and more customized training options in [Galvatron Model Usage](galvatron/models/README.md#training-with-galvatron).

