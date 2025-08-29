# LAER-MoE
This repo contains the official implementation of paper "LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training".

## Installation

```
conda create -n laer-moe python=3.9.2
conda activate laer-moe
pip install -r requirements.txt
pip install flash_attn==2.5.8
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
GALVATRON_FLASH_ATTN_INSTALL=TRUE pip install .
```

## Examples

Use the following command to train Mixtral-8x7B on 32 GPUs:

```
cd galvatron/models/moe
bash scripts/train_dist_fsep_32gpus.sh mixtral-8x7b-e8k2 32 0 LAER 2 {your_data_path} {your_tokenizer_path}
```

Where:
- {your_data_path} is the path to the training data
- {your_tokenizer_path} is the path to the tokenizer

and the format of {your_data_path} and {your_tokenizer_path} is same with Megatron-LM.