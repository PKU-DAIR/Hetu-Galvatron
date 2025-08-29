# LAER-MoE
This repo contains the official implementation of paper "LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training".

## Installation

```
conda create -n laer-moe python=3.9.2
conda activate laer-moe
pip install -r requirements.txt
pip install .
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