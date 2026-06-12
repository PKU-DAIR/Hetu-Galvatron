# 精度对齐（Megatron 数据）

用 `record_loss.sh` 录制逐步 loss 曲线并上传 WandB，对比 baseline vs test。

| 目录 | 模型 | 训练入口 |
|------|------|----------|
| `llama/` | Dense Llama | `models/gpt/train_dist.py` |
| `moe/` | MoE Mixtral | `models/moe/train_dist.py` |

曲线 CSV：`{variant}/{variant}_loss_baseline.csv`、`{variant}_loss_test.csv`

## 准备

```bash
pip install wandb
wandb login
```

编辑对应 `train_dist.yaml` 中的 `paths`。

## Llama

```bash
cd galvatron
export PYTHONPATH=<repo_root>:$PYTHONPATH

bash scripts/accuracy_alignment/record_loss.sh --variant llama --mode baseline
bash scripts/accuracy_alignment/record_loss.sh --variant llama --mode test
```

## MoE

```bash
bash scripts/accuracy_alignment/record_loss.sh --variant moe --mode baseline
bash scripts/accuracy_alignment/record_loss.sh --variant moe --mode test
```

## 说明

- 默认 `--variant llama`；可用环境变量 `ALIGN_VARIANT=moe`
- 默认配置：`{variant}/train_dist.yaml`；`--config` 可覆盖
- WandB：`WANDB_PROJECT`、`WANDB_ENTITY`、`WANDB_RUN_NAME` 可覆盖
