# 精度对齐（WandB）

配置按模型拆到子目录，统一用 `record_loss.sh` 录 loss 曲线。

| 目录 | 说明 | Megatron 数据 | HF 数据 |
|------|------|---------------|---------|
| `llama/` | Dense Llama 对齐 | `train_dist.yaml` | `train_dist_hf.yaml` |
| `moe/` | MoE 对齐 | `train_dist.yaml` | *待补充* `train_dist_hf.yaml` |

曲线 CSV 写在对应子目录，文件名带前缀：

- Llama Megatron：`llama/llama_loss_baseline.csv`、`llama/llama_loss_test.csv`
- Llama HF：`llama/llama_loss_baseline_hf.csv`、`llama/llama_loss_test_hf.csv`
- MoE Megatron：`moe/moe_loss_baseline.csv`、`moe/moe_loss_test.csv`
- MoE HF：预留 `moe/moe_loss_baseline_hf.csv` 等（配置就绪后再跑）

Megatron 与 HF **各自** baseline vs test 对比，两边数据不必一致。

`record_loss.sh` 会按 `--variant` 进入对应训练目录：`llama` → `models/gpt/train_dist.py`，`moe` → `models/moe/train_dist.py`。

## 1) 一次性准备

```bash
pip install wandb
wandb login
```

## 2) Llama — Megatron 数据

```bash
cd /home/pkuhetu/xby/galvatron-moe-test/galvatron

# baseline
bash scripts/accuracy_alignment/record_loss.sh --variant llama --mode baseline

# test（改代码后）
bash scripts/accuracy_alignment/record_loss.sh --variant llama --mode test
```

## 3) Llama — HuggingFace 数据

```bash
bash scripts/accuracy_alignment/record_loss.sh \
  --variant llama \
  --mode test \
  --config scripts/accuracy_alignment/llama/train_dist_hf.yaml
```

## 4) MoE — Megatron 数据

```bash
bash scripts/accuracy_alignment/record_loss.sh --variant moe --mode baseline
bash scripts/accuracy_alignment/record_loss.sh --variant moe --mode test
```

MoE + HF：等 `moe/train_dist_hf.yaml` 补齐后再跑，例如：

```bash
# bash scripts/accuracy_alignment/record_loss.sh \
#   --variant moe \
#   --mode test \
#   --config scripts/accuracy_alignment/moe/train_dist_hf.yaml
```

## 5) 说明

- 默认 `--variant llama`；也可用环境变量 `ALIGN_VARIANT=moe`。
- 默认配置：`{variant}/train_dist.yaml`；`--config` 可覆盖。
- 训练日志在临时目录，只持久化 CSV；W&B run 名默认 `{variant}_{mode}{_hf?}_{RUN_TAG}`。

## 6) 常用环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `ALIGN_VARIANT` | `llama` | 与 `--variant` 相同 |
| `NPROC_PER_NODE` | `8` | GPU 数 |
| `WANDB_PROJECT` | `galvatron` | W&B 项目 |
| `WANDB_ENTITY` | 空 | W&B team/user |
| `RUN_TAG` | 时间戳 | 拼进 run 名 |
| `CURVE_STORE` | 见上表 | 手动指定 CSV 路径 |
