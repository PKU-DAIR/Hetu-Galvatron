# 非 HF 精度记录（WandB）

统一使用一个脚本：`record_loss.sh`。

通过 `--mode` 选择写入基线或测试曲线：

- 默认不传参数时是 `test` 模式；
- `--mode baseline`：写入 `scripts/accuracy_alignment/loss_baseline.csv`
- `--mode test`：写入 `scripts/accuracy_alignment/loss_test.csv`
- 当配置文件名包含 `hf`（如 `train_dist_hf.yaml`）时，默认输出自动带 `_hf` 后缀：
  - `loss_baseline_hf.csv` / `loss_test_hf.csv`

默认训练配置：`scripts/accuracy_alignment/train_dist.yaml`（megatron 数据链路）。
可选 HF 配置：`scripts/accuracy_alignment/train_dist_hf.yaml`。

## 1) 一次性准备

```bash
pip install wandb
wandb login
```

## 2) 记录 baseline 曲线

```bash
cd /home/pkuhetu/xby/galvatron-moe-test/galvatron
bash scripts/accuracy_alignment/record_loss.sh --mode baseline
```

## 3) 记录 test 曲线

```bash
cd /home/pkuhetu/xby/galvatron-moe-test/galvatron
bash scripts/accuracy_alignment/record_loss.sh --mode test
```

## 4) 使用 HuggingFace 数据配置

```bash
cd /home/pkuhetu/xby/galvatron-moe-test/galvatron
bash scripts/accuracy_alignment/record_loss.sh \
  --mode test \
  --config scripts/accuracy_alignment/train_dist_hf.yaml
```

## 5) 说明

- 不再使用 `galvatron/logs/accuracy_alignment` 作为持久目录。
- 脚本运行时会在临时目录产生日志并自动清理。
- 只持久化曲线 CSV 到 `accuracy_alignment` 目录。
- wandb 指标名按模式写入：`baseline_loss` 或 `test_loss`。

## 6) 常用环境变量

- `WANDB_PROJECT`（可选）：默认 `my-awesome-project`
- `WANDB_ENTITY`（可选）：wandb team/user
- `WANDB_RUN_NAME`（可选）：默认 `{mode}{_hf?}_{RUN_TAG}`，其中 `RUN_TAG` 默认为时间戳 `YYYYMMDD_HHMMSS`（`MODE`/`hf` 后缀只出现一次）
- `NNODES`/`NPROC_PER_NODE`/`NODE_RANK`/`MASTER_ADDR`/`MASTER_PORT`
- `RUN_TAG`：覆盖默认时间戳，拼进 `WANDB_RUN_NAME` 末尾
- `ALIGN_CONFIG`：配置路径（默认 `scripts/accuracy_alignment/train_dist.yaml`，可被 `--config` 覆盖）
- `CURVE_STORE`：手动指定输出 CSV 路径（可选）
