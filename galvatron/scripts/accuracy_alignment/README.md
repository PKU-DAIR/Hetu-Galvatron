# Accuracy Alignment (Megatron Data)

Use `record_loss.sh` to record step-by-step loss curves and upload them to WandB for baseline vs test comparison.

| Directory | Model | Training Entry |
|-----------|-------|----------------|
| `llama/` | Dense Llama | `models/gpt/train_dist.py` |
| `moe/` | MoE Mixtral | `models/moe/train_dist.py` |

Loss curve CSVs: `{variant}/{variant}_loss_baseline.csv`, `{variant}_loss_test.csv`

## Setup

```bash
pip install wandb
wandb login
```

Edit `paths` in the corresponding `train_dist.yaml`.

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

## Notes

- Default `--variant llama`; can also set via environment variable `ALIGN_VARIANT=moe`
- Default config: `{variant}/train_dist.yaml`; override with `--config`
- WandB: override with `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`
