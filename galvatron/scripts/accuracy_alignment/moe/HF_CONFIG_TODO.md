# MoE + HuggingFace（预留）

待补充：

- `train_dist_hf.yaml` — HF 数据链路的 MoE 训练配置
- 曲线输出：`moe_loss_baseline_hf.csv` / `moe_loss_test_hf.csv`

就绪后运行示例：

```bash
bash scripts/accuracy_alignment/record_loss.sh \
  --variant moe \
  --mode test \
  --config scripts/accuracy_alignment/moe/train_dist_hf.yaml
```
