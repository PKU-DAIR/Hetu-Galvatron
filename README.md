# LAER-MoE Artifact

Note: Please complete the steps in `../README.md` first. We assume everything has been prepared for conducting experiments.

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

#### Convergence study (Figure 9)

`convergence.sh` will start the convergence study in §5.2. In the following command, the `approach` is the training approach to use. `aux_loss` is the auxiliary loss coefficient. (`approach`, `aux_loss`) can be one of (`LAER, 1e-4`), (`Megatron, 1e-2`), (`Megatron, 1e-4`). We will evaluate on `wikitext` dataset. Each evaluation will be corresponding to a curve in Figure 9. For default setting, `iter` is 3000. 
> NOTE: This command will take a long time to run (16h for `Megatron, 1e-4`, 10h for `Megatron, 1e-2`, 9h for `LAER, 1e-4`), for artifact evaluation, users can set `iter` to 1500 to save time. But users still need to set `iter` to 3000 to get the best performance.
```
./scripts_ae/convergence.sh <approach> <aux_loss> <iter>
```

### Case study

#### Breakdown (Figure 10(a))

`breakdown.sh` will start the case study in §5.3. In the following command, the `approach` is the training approach to use, which can be one of `LAER, FLEX, FSDP`. `model_name` is the name of the model, which can be one of `mixtral-8x7b-e8k2`, `mixtral-8x7b-e16k4`. Each evaluation will be corresponding to a bar in Figure 10(a). For default setting, `aux_loss` is `1e-4`, and `dataset` is `wikitext`.

```
./scripts_ae/breakdown.sh <approach> <model_name>
```

#### Maximum token counts (Figure 10(b))

`token_counts.sh` will start the case study in §5.3. In the following command, the `approach` is the training approach to use. `model_name` is the name of the model, which can be one of `mixtral-8x7b-e8k2`, `mixtral-8x7b-e16k4`. Each evaluation will be corresponding to a curve in Figure 10(b). For default setting, `aux_loss` is `1e-4`, and `dataset` is `wikitext`.

```
./scripts_ae/token_counts.sh <approach> <model_name>
```

### Planner Performance (Figure 11)

`planner.sh` will start the planner performance in §5.4. In the following command, the `N` is the number of devices, and `C` is the capacity of experts per device. Each evaluation will be corresponding to a point in Figure 11.

```
./scripts_ae/planner.sh <N> <C>
```

### Ablation study (Figure 12)

`ablation.sh` will start the ablation study in §5.5. In the following command, the `approach` is the training approach to use, which can be one of `LAER, no_even, no_pq, no_comm_opt`. Each evaluation will be corresponding to a bar in Figure 12. For default setting, `model_name` is `mixtral-8x7b-e8k2`, `aux_loss` is `1e-4`, and `dataset` is `wikitext`.

```
./scripts_ae/ablation.sh <approach>
```

## Plotting Figures

To analysis these data and plot figures presented in the paper, we also provide `plot_{id}.py` to plot the corresponding figure. For example, to plot Figure 8/9, we use the following command:

```
bash scripts_ae/8_plot.sh new
bash scripts_ae/9_plot.sh new
```

To get Figure 8/9 in the paper, we use the following command:

```
bash scripts_ae/8_plot.sh default
bash scripts_ae/9_plot.sh default
```

To plot Figure 10(a), we use the following command:

```
bash scripts_ae/10_plot.sh
```