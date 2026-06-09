#!/bin/bash
# Reproducibility env for accuracy_alignment scripts (sourced by record_loss.sh).
# Disable without editing: GALVATRON_DETERMINISTIC=0 bash record_loss.sh ...

export GALVATRON_DETERMINISTIC="${GALVATRON_DETERMINISTIC:-1}"

# Do not import Transformer Engine (use FlashAttention / Apex / torch fallbacks).
export GALVATRON_USE_TE="${GALVATRON_USE_TE:-0}"

# CuBLAS: required for deterministic CUDA matmul when using deterministic algorithms.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

# Reduce TF32 / fast-math variability between runs.
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-0}"
export TORCH_ALLOW_TF32="${TORCH_ALLOW_TF32:-0}"
