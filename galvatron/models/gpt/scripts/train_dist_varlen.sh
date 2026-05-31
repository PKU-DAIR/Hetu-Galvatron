#!/bin/bash
set -x
set -o pipefail

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=8}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

if [[ "$NNODES" == "1" ]]; then
  additional_args="$additional_args --standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

log_dir="logs/runtime"
mkdir -p $log_dir

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args train_dist_varlen.py scripts/train_dist_varlen.yaml "$@" 2>&1 | tee $log_dir/train_varlen.log
