export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export NODE_RANK=0
# export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_HCA=mlx5_2,mlx5_5
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist_random.py"

MODEL_ARGS="
    --model_size gpt-6.7b \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --vocab_size 50257 \
    --hidden_size 1600 \
    --num_hidden_layers 4 \
    --num_attention_heads 32 \
    --seq_length 2048"

TRAIN_ARGS="
    --global_train_batch_size 16 \
    --epochs 10 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 1 \
    --profile 1 \
    --save_profiled_memory 0"

CKPT_ARGS="
    --load /home/pkuhetu/lxy/checkpoints/Cerebras-GPT-6.7B-split"

PARALLEL_ARGS="
    --pp_deg 2 \
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --sdp 0 \
    --global_checkpoint 0 \
    --vocab_tp 1 \
    --chunks 1 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-flash-attn \
    --initialize_on_meta 1" #  \
    # --galvatron_config_path ./configs/galvatron_config_gpt-6.7b_2nodes_8gpus_per_node_40GB_bf16_example.json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${CKPT_ARGS}