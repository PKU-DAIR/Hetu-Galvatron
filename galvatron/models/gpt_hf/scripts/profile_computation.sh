export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$RANK

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_test.py"

MODEL_ARGS_SIZE15B="
    --model_size gpt-1.5b \
    --set_model_config_manually 0 \
    --vocab_size 50257 \
    --hidden_size 1600 \
    --num_attention_heads 32 \
    --seq_length 1024"

MODEL_ARGS_SIZE27B="
    --model_size gpt-2.7b \
    --set_model_config_manually 0 \
    --vocab_size 50257 \
    --hidden_size 2560 \
    --num_attention_heads 32 \
    --seq_length 2048"

MODEL_ARGS_SIZE67B="
    --model_size gpt-6.7b \
    --set_model_config_manually 0 \
    --vocab_size 50257 \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --seq_length 2048"

PROFILE_ARGS="
    --profile_mode sequence \
    --profile_type computation \
    --profile_batch_size 1 \
    --profile_min_seq_length 4096 \
    --profile_max_seq_length 65536 \
    --profile_seq_length_step 4096 \
    --layernum_min 1 \
    --layernum_max 2 \
    --mixed_precision bf16 \
    --use-flash-attn"

# PROFILE_ARGS="
#     --profile_type computation \
#     --profile_min_batch_size 1 \
#     --profile_max_batch_size 16 \
#     --profile_batch_size_step 1 \
#     --layernum_min 2 \
#     --layernum_max 4 \
#     --mixed_precision bf16 \
#     --use-flash-attn"

# python3 profiler.py ${MODEL_ARGS_SIZE15B} ${PROFILE_ARGS}
# python3 profiler.py ${MODEL_ARGS_SIZE27B} ${PROFILE_ARGS}
python3 profiler.py ${MODEL_ARGS_SIZE67B} ${PROFILE_ARGS}
