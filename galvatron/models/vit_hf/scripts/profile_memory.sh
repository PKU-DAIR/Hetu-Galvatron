export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$RANK

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_random.py"

MODEL_ARGS_BASE="
    --model_size vit-base \
    --set_model_config_manually 0 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --num_hidden_layers 12 \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 3 \
    --num_labels 1000 \
    --hidden_dropout_prob 0.0 \
    --attention_probs_dropout_prob 0.0"

MODEL_ARGS_LARGE="
    --model_size vit-large \
    --set_model_config_manually 0 \
    --hidden_size 1024 \
    --num_attention_heads 16 \
    --num_hidden_layers 24 \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 3 \
    --num_labels 1000 \
    --hidden_dropout_prob 0.0 \
    --attention_probs_dropout_prob 0.0"

MODEL_ARGS_HUGE="
    --model_size vit-huge \
    --set_model_config_manually 0 \
    --hidden_size 1280 \
    --num_attention_heads 16 \
    --num_hidden_layers 32 \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 3 \
    --num_labels 1000 \
    --hidden_dropout_prob 0.0 \
    --attention_probs_dropout_prob 0.0"

MODEL_ARGS_XHUGE="
    --model_size vit-xhuge \
    --set_model_config_manually 0 \
    --hidden_size 2560 \
    --num_attention_heads 32 \
    --num_hidden_layers 128 \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 3 \
    --num_labels 1000 \
    --hidden_dropout_prob 0.0 \
    --attention_probs_dropout_prob 0.0"
# vit model does not support set seqlen manually, please modify config file directly (image_size and patch_size)

PROFILE_ARGS_BF16="
    --profile_mode static \
    --profile_type memory \
    --profile_batch_size 8 \
    --profile_seq_length_list 197 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg 8 \
    --profile_dp_type zero3 \
    --mixed_precision bf16 "

python3 profiler.py ${MODEL_ARGS_BASE} ${PROFILE_ARGS_BF16}
# python3 profiler.py ${MODEL_ARGS_LARGE} ${PROFILE_ARGS_BF16}
# python3 profiler.py ${MODEL_ARGS_HUGE} ${PROFILE_ARGS_BF16}
# python3 profiler.py ${MODEL_ARGS_XHUGE} ${PROFILE_ARGS_BF16}