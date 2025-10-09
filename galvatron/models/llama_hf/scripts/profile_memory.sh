export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$(echo $VC_TASK_HOSTS | cut -d',' -f1) # $MASTER_ADDR
export MASTER_PORT=6000 # $MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$VC_TASK_INDEX # $RANK
export PYTHONPATH=$PYTHONPATH:/opt/dpcvol/models/lxy/Galvatron-ascend-2.4

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_test.py"

MODEL_ARGS="
    --model_size llama2-70b \
    --set_model_config_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --seq_length 2048"

# PROFILE_ARGS_BF16="
#     --profile_type memory \
#     --profile_batch_size 8 \
#     --layernum_min 1 \
#     --layernum_max 2 \
#     --max_tp_deg 8 \
#     --profile_dp_type zero3 \
#     --mixed_precision bf16 \
#     --sequence_parallel \
#     --use-flash-attn"

PROFILE_ARGS_BF16="
    --profile_mode sequence \
    --profile_type memory \
    --profile_batch_size 8 \
    --profile_min_seq_length 512 \
    --profile_max_seq_length 8192 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg 8 \
    --profile_dp_type zero3 \
    --mixed_precision bf16 \
    --use-flash-attn \
    --sequence_parallel \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu"

# PROFILE_ARGS_FP32="
#     --profile_type memory \
#     --profile_batch_size 8 \
#     --layernum_min 1 \
#     --layernum_max 2 \
#     --max_tp_deg 8 \
#     --profile_dp_type zero3 \
#     --mixed_precision fp32"

python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS_BF16} | tee logs/70B_profilem.log

