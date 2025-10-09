source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd ${MA_JOB_DIR}/runtime/MindSpeed-1.1
pip install -e .
cd ${MA_JOB_DIR}/runtime/Galvatron-ascend-2.4/galvatron/models/llama_hf/

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=3600
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export MASTER_ADDR=$(echo $VC_TASK_HOSTS | cut -d',' -f1) # $MASTER_ADDR
export MASTER_PORT=6000 # $MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$VC_TASK_INDEX # $RANK
export PYTHONPATH=$PYTHONPATH:/${MA_JOB_DIR}/runtime/Galvatron-ascend-2.4

NPUS_PER_NODE=8
MASTER_HOST=${MA_VJ_NAME}-${MA_TASK_NAME}-0.${MA_VJ_NAME}:1234
MASTER_ADDR=${MASTER_HOST%%:*}
MASTER_PORT=${MASTER_HOST##*:}
NNODES=$MA_NUM_HOSTS
NODE_RANK=$VC_TASK_INDEX
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_test.py"

MODEL_ARGS="
    --model_size llama2-70b \
    --set_model_config_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --seq_length 2048"

# PROFILE_ARGS="
#     --profile_mode batch \
#     --profile_type computation \
#     --profile_min_batch_size 1 \
#     --profile_max_batch_size 12 \
#     --profile_batch_size_step 1 \
#     --layernum_min 2 \
#     --layernum_max 4 \
#     --mixed_precision bf16 \
#     --use-flash-attn"

PROFILE_ARGS="
    --profile_mode sequence \
    --profile_type computation \
    --profile_batch_size 1 \
    --profile_min_seq_length 4096 \
    --profile_max_seq_length 32768 \
    --profile_seq_length_step 4096 \
    --layernum_min 1 \
    --layernum_max 2 \
    --mixed_precision bf16 \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu"

# models in flash_attn cannot use fp32 without flash_attn
# PROFILE_ARGS="
#     --profile_mode static \
#     --profile_type computation \
#     --profile_batch_size 4 \
#     --layernum_min 12 \
#     --layernum_max 24 \
#     --mixed_precision fp32"

python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS} | tee logs/70B_profilec.log