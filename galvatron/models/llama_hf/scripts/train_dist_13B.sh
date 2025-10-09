source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=3600
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export NUM_NODES=$(echo $VC_TASK_NUM | awk '{print int($0)}')
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$(echo $VC_TASK_HOSTS | cut -d',' -f1) # $MASTER_ADDR
export MASTER_PORT=6000 # $MASTER_PORT
export NODE_RANK=$VC_TASK_INDEX # $RANK
export PYTHONPATH=$PYTHONPATH:/opt/dpcvol/models/Galvatron-ascend-2.1

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"

MODEL_ARGS="
    --model_size llama-13b \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 2 \
    --num_attention_heads 32 \
    --seq_length 2048"

TRAIN_ARGS="
    --global_train_batch_size 64 \
    --epochs 10 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0"

PARALLEL_ARGS="
    --pp_deg 1 \
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --sdp 0 \
    --global_checkpoint 0 \
    --vocab_tp 1 \
    --chunks 8 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --use-fused-rmsnorm \
    --normalization RMSNorm \
    --sequence-parallel \
    --use-flash-attn \
    --initialize_on_meta 1
    --galvatron_config_path ./configs/galvatron_config_hidden5120_head40_seqlen4096_16nodes_8gpus_per_node_32GB_bf16_bsz1024_[tpconsec_off].json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} > new_logs/13B_result_${VC_TASK_INDEX}.log 2> new_logs/13B_error_${VC_TASK_INDEX}.log
