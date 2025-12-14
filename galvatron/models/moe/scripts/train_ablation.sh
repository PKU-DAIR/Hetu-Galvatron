# export CUDA_DEVICE_MAX_CONNECTIONS=1 # to enable CP computation/communication streams to overlap
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # to avoid max_reserved_memory and max_allocated_memory over-sized
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NVTE_BATCH_MHA_P2P_COMM=1 # to force TransformerEngine to use batched send/recv for CP
export NCCL_DEBUG=WARN

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}
export NUM_NODES=4
export NUM_GPUS_PER_NODE=$ARNOLD_WORKER_GPU
export MASTER_ADDR=$METIS_WORKER_0_HOST
export MASTER_PORT=23456 # $port
export NODE_RANK=`expr $ARNOLD_ID - 0`

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

# export NCCL_NVLS_ENABLE=1
export GLOO_SOCKET_IFNAME=eth0
# export CUDA_DEVICE_MAX_CONNECTIONS=1

export MODEL_SIZE=$1
export LAYER_NUM=$2
export AUX=$3
export DATA=$4
export CAPACITY=$5
export BATCH_SIZE=$6
export CHUNK=$7
export SEQUENCE_LENGTH=$8
export GALVATRON_PROFILE=$9

echo "MODEL_SIZE: $MODEL_SIZE"
echo "LAYER_NUM: $LAYER_NUM"
echo "AUX: $AUX"
echo "METHOD: $ABLATION_APPROACH"
echo "DATA: $DATA"
echo "CAPACITY: $CAPACITY"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "CHUNK: $CHUNK"
echo "SEQUENCE_LENGTH: $SEQUENCE_LENGTH"
echo "GALVATRON_PROFILE: $GALVATRON_PROFILE"

if [ "$ABLATION_APPROACH" == "FSDP" ]; then
    export SOLVER="NONE"
else
    export SOLVER="LAER"
fi

export CHECKPOINT=1
export ENABLE_SOLVER=1

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

if [ "$GALVATRON_PROFILE" = "1" ]; then
    TRAINER="train_dist_with_profile.py"
else
    TRAINER="train_dist.py"
fi

CHECKPOINT_PATH=$BASE_DIR/checkpoints/laer/$MODEL_SIZE
TOKENIZER_MODEL=$BASE_DIR/tokenizers/mixtral
DATA_PATH=$BASE_DIR/datasets/processed/$DATA/mixtral_text_document

MODEL_ARGS="
    --model_size $MODEL_SIZE \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --set_seqlen_manually 1 \
    --set_experts_manually 0 \
    --set_aux_loss_manually \
    --router_aux_loss_coef $AUX \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers $LAYER_NUM \
    --num_attention_heads 32 \
    --seq_length $SEQUENCE_LENGTH"

TRAIN_ARGS="
    --global_train_batch_size $BATCH_SIZE \
    --train-iters 70 \
    --eval-iters 1 \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.1 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1.0e-5 \
    --init-method-std 0.01 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 1 \
    --no_async_grad_reduce \
    --save_profiled_memory 0 \
    --use_fsep \
    --expert_capacity_per_device $CAPACITY \
    --clip-grad 0.0
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL}
"

CKPT_ARGS="
    --load $CHECKPOINT_PATH
"

PARALLEL_ARGS="
    --pp_deg 1 \
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --global_ep_deg 32 \
    --global_tp_of_ep_deg 1 \
    --sdp 1 \
    --global_checkpoint $CHECKPOINT \
    --vocab_tp 1 \
    --chunks $CHUNK \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-flash-attn \
    --initialize_on_meta 1"

if [ "$GALVATRON_PROFILE" = "1" ]; then
    ${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS} ${CKPT_ARGS}
else
    ${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS} ${CKPT_ARGS} | tee training_log/ablation/${ABLATION_APPROACH}_${MODEL_SIZE}_${DATA}_batch${BATCH_SIZE}_seq${SEQUENCE_LENGTH}_aux${AUX}_${ARNOLD_ID}.log
fi