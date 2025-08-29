# export CUDA_DEVICE_MAX_CONNECTIONS=1 # to enable CP computation/communication streams to overlap
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # to avoid max_reserved_memory and max_allocated_memory over-sized
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NVTE_BATCH_MHA_P2P_COMM=1 # to force TransformerEngine to use batched send/recv for CP
export NCCL_DEBUG=WARN
export ENABLE_SOLVER=1

export MODEL_SIZE=$1 # model size, e.g. mixtral-8x7b-e8k2, mixtral-8x7b-e16k4, etc.
export LAYER_NUM=$2 # number of layers
export AUX=$3 # auxiliary loss coefficient
export SOLVER=$4 # solver type (LAER or NONE)
export CAPACITY=$5 # capacity of experts per device
export DATA_PATH=$6 # data path
export TOKENIZER_MODEL=$7 # tokenizer model path
# export CHECKPOINT_PATH=$8 # checkpoint path

export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=23456
export NODE_RANK=0

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 # modify this to your own HCA
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

export BATCH_SIZE=128
export CHECKPOINT=1
export CHUNK=2
export SEQUENCE_LENGTH=8192

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"

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
    --train-iters 60 \
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

# CKPT_ARGS="
#     --load $CHECKPOINT_PATH
# "

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
    # --galvatron_config_path ./configs/galvatron_config_hidden4096_head32_1nodes_8gpus_per_node_36GB_bf16_[tpconsec_off].json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS} # ${CKPT_ARGS}
# ${CKPT_ARGS}