export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$RANK
# export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export NCCL_TIMEOUT=1800

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"
DATA_PATH="/home/galvatron_dev/lqs/Hetu-Galvatron/galvatron/data/imagenet"  

MODEL_ARGS="
    --model_size vit-base \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 3 \
    --num_labels 1000 \
    --mlp_ratio 4.0 \
    --qkv_bias True \
    --hidden_dropout_prob 0.0 \
    --attention_probs_dropout_prob 0.0 \
    --drop_path_rate 0.1"

TRAIN_ARGS="
    --global_train_batch_size 16 \
    --train-iters 25 \
    --eval_interval 5 \
    --lr 1e-4 \
    --adam_weight_decay 0.05 \
    --check_loss 1 \
    --profile 1 \
    --save_profiled_memory 0"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 900,100,0 \
    --vision-pretraining \
    --image-size 224"

CKPT_ARGS="
    --load /path/to/vit/checkpoints" 

PARALLEL_ARGS="
    --pp_deg 2 \
    --global_tp_deg 2 \
    --global_tp_consec 1 \
    --sdp 1 \
    --global_checkpoint 0 \
    --chunks 2 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-flash-attn \
    --initialize_on_meta 1"
    #--galvatron_config_path"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS} #${CKPT_ARGS}