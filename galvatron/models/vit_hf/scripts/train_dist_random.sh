export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
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
    --model_size vit-huge \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --hidden_size 1280 \
    --num_hidden_layers 4 \
    --num_attention_heads 16 \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 3 \
    --num_labels 1000 \
    --hidden_dropout_prob 0.0 \
    --attention_probs_dropout_prob 0.0"

TRAIN_ARGS="
    --global_train_batch_size 8 \
    --epochs 10 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --check_loss 1 \
    --profile 1 \
    --save_profiled_memory 0"

#CKPT_ARGS="
    #--load /home/galvatron_dev/lqs/Hetu-Galvatron/galvatron/models/vit_hf/checkpoints/vit-base-split"

PARALLEL_ARGS="
    --pp_deg 2 \
    --global_tp_deg 2 \
    --global_tp_consec 1 \
    --sdp 0 \
    --global_checkpoint 0 \
    --chunks 1 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --initialize_on_meta 1 \
    --galvatron_config_path configs/galvatron_config_hidden1280_head16_1nodes_4gpus_per_node_23GB_bf16_[tpconsec_off].json"
${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} 
#${CKPT_ARGS}