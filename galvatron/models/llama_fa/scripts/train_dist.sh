export NUM_NODES=2
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$RANK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"
DATA_PATH=/home/pkuhetu/lxy/dataset/llama/my-llama2_text_document
VOCAB_FILE=/home/pkuhetu/lxy/checkpoints/llama2-7b-chat-hf/tokenizer.json
TOKENIZER_MODEL=/home/pkuhetu/lxy/checkpoints/llama2-7b-chat-hf/tokenizer.model


MODEL_ARGS="
    --model_size llama-7b \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 8 \
    --num_attention_heads 32 \
    --seq_length 2048"

TRAIN_ARGS="
    --global_train_batch_size 8 \
    --train-iters 25 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL}
"

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
    --use-flash-attn \
    --shape_order BSH \
    --initialize_on_meta 1" #  \
   #  --galvatron_config_path ./configs/galvatron_config_hidden4096_head32_seqlen2048_2nodes_8gpus_per_node_34GB_bf16_bsz48_[tpconsec_off].json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS}
