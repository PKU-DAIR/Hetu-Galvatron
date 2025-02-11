export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$RANK
# export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export NCCL_TIMEOUT=1800  # 30分钟

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"
DATA_PATH="/home/galvatron_dev/lqs/Hetu-Galvatron/galvatron/models/bert_hf/pretrain_data/bert-pretrain-data_text_document"
VOCAB_FILE="/home/galvatron_dev/lqs/Hetu-Galvatron/galvatron/models/bert_hf/bert_uncased_vocab.txt"

MODEL_ARGS="
    --model_size bert-base \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --vocab_size 30522 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --seq_length 512"

TRAIN_ARGS="
    --global_train_batch_size 16 \
    --train-iters 25 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 1 \
    --profile 1 \
    --save_profiled_memory 0"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 800,100,100 \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file $VOCAB_FILE 
"
#--split 949,50,1 \
# --train-data-path ${DATA_PATH}_text_document \
# --valid-data-path ${DATA_PATH}_text_document \
# --test-data-path ${DATA_PATH}_text_document \

#CKPT_ARGS="
#   --load /home/pkuhetu/lxy/checkpoints/bert-base-uncased"#

PARALLEL_ARGS="
    --pp_deg 2 \
    --global_tp_deg 2 \
    --global_tp_consec 1 \
    --sdp 1 \
    --global_checkpoint 0 \
    --vocab_tp 2 \
    --chunks 2 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-flash-attn \
    --initialize_on_meta 1 "
    #--galvatron_config_path ./configs/galvatron_config_hidden4096_head32_seqlen2048_2nodes_8gpus_per_node_34GB_bf16_bsz48_[tpconsec_off]"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS} #${CKPT_ARGS}