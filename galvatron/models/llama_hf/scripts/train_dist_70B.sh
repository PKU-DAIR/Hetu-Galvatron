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

NPUS_PER_NODE=8
MASTER_HOST=${MA_VJ_NAME}-${MA_TASK_NAME}-0.${MA_VJ_NAME}:1234
MASTER_ADDR=${MASTER_HOST%%:*}
MASTER_PORT=${MASTER_HOST##*:}
NNODES=$MA_NUM_HOSTS
NODE_RANK=$VC_TASK_INDEX
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


export PYTHONPATH=$PYTHONPATH:/${MA_JOB_DIR}/runtime/Galvatron-ascend-2.4

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NNODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"
DATA_PATH="/home/ma-user/modelarts/inputs/data_path_0/alpaca_text_document"
TOKENIZER_MODEL="/home/ma-user/modelarts/inputs/data_path_0/tokenizer.model"

MODEL_ARGS="
    --model_size llama2-70b \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 1 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 20 \
    --num_attention_heads 32 \
    --seq_length 98304"

TRAIN_ARGS="
    --global_train_batch_size 32 \
    --train-iters 20 \
    --eval-iters 1 \
    --lr 1e-6 \
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
    --pp_deg 1 \
    --global_tp_deg 16 \
    --global_tp_consec 1 \
    --sdp 1 \
    --global_checkpoint 1 \
    --vocab_tp 16 \
    --chunks 2 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --use-fused-rmsnorm \
    --normalization RMSNorm \
    --use-fused-rotary-pos-emb \
    --position-embedding-type rope \
    --use-fused-swiglu \
    --swiglu \
    --sequence-parallel \
    --use-flash-attn \
    --no-create-attention-mask-in-dataloader \
    --initialize_on_meta 1" 
    # --galvatron_config_path ./configs/galvatron_config_hidden8192_head64_seqlen4096_16nodes_8gpus_per_node_28GB_bf16_bsz128_[tpconsec_off]_80.json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS}