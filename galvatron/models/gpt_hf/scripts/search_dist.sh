export NUM_NODES=2
export NUM_GPUS_PER_NODE=8

MODEL_SIZE="gpt-6.7b"
MEMORY=40

MODEL_ARGS="
    --model_size ${MODEL_SIZE} \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 0 \
    --vocab_size 50257 \
    --hidden_size 1600 \
    --num_hidden_layers 12 \
    --num_attention_heads 32 \
    --seq_length 2048"

BSZ_ARGS="
    --min_bsz 16 \
    --max_bsz 1024 \
    --bsz_scale 16 \
    --settle_bsz 128 \
    --recommend_min_bsz 0 \
    --settle_chunk -1
"

SEARCH_SPACE_ARGS="
    --search_space full \
    --disable_dp 0 \
    --disable_tp 0 \
    --disable_pp 0 \
    --disable_sdp 0 \
    --disable_ckpt 0 \
    --disable_vtp 0 \
    --disable_tp_consec 1 \
    --max_tp_deg 8 \
    --max_pp_deg 8 \
    --fine_grained_mode 1 \
    --profile_mode sequence \
    --sequence_parallel
"

SEARCH_ARGS="
    ${BSZ_ARGS} \
    ${SEARCH_SPACE_ARGS} \
    ${MODEL_ARGS} \
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
    --memory_constraint $MEMORY \
    --mixed_precision bf16 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --sequence_parallel \
    --embed_sdp 0
"

BACKGROUND=1

if [ $BACKGROUND -eq 1 ]; then
    echo "Search in background..."
    OUTPUT_FILE="Search_${MODEL_SIZE}_${MEMORY}GB_${NUM_NODES}Nodes_${NUM_GPUS_PER_NODE}GPUs_per_node.log"
    nohup python3 search_dist.py ${SEARCH_ARGS} 1> ${OUTPUT_FILE} 2>&1 &
else
    echo "Search in foreground..."
    python3 search_dist.py ${SEARCH_ARGS}
fi