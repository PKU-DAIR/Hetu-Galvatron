export NUM_NODES=1
export NUM_GPUS_PER_NODE=4

MODEL_SIZE="vit-huge"
MEMORY=23

MODEL_ARGS="
    --model_size ${MODEL_SIZE} \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 0 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --image_size 224 \
    --patch_size 16 \
    --num_channels 3 \
    --num_labels 1000 \
    --hidden_dropout_prob 0.0 \
    --attention_probs_dropout_prob 0.0 "

BSZ_ARGS="
    --min_bsz 4 \
    --max_bsz 256 \
    --bsz_scale 4 \
    --settle_bsz -1 \
    --settle_chunk -1 \
    --recommend_min_bsz 0
"

SEARCH_SPACE_ARGS="
    --search_space full \
    --sp_space tp \
    --disable_dp 0 \
    --disable_tp 0 \
    --disable_pp 0 \
    --disable_sdp 0 \
    --disable_ckpt 0 \
    --disable_vtp 0 \
    --disable_tp_consec 1 \
    --max_tp_deg 4 \
    --max_pp_deg 4 \
    --fine_grained_mode 0 \
    --time_profile_mode static \
    --memory_profile_mode static \
    --no_async_grad_reduce
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
    --embed_sdp 0
"

BACKGROUND=0

if [ $BACKGROUND -eq 1 ]; then
    echo "Search in background..."
    OUTPUT_FILE="Search_${MODEL_SIZE}_${MEMORY}GB_${NUM_NODES}Nodes_${NUM_GPUS_PER_NODE}GPUs_per_node.log"
    nohup python3 search_dist.py ${SEARCH_ARGS} 1> ${OUTPUT_FILE} 2>&1 &
else
    echo "Search in foreground..."
    python3 search_dist.py ${SEARCH_ARGS}
fi