
INPUT_PATH=/opt/dpcvol/models/llama-7b/llama-2-7b-hf
OUTPUT_PATH=/opt/dpcvol/models/lxy/llama2-7b-chat-hf-split

CHECKPOINT_ARGS="
    --input_checkpoint $INPUT_PATH \
    --output_dir $OUTPUT_PATH
"

python checkpoint_convert.py --model_type llama ${CHECKPOINT_ARGS}