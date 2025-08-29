INPUT_PATH={your_checkpoint_input_path}
OUTPUT_PATH={your_checkpoint_output_path}

CHECKPOINT_ARGS="
    --input_checkpoint $INPUT_PATH \
    --output_dir $OUTPUT_PATH
"

python checkpoint_convert_h2g.py --model_type llama ${CHECKPOINT_ARGS}