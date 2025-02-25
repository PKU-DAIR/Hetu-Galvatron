
INPUT_PATH=/path/to/huggingface/bert/checkpoint/
OUTPUT_PATH=/path/to/galvatron/bert/checkpoint/

CHECKPOINT_ARGS="
    --input_checkpoint $INPUT_PATH \
    --output_dir $OUTPUT_PATH
"

python checkpoint_convert_h2g.py --model_type bert-mlm ${CHECKPOINT_ARGS}