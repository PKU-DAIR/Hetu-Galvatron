
INPUT_PATH=/path/to/galvatron/bert/checkpoint/
OUTPUT_PATH=/path/to/huggingface/bert/checkpoint/

CHECKPOINT_ARGS="
    --input_checkpoint $INPUT_PATH \
    --output_dir $OUTPUT_PATH \
    --model_config bert-base \        
    --load_iteration 10                
"

python checkpoint_convert_g2h.py --model_type bert-mlm ${CHECKPOINT_ARGS}