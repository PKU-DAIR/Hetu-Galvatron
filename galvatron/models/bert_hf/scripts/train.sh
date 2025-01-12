LAUNCHER="python3"

TRAINER="train.py"

${LAUNCHER} ${TRAINER} \
--gpu_id 0 \
--global_train_batch_size 1 \
--model_size bert-base \
--set_model_config_manually 0 \
--set_layernum_manually 0 \
--vocab_size 30522 \
--hidden_size 768 \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--seq_length 512 \
--epochs 10 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--dropout_prob 0.1 \
--check_loss 0 \
--profile 1