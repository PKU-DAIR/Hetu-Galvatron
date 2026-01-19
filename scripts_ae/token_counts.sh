if [ $# -ne 2 ]; then
    echo "Usage: $0 <approach> <model_name>"
    exit 1
fi

source ./scripts_ae/env.sh

approach=$1
model_name=$2
aux_loss=0
dataset=wikitext

batch_size=128
seq_length=8192
save_or_load=2
export TOKEN_COUNTS_ITER=40

export BASE_DIR=$(cd $(dirname $0)/../..; pwd)

chunk=2
if [ "$model_name" == "mixtral-8x7b-e8k2" ] || [ "$model_name" == "qwen-8x7b-e8k2" ]; then
    layer_num=32
    capacity=2    
fi
if [ "$model_name" == "mixtral-8x7b-e16k4" ] || [ "$model_name" == "qwen-8x7b-e16k4" ]; then
    layer_num=24
    capacity=4
fi
if [ "$model_name" == "mixtral-8x22b-e8k2" ]; then
    layer_num=18
    capacity=2
fi
if [ "$model_name" == "mixtral-8x22b-e16k4" ]; then
    layer_num=14
    capacity=4
fi

if [ -d "$BASE_DIR/checkpoints/laer/$model_name" ]; then
    echo "Generating token counts for $model_name with $approach..."
    cd $BASE_DIR/LAER-MoE/galvatron/models/moe
    mkdir -p $BASE_DIR/LAER-MoE/galvatron/models/moe/training_log/token_counts
    bash scripts/train.sh $model_name $layer_num $aux_loss $approach $dataset $capacity $batch_size $chunk $seq_length 0
    echo "Token counts finished for $model_name"
else
    echo "checkpoints/laer/$model_name does not exist"
    exit 1
fi