if [ $# -ne 4 ]; then
    echo "Usage: $0 <approach> <model_name> <aux_loss> <dataset>"
    exit 1
fi

approach=$1
model_name=$2
aux_loss=$3
dataset=$4

batch_size=128
seq_length=8192
save_or_load=2

export BASE_DIR=$(cd $(dirname $0)/../..; pwd)

if [ "$approach" == "megatron" ]; then
    if [ "$model_name" == "mixtral-8x7b-e8k2" ] || [ "$model_name" == "qwen-8x7b-e8k2" ]; then
        layer_num=32
        hidden_size=4096
        num_attention_heads=32
        ffn_hidden_size=14336
        expert_num=8
        topk=2
        expert_parallel=8
        tensor_parallel=8
    fi
    if [ "$model_name" == "mixtral-8x7b-e16k4" ] || [ "$model_name" == "qwen-8x7b-e16k4" ]; then
        layer_num=24
        hidden_size=4096
        num_attention_heads=32
        ffn_hidden_size=7168
        expert_num=16
        topk=4
        expert_parallel=8
        tensor_parallel=4
    fi
    if [ "$model_name" == "mixtral-8x22b-e8k2" ]; then
        layer_num=18
        hidden_size=6144
        num_attention_heads=48
        ffn_hidden_size=16384
        expert_num=8
        topk=2
        expert_parallel=8
        tensor_parallel=8
    fi
    if [ "$model_name" == "mixtral-8x22b-e16k4" ]; then
        layer_num=14
        hidden_size=6144
        num_attention_heads=48
        ffn_hidden_size=8192
        expert_num=16
        topk=4
        expert_parallel=8
        tensor_parallel=4
    fi


    if [ -d "$BASE_DIR/checkpoints/megatron/$model_name" ]; then
        echo "Training $model_name with $approach..."
        cd $BASE_DIR/Megatron
        mkdir -p $BASE_DIR/Megatron/training_log
        bash scripts/train.sh $model_name $dataset $layer_num $hidden_size $num_attention_heads $ffn_hidden_size $expert_num $topk $aux_loss $batch_size $seq_length $expert_parallel $tensor_parallel $save_or_load
        echo "Training finished for $model_name"
    else
        echo "checkpoints/megatron/$model_name does not exist"
        exit 1
    fi
else
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
        echo "Training $model_name with $approach..."
        cd $BASE_DIR/LAER-MoE/galvatron/models/moe
        mkdir -p $BASE_DIR/LAER-MoE/galvatron/models/moe/training_log
        bash scripts/train.sh $model_name $layer_num $aux_loss $approach $dataset $capacity $batch_size $chunk $seq_length 0
        echo "Training finished for $model_name"
    else
        echo "checkpoints/laer/$model_name does not exist"
        exit 1
    fi
fi