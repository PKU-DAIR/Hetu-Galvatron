if [ $# -ne 3 ]; then
    echo "Usage: $0 <approach> <aux_loss> <iter>"
    exit 1
fi

source ./scripts_ae/env.sh

approach=$1
aux_loss=$2
iter=$3

model_name=mixtral-8x7b-e8k2
dataset=wikitext
batch_size=128
seq_length=4096
save_or_load=2

export BASE_DIR=$(cd $(dirname $0)/../..; pwd)

if [ "$approach" == "megatron" ]; then
    layer_num=32
    hidden_size=4096
    num_attention_heads=32
    ffn_hidden_size=14336
    expert_num=8
    topk=2
    expert_parallel=8
    tensor_parallel=4

    if [ -d "$BASE_DIR/checkpoints/megatron_convergence/$model_name" ]; then
        echo "Training $model_name with $approach..."
        cd $BASE_DIR/Megatron
        mkdir -p $BASE_DIR/Megatron/training_log
        bash scripts/train_convergence.sh $model_name $dataset $layer_num $hidden_size $num_attention_heads $ffn_hidden_size $expert_num $topk $aux_loss $batch_size $seq_length $expert_parallel $tensor_parallel $save_or_load $iter
        echo "Training finished for $model_name"
    else
        echo "checkpoints/megatron_convergence/$model_name does not exist"
        exit 1
    fi
else
    chunk=1
    layer_num=32
    capacity=2    
    if [ -d "$BASE_DIR/checkpoints/laer_convergence/$model_name" ]; then
        echo "Training $model_name with $approach..."
        cd $BASE_DIR/LAER-MoE/galvatron/models/moe
        mkdir -p $BASE_DIR/LAER-MoE/galvatron/models/moe/training_log
        bash scripts/train_convergence.sh $model_name $layer_num $aux_loss $approach $dataset $capacity $batch_size $chunk $seq_length 0 $iter
        echo "Training finished for $model_name"
    else
        echo "checkpoints/laer_convergence/$model_name does not exist"
        exit 1
    fi
fi