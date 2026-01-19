if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <type>"
    exit 1
fi

source ./scripts_ae/env.sh

model_name=$1
type=$2

export BASE_DIR=$(cd $(dirname $0)/../..; pwd)

if [ "$type" == "convergence" ]; then
    dataset=wikitext
    aux_loss=0
    batch_size=256
    seq_length=4096
    save_or_load=1
    if [ "$model_name" == "mixtral-8x7b-e8k2" ]; then
        layer_num=32
        hidden_size=4096
        num_attention_heads=32
        ffn_hidden_size=14336
        expert_num=8
        topk=2
        expert_parallel=8
        tensor_parallel=4
        mkdir -p $BASE_DIR/checkpoints/megatron_convergence

        if [ -d "$BASE_DIR/checkpoints/megatron_convergence/$model_name" ]; then
            echo "checkpoints/megatron_convergence/$model_name already exists"
        else
            echo "Saving checkpoint for $model_name..."
            cd $BASE_DIR/Megatron
            bash scripts/train_convergence.sh $model_name $dataset $layer_num $hidden_size $num_attention_heads $ffn_hidden_size $expert_num $topk $aux_loss $batch_size $seq_length $expert_parallel $tensor_parallel $save_or_load 70
            echo "Checkpoint saved for $model_name in convergence mode."
            if [ `expr $ARNOLD_ID - 0` -eq 0 ]; then
                mv $BASE_DIR/checkpoints/megatron_convergence/$model_name/iter_0000000 $BASE_DIR/checkpoints/megatron_convergence/$model_name/release
                echo "release" > $BASE_DIR/checkpoints/megatron_convergence/$model_name/latest_checkpointed_iteration.txt
                echo "Convert to release version for $model_name"
            fi
        fi

        if [ `expr $ARNOLD_ID - 0` -eq 0 ]; then
            echo "Converting checkpoint to LAER-MoE format..."
            cd $BASE_DIR/LAER-MoE
            mkdir -p $BASE_DIR/checkpoints/laer_convergence
            python galvatron/tools/convert_to_laer.py \
                --input_dir $BASE_DIR/checkpoints/megatron_convergence/$model_name/release \
                --output_dir $BASE_DIR/checkpoints/laer_convergence/$model_name \
                --tp_size $tensor_parallel \
                    --ep_size $expert_parallel \
                    --num_layers $layer_num \
                    --num_attention_heads $num_attention_heads \
                    --hidden_size $hidden_size \
                    --num_experts $expert_num
            echo "Checkpoint converted to LAER-MoE format for $model_name in convergence mode."
        fi
    else
        echo "Convergence mode only supports mixtral-8x7b-e8k2."
        exit 1
    fi
    exit 0
fi

dataset=wikitext
aux_loss=0
batch_size=128
seq_length=8192
save_or_load=1

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

mkdir -p $BASE_DIR/checkpoints/megatron
if [ -d "$BASE_DIR/checkpoints/megatron/$model_name" ]; then
    echo "checkpoints/megatron/$model_name already exists"
else
    echo "Saving checkpoint for $model_name..."
    cd $BASE_DIR/Megatron
    bash scripts/train.sh $model_name $dataset $layer_num $hidden_size $num_attention_heads $ffn_hidden_size $expert_num $topk $aux_loss $batch_size $seq_length $expert_parallel $tensor_parallel $save_or_load
    echo "Checkpoint saved for $model_name"
    if [ `expr $ARNOLD_ID - 0` -eq 0 ]; then
        mv $BASE_DIR/checkpoints/megatron/$model_name/iter_0000000 $BASE_DIR/checkpoints/megatron/$model_name/release
        echo "release" > $BASE_DIR/checkpoints/megatron/$model_name/latest_checkpointed_iteration.txt
        echo "Convert to release version for $model_name"
    fi
fi

if [ `expr $ARNOLD_ID - 0` -eq 0 ]; then
    echo "Converting checkpoint to LAER-MoE format..."
    cd $BASE_DIR/LAER-MoE
    mkdir -p $BASE_DIR/checkpoints/laer
    python galvatron/tools/convert_to_laer.py \
        --input_dir $BASE_DIR/checkpoints/megatron/$model_name/release \
        --output_dir $BASE_DIR/checkpoints/laer/$model_name \
        --tp_size $tensor_parallel \
            --ep_size $expert_parallel \
            --num_layers $layer_num \
            --num_attention_heads $num_attention_heads \
            --hidden_size $hidden_size \
            --num_experts $expert_num
    echo "Checkpoint converted to LAER-MoE format for $model_name"
fi