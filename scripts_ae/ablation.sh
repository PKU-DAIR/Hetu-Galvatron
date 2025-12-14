if [ $# -ne 1 ]; then
    echo "Usage: $0 <approach>"
    exit 1
fi

approach=$1
model_name=mixtral-8x7b-e8k2
aux_loss=0
dataset=wikitext
batch_size=128
seq_length=8192
chunk=2
layer_num=32
capacity=2

if [ "$approach" != "LAER" ] && [ "$approach" != "no_even" ] && [ "$approach" != "no_pq" ] && [ "$approach" != "no_comm_opt" ] && [ "$approach" != "FSDP" ]; then
    echo "Invalid approach: $approach"
    exit 1
fi

export ABLATION_APPROACH=$approach
export BASE_DIR=$(cd $(dirname $0)/../..; pwd)

if [ -d "$BASE_DIR/checkpoints/laer/$model_name" ]; then
    echo "Generating ablation study for $model_name with $approach..."
    cd $BASE_DIR/LAER-MoE/galvatron/models/moe
    mkdir -p $BASE_DIR/LAER-MoE/galvatron/models/moe/training_log/ablation
    bash scripts/train_ablation.sh $model_name $layer_num $aux_loss $dataset $capacity $batch_size $chunk $seq_length 0
    echo "Ablation study finished for $model_name"
else
    echo "checkpoints/laer/$model_name does not exist"
    exit 1
fi