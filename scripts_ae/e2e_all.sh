source ./scripts_ae/env.sh

approach_list=("LAER" "FLEX" "FSDP" "megatron")
model_name_list=("mixtral-8x7b-e8k2" "mixtral-8x7b-e16k4" "mixtral-8x22b-e8k2" "mixtral-8x22b-e16k4" "qwen-8x7b-e8k2" "qwen-8x7b-e16k4")
aux_loss_list=("0.0" "1e-4")
dataset_list=("wikitext" "C4")

for aux_loss in ${aux_loss_list[@]}; do
    for model_name in ${model_name_list[@]}; do
        for dataset in ${dataset_list[@]}; do
            for approach in ${approach_list[@]}; do
                echo "Running $approach $model_name $aux_loss $dataset"
                bash scripts_ae/e2e.sh $approach $model_name $aux_loss $dataset
                echo "Finished $approach $model_name $aux_loss $dataset"
            done
        done
    done
done