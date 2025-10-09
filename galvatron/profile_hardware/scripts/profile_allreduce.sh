export NUM_NODES=$(echo $VC_TASK_NUM | awk '{print int($0)}') 
export NUM_GPUS_PER_NODE=8 
export MASTER_ADDR=$(echo $VC_TASK_HOSTS | cut -d',' -f1) # $MASTER_ADDR 
export MASTER_PORT=6000 # $MASTER_PORT
#export NCCL_SOCKET_IFNAME=ib0 
export NODE_RANK=$VC_TASK_INDEX # $RANK 
export PYTHONPATH=$PYTHONPATH:/opt/dpcvol/models/Galvatron-ascend-2.1 
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 128 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 128 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 64 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 64 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 64 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 64 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 32 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 32 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 32 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 32 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 16 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 16 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 16 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 16 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 8 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 8 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 8 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 8 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 4 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 4 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 4 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 4 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
