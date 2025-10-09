source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd ${MA_JOB_DIR}/runtime/MindSpeed-1.1
pip install -e .
cd ${MA_JOB_DIR}/runtime/Galvatron-ascend-2.4/galvatron/profile_hardware
pwd

NUM_NODES=32
NUM_GPUS_PER_NODE=8
NCCLTEST_DIR="../site_package/hccl_test"
ASCEND_DIR="/usr/local/Ascend/ascend-toolkit/latest"
MPI_PATH=/usr/local/openmpi-4.1.6/
START_MB=16
END_MB=256
SCALE=2
HOSTFILE="hostfile"
export PYTHONPATH=$PYTHONPATH:/${MA_JOB_DIR}/runtime/Galvatron-ascend-2.4

# These args will be directly added to nccl-test arguments
# export NCCLTEST_OTHER_ARGS="-x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_2,mlx5_5"

PROFILE_ARGS="
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
    --nccl_test_dir ${NCCLTEST_DIR} \
    --ascend_dir ${ASCEND_DIR} \
    --mpi_path ${MPI_PATH} \
    --start_mb ${START_MB} \
    --end_mb ${END_MB} \
    --scale ${SCALE} \
    --hostfile ${HOSTFILE} \
    --master_addr 10.50.112.88 \
    --master_port 23456 \
    --node_rank 0 \
    --avg_or_min_or_first first \
    --max_pp_deg 16 \
    --overlap_time_multiply 4"
python3 profile_hardware.py ${PROFILE_ARGS}

# export HCCL_CONNECT_TIMEOUT=3600
# export HCCL_EXEC_TIMEOUT=7200
# export HCCL_ASYNC_ERROR_HANDLING=3600
# export NUM_NODES=$MA_NUM_HOSTS
# export NUM_GPUS_PER_NODE=8
# MASTER_HOST=${MA_VJ_NAME}-${MA_TASK_NAME}-0.${MA_VJ_NAME}:1234
# export MASTER_ADDR=${MASTER_HOST%%:*}
# export MASTER_PORT=${MASTER_HOST##*:}
# export NODE_RANK=$VC_TASK_INDEX
# export PYTHONPATH=$PYTHONPATH:/${MA_JOB_DIR}/runtime/Galvatron-ascend-2.4

# python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE

# python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE



# bash scripts/profile_allreduce.sh

bash scripts/profile_p2p.sh

ls hardware_configs

cat hardware_configs/allreduce_bandwidth_32nodes_8gpus_per_node.json

cat hardware_configs/p2p_bandwidth_32nodes_8gpus_per_node.json
