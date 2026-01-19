ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}
export NUM_NODES=4
export NNODES=4
export NUM_GPUS_PER_NODE=8
export GPUS_PER_NODE=8
export MASTER_ADDR=10.233.111.62
export MASTER_PORT=23456
export NODE_RANK=`expr $ARNOLD_ID - 0`

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_101,mlx5_103,mlx5_105,mlx5_107
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export GLOO_SOCKET_IFNAME=eth0