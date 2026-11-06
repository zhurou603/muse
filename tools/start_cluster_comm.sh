#!/usr/bin/env bash
set -euo pipefail

node_count=4
test_port=29686
log_subdir=cluster_comm_logs
if [ "$#" -eq 1 ] && [ "$1" = "--six-nodes" ]; then
    node_count=6
    test_port=29687
    log_subdir=cluster_comm6_logs
elif [ "$#" -ne 0 ]; then
    echo "Usage: bash start_cluster_comm.sh [--six-nodes]" >&2
    exit 1
fi
expected_world_size=$((node_count * 8))

node_host=$(hostname -s)
case "$node_host" in
    VM-2-17-tencentos) node_rank=0; node_ip=172.17.2.17 ;;
    VM-2-184-tencentos) node_rank=1; node_ip=172.17.2.184 ;;
    VM-2-46-tencentos) node_rank=2; node_ip=172.17.2.46 ;;
    VM-1-84-tencentos) node_rank=3; node_ip=172.17.1.84 ;;
    VM-0-171-tencentos) node_rank=4; node_ip=172.17.0.171 ;;
    VM-0-178-tencentos) node_rank=5; node_ip=172.17.0.178 ;;
    *) echo "ERROR: unexpected host $node_host" >&2; exit 1 ;;
esac
if [ "$node_rank" -ge "$node_count" ]; then
    echo "ERROR: this host requires --six-nodes and coordinated availability of all 8 cards." >&2
    exit 1
fi

runtime_python=/home/zhaoyao/miniconda3/envs/muse-pretrain/bin/python
tools_dir=/yinghepool/tech-dcu/zhaoyao/tools
check_script="$tools_dir/check_ddp_runtime.py"
test_master=172.17.2.17
test_logs="$tools_dir/$log_subdir"

test -x "$runtime_python"
test -f "$check_script"
ip -4 -o addr show dev eth0 | grep -Fq "inet $node_ip/"
if [ "$node_rank" -eq 0 ]; then
    listeners=$(ss -H -ltn "sport = :$test_port")
    if [ -n "$listeners" ]; then
        echo "ERROR: master port $test_port is already in use; no process stopped." >&2
        exit 1
    fi
fi

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
export NCCL_SOCKET_IFNAME='=eth0'
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA='=mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1'
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
unset NCCL_DEBUG_FILE

mkdir -p "$test_logs"
log_file="$test_logs/${node_host}_$(date +%Y%m%d_%H%M%S_%N).log"
echo "HOST=$node_host NODE_RANK=$node_rank MASTER=$test_master:$test_port"
echo "LOG=$log_file"
echo "Start the same mode on all $node_count nodes; first nodes wait for the others."
echo "Diagnostic only: $expected_world_size ranks, tiny tensors, no production model or optimizer updates."

if timeout --signal=TERM --kill-after=10s 600s "$runtime_python" -B -u -m torch.distributed.run --nnodes="$node_count" --nproc_per_node=8 --node_rank="$node_rank" --master_addr="$test_master" --master_port="$test_port" --max_restarts=0 "$check_script" --expected-world-size "$expected_world_size" 2>&1 | tee "$log_file"; then
    echo "NODE LAUNCH COMPLETE: $node_host; inspect all $node_count logs for final acceptance."
else
    test_status=$?
    echo "ERROR: diagnostic failed or timed out, exit=$test_status; log=$log_file" >&2
    exit "$test_status"
fi
