#!/usr/bin/env bash
#SBATCH -A amd-spur
#SBATCH -p amd-spur
#SBATCH -N 1
#SBATCH -G 8
#SBATCH --job-name=flux
#SBATCH --time=5-00:00:00
#SBATCH --output=flux-%j.log
#SBATCH --error=flux-%j.log

set -euo pipefail

IMAGE="wanghanthu/torchtitan:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2-patch"
CONTAINER_NAME="titan-${SLURM_JOB_ID:-$$}"

docker run --rm \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --net=host \
    --privileged \
    --shm-size 20G \
    --name "${CONTAINER_NAME}" \
    --env ROCR_VISIBLE_DEVICES \
    --env HIP_VISIBLE_DEVICES \
    --env CUDA_VISIBLE_DEVICES \
    -v /shared_nfs:/shared_nfs \
    -v /shared_nfs/zirui/models:/models \
    -v /shared_nfs/zirui:/zirui \
    -v /shared_nfs/zirui/data:/data \
    -w /zirui/code/torchtitan \
    "${IMAGE}" \
    bash ./flux.sh
