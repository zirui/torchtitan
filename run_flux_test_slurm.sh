#!/usr/bin/env bash
#SBATCH --job-name=flux-test
##SBATCH --output=logs/flux-test.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=7-00:00:00
##SBATCH --partition=xxx
##SBATCH --exclude=xxx

set -ex

# Set wandb
export WANDB_API_KEY=73ba028fade29b4baafdba2d6996a4865e28f410
export WANDB_PROJECT=mlperf-flux
export WANDB_RUN_NAME=torchtitan-flux-bf16-n1
#

DOCKER_IMAGE=${DOCKER_IMAGE:-"zirui3/autodev:v0.1.0"}
CONTAINER_NAME_PREFIX=${CONTAINER_NAME_PREFIX:-"flux-test"}
HOST_SHARED_ROOT=${HOST_SHARED_ROOT:-"/mnt/shared/zirui"}
HOST_MODEL_DIR=${HOST_MODEL_DIR:-"/mnt/vast/zirui/models"}
HOST_DATASET_DIR=${HOST_DATASET_DIR:-"/mnt/vast/zirui/data"}
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-"/zirui/code/torchtitan-main"}

GPUS_PER_NODE=${GPUS_PER_NODE:-${NGPU:-8}}
LOG_RANK=${LOG_RANK:-0}

# Flux launcher config
MODULE=${MODULE:-flux}
CONFIG=${CONFIG:-flux_schnell_mlperf_preprocessed}

# Quick test parameters
STEPS=${STEPS:-30000}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-64}
LR=${LR:-2e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1600}
AC_MODE=${AC_MODE:-none}
DEBUG_SEED=${DEBUG_SEED:-1234}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
    MASTER_PORT=${MASTER_PORT:-1234}
    NNODES=${NNODES:-${SLURM_JOB_NUM_NODES}}
else
    MASTER_ADDR=${MASTER_ADDR:-localhost}
    MASTER_PORT=${MASTER_PORT:-1234}
    NNODES=${NNODES:-1}
fi

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}, GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "DOCKER_IMAGE=${DOCKER_IMAGE}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -s -- "$@" <<EOF_INNER
set -ex

export TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}"

NODE_RANK=\${SLURM_NODEID:-0}
CONTAINER_NAME="${CONTAINER_NAME_PREFIX}-\${SLURM_JOB_ID:-job}-\${NODE_RANK}"

docker rm -f "\${CONTAINER_NAME}" || true

docker_env_args=(
    -e MASTER_ADDR="${MASTER_ADDR}"
    -e MASTER_PORT="${MASTER_PORT}"
    -e NNODES="${NNODES}"
    -e NODE_RANK="\${NODE_RANK}"
    -e GPUS_PER_NODE="${GPUS_PER_NODE}"
    -e LOG_RANK="${LOG_RANK}"
    -e MODULE="${MODULE}"
    -e CONFIG="${CONFIG}"
    -e STEPS="${STEPS}"
    -e LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE}"
    -e LR="${LR}"
    -e WARMUP_STEPS="${WARMUP_STEPS}"
    -e AC_MODE="${AC_MODE}"
    -e DEBUG_SEED="${DEBUG_SEED}"
    -e TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}"
    -e PYTORCH_ALLOC_CONF="expandable_segments:True"
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
)

if [ -n "\${WANDB_API_KEY:-}" ]; then
    docker_env_args+=(-e WANDB_API_KEY="\${WANDB_API_KEY}")
fi
if [ -n "\${WANDB_PROJECT:-}" ]; then
    docker_env_args+=(-e WANDB_PROJECT="\${WANDB_PROJECT}")
fi
if [ -n "\${WANDB_RUN_NAME:-}" ]; then
    docker_env_args+=(-e WANDB_RUN_NAME="\${WANDB_RUN_NAME}")
fi

docker run --rm -i \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    --name "\${CONTAINER_NAME}" \
    -v "${HOST_SHARED_ROOT}:/mnt/shared/zirui" \
    -v "${HOST_SHARED_ROOT}:/zirui" \
    -v "${HOST_MODEL_DIR}:/models" \
    -v "${HOST_DATASET_DIR}:/dataset" \
    -w "${CONTAINER_WORKDIR}" \
    "\${docker_env_args[@]}" \
    "${DOCKER_IMAGE}" \
    bash -s -- "\$@" <<'INNER'
set -ex

if [ -f "/zirui/titan-env/bin/activate" ]; then
    source /zirui/titan-env/bin/activate
fi

torchrun \
    --nnodes="\${NNODES}" \
    --node_rank="\${NODE_RANK}" \
    --nproc_per_node="\${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \
    --local-ranks-filter "\${LOG_RANK}" \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    --module "\${MODULE}" \
    --config "\${CONFIG}" \
    --training.steps="\${STEPS}" \
    --training.local_batch_size="\${LOCAL_BATCH_SIZE}" \
    --optimizer.lr="\${LR}" \
    --lr_scheduler.warmup_steps="\${WARMUP_STEPS}" \
    --activation_checkpoint.mode="\${AC_MODE}" \
    --debug.seed="\${DEBUG_SEED}" \
    --metrics.enable_wandb \
    "\$@"
INNER
EOF_INNER
