#!/usr/bin/env bash

set -ex

if [ -f "/zirui/titan-env/bin/activate" ]; then
    # Optional local venv used on this machine.
    source /zirui/titan-env/bin/activate
fi

# if [ -d "/zirui/code/mlperf-common" ]; then
#     export PYTHONPATH="/zirui/code/mlperf-common:${PYTHONPATH}"
# fi

# Set wandb
export WANDB_API_KEY=73ba028fade29b4baafdba2d6996a4865e28f410
export WANDB_PROJECT=mlperf-flux
export WANDB_RUN_NAME=torchtitan-flux-bf16-n1
#

# Cluster config
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

NGPU=${NGPU:-8}
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

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# Optional ROCm overrides:
# export USE_ROCM_AITER_ROPE_BACKEND=0


PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${NGPU} \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --local-ranks-filter "${LOG_RANK}" \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    --module "${MODULE}" \
    --config "${CONFIG}" \
    --training.steps="${STEPS}" \
    --training.local_batch_size="${LOCAL_BATCH_SIZE}" \
    --optimizer.lr="${LR}" \
    --lr_scheduler.warmup_steps="${WARMUP_STEPS}" \
    --activation_checkpoint.mode="${AC_MODE}" \
    --metrics.enable_wandb \
    "$@"
