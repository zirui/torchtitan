
source /zirui/titan-env/bin/activate

# Set mlperf common path
export PYTHONPATH=/zirui/code/mlperf-common:$PYTHONPATH

# Set cluster config
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

NGPU=${NGPU:-8}
LOG_RANK=${LOG_RANK:-0}

# Set config file & training parameters
CONFIG_FILE=${CONFIG:-"./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf_preprocessed.toml"}

# Set training parameters
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-2e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1600}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \


# export USE_ROCM_AITER_ROPE_BACKEND=0
# Launch torchtitan
torchrun --nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${NGPU} \
    --rdzv_backend c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.experiments.flux.train --job.config_file ${CONFIG_FILE} \
    --training.batch_size=${BATCH_SIZE} \
    --training.seed=1234 \
    --optimizer.lr=${LR} \
    --lr_scheduler.warmup_steps=${WARMUP_STEPS} \
    --activation_checkpoint.mode="none" \

#--training.compile
# --parallelism.data_parallel_replicate_degree=8 \
# --parallelism.data_parallel_shard_degree=1 \
