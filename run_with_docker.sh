#!/usr/bin/env bash

set -euo pipefail

DOCKER_IMAGE="wanghanthu/torchtitan:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2-patch"

# Allocate a TTY only when this script is launched interactively.
docker_tty_args=()
if [[ -t 0 && -t 1 ]]; then
    docker_tty_args=(-it)
fi

docker run "${docker_tty_args[@]}" --rm \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --net=host \
    --privileged \
    --shm-size 20G \
    --name titan \
    -v /shared_nfs:/shared_nfs \
    -v /shared_nfs/zirui/models:/models \
    -v /shared_nfs/zirui:/zirui \
    -v /shared_nfs/zirui/data:/data \
    -w /zirui \
    "${DOCKER_IMAGE}" \
    bash -lc '
        set -euo pipefail
        cd /zirui/code/ALTO
        pip install -e .
        cd /zirui/code/torchtitan
        exec bash flux.sh
    '
