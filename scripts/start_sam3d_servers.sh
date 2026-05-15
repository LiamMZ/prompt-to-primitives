#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/sam3d_server.py"
CHECKPOINT="${SAM3D_CKPT:-/home/liam/installs/sam-3d-objects/checkpoints/hf}"
CONDA_ENV="sam3d-objects"

echo "Starting SAM3D servers"
echo "  script:     $SERVER_SCRIPT"
echo "  checkpoint: $CHECKPOINT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

prefix() {
    local label=$1
    while IFS= read -r line; do
        printf '%s %s\n' "$label" "$line"
    done
}

start_server() {
    local gpu=$1
    local port=$2
    local label="[GPU${gpu}|${port}]"
    conda run -n "$CONDA_ENV" env \
        CUDA_VISIBLE_DEVICES=$gpu \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONUNBUFFERED=1 \
        python -u "$SERVER_SCRIPT" \
        --host 0.0.0.0 --port "$port" --checkpoint "$CHECKPOINT" 2>&1 | prefix "$label" &
    echo "$label started (pid $!)"
}

start_server 0 8766

echo "Waiting for GPU0 server to be ready …"
until curl -sf http://127.0.0.1:8766/health > /dev/null 2>&1; do sleep 2; done
echo "[GPU0|8766] ready"

start_server 1 8767

echo "Waiting for GPU1 server to be ready …"
until curl -sf http://127.0.0.1:8767/health > /dev/null 2>&1; do sleep 2; done
echo "[GPU1|8767] ready"

trap "echo 'Stopping servers…'; kill -- -$$ 2>/dev/null; wait" SIGINT SIGTERM

wait
