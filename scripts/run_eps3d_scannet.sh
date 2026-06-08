#!/bin/bash
set -e

# Usage:
#   bash run_eps3d_scannet.sh                  # novel-view (default)
#   bash run_eps3d_scannet.sh context          # context-view

EVAL_MODE="${1:-novel}"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORTABLE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

# ---- Environment setup ----
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="/home/rszhu/anaconda3/envs/anysplat/bin:${CUDA_HOME}/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Configurable paths (override via env vars) ----
MODEL_PATH="${MODEL_PATH:-${PORTABLE_ROOT}/checkpoints/EPS3D}"
DATA_ROOT="${DATA_ROOT:-${PORTABLE_ROOT}/data/scannet_test}"
export CLIP_CHECKPOINT_PATH="${CLIP_CHECKPOINT_PATH:-${PORTABLE_ROOT}/checkpoints/ViT-B-32.pt}"
GPU_ID="${GPU_ID:-0}"

# ---- Evaluation settings ----
VIEW_NUMBER=8
LABEL_FLAG=1

EVAL_CONTEXT_FLAG=""
if [ "$EVAL_MODE" = "context" ]; then
    EVAL_CONTEXT_FLAG="--eval_context"
    SAVE_DIR="res_eps3d/scannet/context"
else
    SAVE_DIR="res_eps3d/scannet/novel"
fi

echo "Starting EPS3D Panoptic testing (8 views, ${EVAL_MODE}-view)..."
echo "Configuration:"
echo "  PROJECT_ROOT:         ${PROJECT_ROOT}"
echo "  MODEL_PATH:           ${MODEL_PATH}"
echo "  DATA_ROOT:            ${DATA_ROOT}"
echo "  CLIP_CHECKPOINT_PATH: ${CLIP_CHECKPOINT_PATH}"
echo "  CUDA_HOME:            ${CUDA_HOME}"
echo "  Python:               $(which python) ($(python --version 2>&1))"
echo "  GPU_ID:               ${GPU_ID}"
echo "  MODE:                 ${EVAL_MODE}-view"

mkdir -p "${SAVE_DIR}"
cd "${SCRIPT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python test_eps3d_panoptic.py \
    --test_dataset "TestDataset(split='test', ROOT='${DATA_ROOT}', instance_flag=0, label_flag=${LABEL_FLAG}, resolution=(256, 256), seed=777, num_views=${VIEW_NUMBER})" \
    --test_results_dir "${SAVE_DIR}/${VIEW_NUMBER}/" \
    --model_path "${MODEL_PATH}" \
    --num_views_value ${VIEW_NUMBER} \
    --root "${DATA_ROOT}" \
    --label_flag ${LABEL_FLAG} \
    ${EVAL_CONTEXT_FLAG}

echo "Running PQ evaluation..."
python evaluate_pq.py \
    --results_dir "${SAVE_DIR}/${VIEW_NUMBER}/"

echo "EPS3D Panoptic testing (8 views, ${EVAL_MODE}-view) complete!"
