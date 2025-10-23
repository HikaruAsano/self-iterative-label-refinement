#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

SERVICE_NAME="${SERVICE_NAME:-self_iterative_refinement}"
DATASET_NAME="${DATASET_NAME:-fake}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
ANNOTATE_MODEL_NAME="${ANNOTATE_MODEL_NAME:-Llama-2-7b-chat-hf}"
SEED="${SEED:-0}"
LOG_PREFIX="${LOG_PREFIX:-for_open_source}"
GAMMA="${GAMMA:--0.001}"
USE_DEBUG="${USE_DEBUG:-false}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run.sh [options]

Options:
  --dataset NAME          Dataset name (fake, safety, saroco, protein)
  --model NAME            Training model name
  --annotate-model NAME   Annotator model name
  --seed INT              Random seed (default: 0)
  --log-prefix NAME       W&B log prefix
  --gamma FLOAT           Loss gamma (default: -0.001)
  --debug                 Enable debug mode (small dataset)
  -h, --help              Show this help

Environment overrides:
  SERVICE_NAME, DATASET_NAME, MODEL_NAME, ANNOTATE_MODEL_NAME,
  SEED, LOG_PREFIX, GAMMA, USE_DEBUG
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --annotate-model)
            ANNOTATE_MODEL_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --log-prefix)
            LOG_PREFIX="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --debug)
            USE_DEBUG="true"
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

DEBUG_FLAG=()
if [[ "$USE_DEBUG" == "true" ]]; then
    DEBUG_FLAG=(--use_debug)
fi

docker compose exec "$SERVICE_NAME" accelerate launch \
    scripts/train.py \
    --dataset_name="$DATASET_NAME" \
    --model_name="$MODEL_NAME" \
    --annotate_model_name="$ANNOTATE_MODEL_NAME" \
    --seed="$SEED" \
    --log_prefix="$LOG_PREFIX" \
    --gamma="$GAMMA" \
    "${DEBUG_FLAG[@]}"
