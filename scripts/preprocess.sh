#!/bin/bash
# Preprocess datasets: fake, safety, and saroco
# This script downloads datasets from HuggingFace and saves them as CSV files

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Dataset Preprocessing Script"
echo "============================================================"
echo ""

# Process all datasets
uv run docker compose exec self_iterative_refinement uv run src/constraint_learning_rlhf/preprocess/format_dataset.py --dataset all

echo ""
echo "============================================================"
echo "All datasets processed!"
echo "============================================================"

