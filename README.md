<div align="center">
<h1>Self Iterative Label Refinement via Robust Unlabeled Learning</h1>

<p align="center">
    <a href="https://hikaruasano.github.io/">Hikaru Asano</a><sup>1</sup> &nbsp;
    <a href="https://tadashik.github.io/">Tadashi Kozuno</a><sup>2</sup> &nbsp;
    <a href="https://yukinobaba.jp/">Yukino Baba</a><sup>1</sup>
</p>

<p align="center">
    <sup>1</sup>The University of Tokyo &nbsp;
    <sup>2</sup>OMRON SINIC X
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2502.12565"><img src="https://img.shields.io/badge/arXiv-paper-orange" alt="arXiv paper"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
</p>

</div>

---

## Overview

This is the official repository of [Self Iterative Label Refinement via Robust Unlabeled Learning](https://arxiv.org/abs/2502.12565) (NeurIPS 2025).
The pipeline includes dataset reconstruction, preprocessing, and training for iterative label refinement with LLM
annotations across multiple classification datasets.

## Requirements

- NVIDIA GPU (training assumes GPU environment)
- Docker and Docker Compose
- Permission to access `meta-llama/Llama-3.2-*` on Hugging Face
- Hugging Face token and Weights & Biases (wandb) account info in `.env`

## Setup

### 1) Environment variables

Copy `.env.example` to `.env` and set the required values:

```bash
cp .env.example .env
```

Required keys:

- `HUGGINGFACE_HUB_TOKEN`
- `WANDB_API_KEY`
- `WANDB_ENTITY`

### 2) Dataset prerequisites

The preprocessing pipeline downloads datasets from Hugging Face, with the following exceptions:

- `fake` (optional): If you want to run experiments on the fake news dataset, download the Kaggle dataset and place `Fake.csv` and `True.csv` under `data/fake_news/`
  - https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  - You can skip this step if you don't need to run experiments on the fake news dataset

### 3) Start container and preprocess

```bash
bash setup.sh
```

This runs `docker compose up -d` and `bash scripts/preprocess.sh` to download and preprocess datasets.

## Training (run.sh)

Training starts with `bash scripts/run.sh`. Key options are below.

```bash
bash scripts/run.sh \
  --dataset fake \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --annotate-model Llama-2-7b-chat-hf \
  --seed 0 \
  --log-prefix for_open_source \
  --gamma -0.001
```

Add `--debug` to run on small dataset subsets for a quick check.

Supported datasets: `fake`, `safety`, `saroco`, `protein`.

Environment overrides:

- `SERVICE_NAME` (default: `self_iterative_refinement`): Docker service name for the container
- `DATASET_NAME`: Override the dataset specified in `--dataset`
- `MODEL_NAME`: Override the model specified in `--model`
- `ANNOTATE_MODEL_NAME`: Override the annotation model specified in `--annotate-model`
- `SEED`: Override the random seed specified in `--seed`
- `LOG_PREFIX`: Override the logging prefix specified in `--log-prefix`
- `GAMMA`: Override the gamma parameter specified in `--gamma`
- `USE_DEBUG`: Set to `true` to enable debug mode (equivalent to `--debug` flag)

## Outputs and logging

- Training outputs are saved under `outputs/train/<log_prefix>/<dataset>/...`
- Weights & Biases tracking uses `WANDB_ENTITY` and the `--log-prefix` value

## Files

- `setup.sh`: Runs `docker compose up -d` and preprocessing via `scripts/preprocess.sh`.
- `scripts/preprocess.sh`: Calls `format_dataset.py` to preprocess all datasets.
- `scripts/run.sh`: Training entrypoint; accepts dataset/model arguments.
- `scripts/train.py`: Main training logic.
- `src/constraint_learning_rlhf/preprocess/format_dataset.py`: Dataset preprocessing logic.
- `data/`: Storage for raw and preprocessed datasets.

## Citation

```bibtex
@inproceedings{
asano2025self,
title={Self Iterative Label Refinement via Robust Unlabeled Learning},
author={Hikaru Asano and Tadashi Kozuno and Yukino Baba},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
}
```
