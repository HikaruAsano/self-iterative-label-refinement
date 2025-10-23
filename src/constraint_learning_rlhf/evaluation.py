"""Training evaluation helpers.

Author: Hikaru Asano
Affiliation: The University of Tokyo
"""

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from constraint_learning_rlhf.data import get_data_loaders
from constraint_learning_rlhf.metrics import compute_metrics, save_metrics


def _collect_predictions(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.where(logits > 0, 1, -1)
    processed_labels = labels[:, 0] if labels.ndim == 2 else labels
    return predictions, processed_labels


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    loss_fn: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            if loss_fn is not None:
                target_labels = labels[:, 1] if labels.ndim == 2 else labels
                loss = loss_fn(logits, target_labels)
                total_loss += loss.item() * labels.size(0)

            predictions, processed_labels = _collect_predictions(logits, labels)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(processed_labels.cpu().numpy())
            total_samples += labels.size(0)

    accelerator.wait_for_everyone()

    total_loss = gather_object([total_loss])
    total_samples = gather_object([total_samples])
    all_predictions = gather_object(all_predictions)
    all_labels = gather_object(all_labels)

    metrics = compute_metrics(all_predictions, all_labels)
    if loss_fn is not None:
        average_loss = sum(total_loss) / sum(total_samples)
        metrics["loss"] = average_loss

    return metrics


def evaluate_and_save_results(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    accelerator: Accelerator,
    log_dir: Path,
    eval_batch_size: int,
) -> Dict[str, Any]:
    model.eval()
    all_predictions = []
    all_labels = []

    dataset = dataset.add_column("id", list(range(len(dataset))))
    data_loader = get_data_loaders(tokenizer, dataset, eval_batch_size)

    data_loader = accelerator.prepare(data_loader)

    annotation_dir = log_dir / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    csv_file = annotation_dir / f"{accelerator.process_index}.csv"

    with torch.no_grad(), open(csv_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "pred"])
        for batch in tqdm(data_loader, desc="Evaluating"):
            indices = batch["id"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            predictions, labels = _collect_predictions(logits, labels)
            for label, pred, index in zip(labels, predictions, indices):
                text = dataset[int(index)]["text"]
                writer.writerow([text, int(label.item()), int(pred.item())])

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del input_ids, attention_mask, labels, outputs, logits, predictions
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    all_predictions = gather_object(all_predictions)
    all_labels = gather_object(all_labels)

    metrics = compute_metrics(all_predictions, all_labels)
    eval_dir = log_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, eval_dir)

    return metrics
