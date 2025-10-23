"""Metrics computation utilities.

Author: Hikaru Asano
Affiliation: The University of Tokyo
"""

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def compute_metrics(all_preds: List[int], all_labels: List[int]) -> Dict[str, Any]:
    label_array = np.where(np.array(all_labels) == -1, 0, 1)
    pred_array = np.where(np.array(all_preds) == -1, 0, 1)

    confusion_matrix = np.zeros((2, 2), dtype=int)
    for true, pred in zip(label_array, pred_array):
        confusion_matrix[true, pred] += 1

    total_samples = confusion_matrix.sum()
    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / total_samples

    precision = (
        confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
        if (confusion_matrix[1, 1] + confusion_matrix[0, 1]) > 0
        else 0
    )
    recall = (
        confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
        if (confusion_matrix[1, 1] + confusion_matrix[1, 0]) > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    pred1_label1_rate = (
        confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
        if (confusion_matrix[1, 1] + confusion_matrix[0, 1]) > 0
        else 0
    )
    pred0_label1_rate = (
        confusion_matrix[1, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
        if (confusion_matrix[0, 0] + confusion_matrix[1, 0]) > 0
        else 0
    )

    pred1_rate = (confusion_matrix[1, 1] + confusion_matrix[0, 1]) / total_samples
    pred0_rate = (confusion_matrix[0, 0] + confusion_matrix[1, 0]) / total_samples

    result = {
        "confusion_matrix": confusion_matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred1_label1_rate": pred1_label1_rate,
        "pred0_label1_rate": pred0_label1_rate,
        "pred1_rate": pred1_rate,
        "pred0_rate": pred0_rate,
    }
    return result


def save_metrics(metrics: Dict[str, Any], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "confusion_matrix.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [metrics["confusion_matrix"][0][0], metrics["confusion_matrix"][0][1]]
        )
        writer.writerow(
            [metrics["confusion_matrix"][1][0], metrics["confusion_matrix"][1][1]]
        )

    with open(log_dir / "classification_report.csv", "w") as f:
        writer = csv.writer(f)
        for key, value in metrics.items():
            if key != "confusion_matrix":
                writer.writerow([key, value])
