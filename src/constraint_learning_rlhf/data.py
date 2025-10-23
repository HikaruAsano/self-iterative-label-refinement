"""Dataset loading and preparation utilities.

Author: Hikaru Asano
Affiliation: The University of Tokyo
"""

import csv
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

from constraint_learning_rlhf.metrics import compute_metrics

SEED = 42
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def format_labels(loss_type: str, dataset: Dataset) -> Dataset:
    """
    Format labels based on the loss type.

    Args:
        loss_type: Type of loss ('pn', 'pu', 'uu').
        dataset: The dataset to format.

    Returns:
        Dataset: The dataset with formatted labels.
    """
    if loss_type == "uu":
        dataset = dataset.remove_columns("label").rename_column("dataset", "label")
    elif loss_type in {"upu", "nnpu"}:
        dataset = dataset.remove_columns("label").rename_column("is_labeled", "label")
    return dataset


def get_tokenizer(tokenizer_instance: Callable) -> Callable:
    """
    Get a tokenizer function for preprocessing.

    Args:
        tokenizer_instance: The tokenizer instance.

    Returns:
        A function that tokenizes a batch of texts.
    """

    def tokenize(batch: dict) -> dict:
        return tokenizer_instance(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

    return tokenize


def get_data_loaders(
    tokenizer: Callable, dataset: Dataset, batch_size: int
) -> DataLoader:
    """
    Create a DataLoader from a dataset.

    Args:
        tokenizer: The tokenizer function.
        dataset: The dataset to load.
        batch_size: Batch size for DataLoader.

    Returns:
        DataLoader for the dataset.
    """
    tokenize_func = get_tokenizer(tokenizer)
    tokenized_dataset = dataset.map(
        tokenize_func, batched=True, remove_columns=["text"]
    ).with_format("torch")
    data_loader = DataLoader(tokenized_dataset, shuffle=True, batch_size=batch_size)
    return data_loader


def load_preprocessed_csv(
    dataset_name: str,
    split: str,
) -> pd.DataFrame:
    """
    Load preprocessed CSV file.

    Args:
        dataset_name: Name of the dataset (e.g., 'safety', 'helpful')
        split: 'train' or 'test'

    Returns:
        DataFrame with columns: text, label, model1, model2, ...
    """
    csv_path = DATA_DIR / f"{dataset_name}_{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {csv_path}")

    csv.field_size_limit(10**7)
    df = pd.read_csv(csv_path, encoding="utf-8")

    return df


def get_available_models_from_csv(df: pd.DataFrame) -> List[str]:
    """
    Get list of model names from CSV columns.

    Args:
        df: DataFrame loaded from preprocessed CSV

    Returns:
        List of model names (columns other than 'text' and 'label')
    """
    return [col for col in df.columns if col not in ["text", "label"]]


def load_preprocessed_dataset(
    dataset_name: str,
    model_name: str,
    validation_size: float = 0.125,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load preprocessed dataset with model predictions.

    This function replaces load_annotated_dataset and provides a simpler interface
    for loading datasets from preprocessed CSV files.

    Args:
        dataset_name: Name of the dataset (e.g., 'safety', 'helpful')
        model_name: Name of the model whose predictions to use
        validation_size: Proportion of training data for validation

    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
        - train_dataset: {"text": str, "label": [ground_truth, model_prediction]}
        - validation_dataset: {"text": str, "label": [ground_truth, model_prediction]}
        - test_dataset: {"text": str, "label": ground_truth}
    """
    train_df = load_preprocessed_csv(dataset_name, "train")
    test_df = load_preprocessed_csv(dataset_name, "test")

    available_models = get_available_models_from_csv(train_df)
    if model_name not in available_models:
        raise ValueError(
            f"Model '{model_name}' not found in dataset. "
            f"Available models: {available_models}"
        )

    train_df = train_df[train_df[model_name].notna()].copy()

    train_data = {
        "text": train_df["text"].tolist(),
        "label": train_df["label"].astype(int).tolist(),
        "dataset": train_df[model_name].astype(int).tolist(),
    }
    train_validate = Dataset.from_dict(train_data)

    train_valid_split = train_validate.train_test_split(
        test_size=validation_size, seed=SEED
    )

    train_dataset = format_labels("uu", train_valid_split["train"])
    validation_dataset = train_valid_split["test"].map(
        lambda data: {
            "text": data["text"],
            "label": [data["label"], data.pop("dataset", None)],
        },
        remove_columns=["dataset"],
    )

    test_data = {
        "text": test_df["text"].tolist(),
        "label": test_df["label"].astype(int).tolist(),
    }
    test_dataset = Dataset.from_dict(test_data)

    return train_dataset, validation_dataset, test_dataset


def load_preprocessed_original_dataset(
    dataset_name: str,
    split: str = "train",
) -> Dataset:
    """
    Load preprocessed dataset with only text and ground truth labels.

    This function is used for iterative updates where we need the original
    dataset without model predictions.

    Args:
        dataset_name: Name of the dataset
        split: 'train' or 'test'

    Returns:
        Dataset with {"text": str, "label": int}
    """
    df = load_preprocessed_csv(dataset_name, split)

    data = {
        "text": df["text"].tolist(),
        "label": df["label"].astype(int).tolist(),
    }

    return Dataset.from_dict(data)


def compute_theta_from_preprocessed(
    dataset_name: str,
    model_name: str,
    seed: int,
    num_labeled_size: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute theta_1 and theta_2 from preprocessed dataset.

    This is a simplified version of compute_theta_1_theta_2 that uses
    the preprocessed CSV directly instead of loading from multiple sources.

    theta_1 = P(model predicts 1 | ground truth is 1)
    theta_2 = P(model predicts 1 | ground truth is -1)

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        seed: Random seed for sampling
        num_labeled_size: Number of labeled samples to use.
            If None, use all available data with ground truth labels.

    Returns:
        Tuple of (theta_1, theta_2)
    """
    df = load_preprocessed_csv(dataset_name, "train")

    df = df[df[model_name].notna()].copy()

    positive_df = df[df["label"] == 1]
    negative_df = df[df["label"] == -1]

    if num_labeled_size is None:
        print("Computing theta using all available data (ground truth labels)")
        selected_positive = list(positive_df.index)
        selected_negative = list(negative_df.index)
    else:
        random.seed(seed)
        half_size = num_labeled_size // 2

        positive_indices = list(positive_df.index)
        negative_indices = list(negative_df.index)

        random.shuffle(positive_indices)
        random.shuffle(negative_indices)

        selected_positive = positive_indices[:half_size]
        selected_negative = negative_indices[:half_size]

    pred_labels = []
    true_labels = []

    for idx in selected_positive:
        pred_labels.append(int(df.loc[idx, model_name]))
        true_labels.append(int(df.loc[idx, "label"]))

    for idx in selected_negative:
        pred_labels.append(int(df.loc[idx, model_name]))
        true_labels.append(int(df.loc[idx, "label"]))

    metrics = compute_metrics(pred_labels, true_labels)
    theta_1 = metrics["pred1_label1_rate"]
    theta_2 = metrics["pred0_label1_rate"]

    print(
        f"Computed theta_1={theta_1:.4f}, theta_2={theta_2:.4f} "
        f"from {len(pred_labels)} samples"
    )

    return theta_1, theta_2


def compute_positive_prior_from_original(
    dataset_name: str,
    split: str = "train",
) -> float:
    """
    Compute positive prior (P(Y=1)) from original dataset labels.

    This calculates the proportion of positive labels (label=1) in the
    original dataset, which represents the true class distribution.

    Args:
        dataset_name: Name of the dataset
        split: 'train' or 'test' (default: 'train')

    Returns:
        Positive prior probability (proportion of label=1)
    """
    df = load_preprocessed_csv(dataset_name, split)

    positive_count = (df["label"] == 1).sum()
    total_count = len(df)

    positive_prior = positive_count / total_count

    return positive_prior


def get_annotation_metrics_from_preprocessed(
    dataset_name: str,
    model_name: str,
) -> dict:
    """
    Compute annotation metrics from preprocessed dataset.

    This replaces get_annotation_metrics from utils.py and computes
    metrics directly from the preprocessed CSV.

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model

    Returns:
        Dictionary with 'train' and 'test' metrics
    """
    metrics = {"train": {}, "test": {}}

    for split in ["train", "test"]:
        df = load_preprocessed_csv(dataset_name, split)
        original_size = len(df)

        df_valid = df[df[model_name].notna()].copy()

        if len(df_valid) == 0:
            print(f"Warning: No predictions found for {model_name} in {split}")
            continue

        if original_size != len(df_valid):
            print(
                f"  [{split}] Filtered: {original_size} -> {len(df_valid)} "
                f"(removed {original_size - len(df_valid)} rows with None predictions)"
            )

        pred_labels = df_valid[model_name].astype(int).tolist()
        true_labels = df_valid["label"].astype(int).tolist()

        split_metrics = compute_metrics(pred_labels, true_labels)
        metrics[split] = split_metrics

    return metrics


def prepare_iterative_update_dataset(
    dataset_dir: Path,
    validation_size: float = 0.125,
) -> Tuple[Dataset, Dataset]:
    """
    Prepare train/validation datasets from iterative update annotations.

    Args:
        dataset_dir: Directory containing annotation CSV files.
        validation_size: Proportion of data used for validation.

    Returns:
        Tuple of (train_dataset, validation_dataset).
    """
    texts, labels, datasets = [], [], []

    for file_path in dataset_dir.glob("*.csv"):
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                texts.append(row[0])
                labels.append(int(row[1]))
                datasets.append(int(row[2]))

    dataset = Dataset.from_dict({"text": texts, "label": labels, "dataset": datasets})

    train_valid_split = dataset.train_test_split(test_size=validation_size)

    train_dataset = format_labels("uu", train_valid_split["train"])
    validation_dataset = train_valid_split["test"].map(
        lambda data: {
            "text": data["text"],
            "label": [data["label"], data.pop("dataset", None)],
        },
        remove_columns=["dataset"],
    )

    return train_dataset, validation_dataset


prepare_dataset = prepare_iterative_update_dataset
