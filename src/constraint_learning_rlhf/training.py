"""Training loops and logging utilities.

Author: Hikaru Asano
Affiliation: The University of Tokyo
"""

import argparse
import csv
import gc
import logging
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from constraint_learning_rlhf.evaluation import evaluate_model

HOME_DIR = Path(__file__).parent.parent.parent


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_logger(
    args: argparse.Namespace, timestamp: str
) -> Tuple[logging.Logger, Path]:
    """
    Initialize logger and create log directory.

    Args:
        args: Command line arguments.
        timestamp: Current timestamp string.

    Returns:
        Tuple of (logger, log_dir).
    """
    log_dir = (
        HOME_DIR
        / "outputs"
        / "train"
        / args.log_prefix
        / args.dataset_name
        / f"annotate_{args.annotate_model_name}_train_{args.model_name}_{args.loss_type}_{args.lr:.1e}"
        / f"seed{args.seed}_{timestamp}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        filename=log_dir / "log.txt",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration: {args}")
    logger.info(f"Args: {vars(args)}")
    return logger, log_dir


def log_metrics(
    accelerator: Accelerator, metrics: dict, prefix: str, step: int
) -> None:
    """
    Log metrics to wandb via accelerator.

    Args:
        accelerator: Accelerator instance.
        metrics: Dictionary of metrics to log.
        prefix: Prefix for metric names (e.g., 'train', 'eval').
        step: Current step number.
    """
    logged_metrics = {
        f"{prefix}/{key}": value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }
    logged_metrics[f"{prefix}/step"] = step
    accelerator.log(logged_metrics)


def save_metrics(metrics: dict, file_path: Path, iteration: int) -> None:
    """
    Save metrics to CSV file.

    Args:
        metrics: Dictionary of metrics to save.
        file_path: Path to the CSV file.
        iteration: Current iteration number.
    """
    metrics["iteration"] = iteration
    metrics = {
        key: value for key, value in metrics.items() if key != "confusion_matrix"
    }
    if file_path.exists():
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
        with open(file_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(metrics)
    else:
        with open(file_path, "w", newline="") as csvfile:
            fieldnames = ["iteration"] + [key for key in metrics if key != "iteration"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(metrics)


def save_model(
    model: nn.Module, log_dir: Path, accelerator: Accelerator, suffix: str
) -> None:
    model_path = log_dir / suffix
    accelerator.unwrap_model(model).save_pretrained(model_path)
    logging.getLogger().info(f"Model saved to {model_path}")


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    scheduler,
    accelerator: Accelerator,
    loss_fn: nn.Module,
    epoch: int,
    total_epochs: int,
    current_iterates: int = 0,
    steps_so_far: int = 0,
) -> float:
    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Iterative Update {current_iterates} - Epoch {epoch}/{total_epochs}",
    )
    for step, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits.squeeze(-1)
        loss = loss_fn(logits, batch["label"])

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        loss_value = loss.item()
        pred_1_rate = (logits > 0).sum() / len(logits)
        epoch_loss += loss_value

        current_step = (epoch - 1) * len(train_loader) + step + steps_so_far
        log_metrics(
            accelerator,
            {
                "loss": loss_value,
                "learning_rate": scheduler.get_last_lr()[0],
                "pred_1_rate": pred_1_rate,
            },
            "train",
            current_step,
        )
        progress_bar.set_postfix({"loss": loss_value})

        del outputs, logits, loss
        torch.cuda.empty_cache()

    avg_epoch_loss = epoch_loss / len(train_loader)
    gc.collect()
    torch.cuda.empty_cache()
    return avg_epoch_loss


def validate_epoch(
    model: nn.Module,
    validate_loader: DataLoader,
    accelerator: Accelerator,
    loss_fn: nn.Module,
    epoch: int,
) -> float:
    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(model, validate_loader, accelerator, loss_fn)
    validation_loss = metrics["loss"]

    log_metrics(accelerator, metrics, "eval", epoch)
    del metrics
    torch.cuda.empty_cache()
    gc.collect()
    return validation_loss


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    validate_loader: DataLoader,
    scheduler,
    accelerator: Accelerator,
    loss_fn: nn.Module,
    epochs: int,
    logger: logging.Logger,
    log_dir: Path,
    current_iterates: int = 0,
    steps_so_far: int = 0,
) -> Path:
    best_validation_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        logger.info(
            f"Iterative Update {current_iterates} - Starting Epoch {epoch}/{epochs}"
        )

        avg_epoch_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            scheduler,
            accelerator,
            loss_fn,
            epoch,
            epochs,
            current_iterates,
            steps_so_far,
        )
        logger.info(
            "Iterative Update %s - Epoch %s - Average Training Loss: %.4f",
            current_iterates,
            epoch,
            avg_epoch_loss,
        )

        accelerator.wait_for_everyone()

        validation_loss = validate_epoch(
            model,
            validate_loader,
            accelerator,
            loss_fn,
            epoch + current_iterates * epochs,
        )
        logger.info(
            "Iterative Update %s - Epoch %s - Validation Loss: %.4f",
            current_iterates,
            epoch,
            validation_loss,
        )

        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss
            best_epoch = epoch
            accelerator.unwrap_model(model).save_pretrained(log_dir / "best_model")
            logger.info(
                "Iterative Update %s - Best model updated at epoch %s with Validation Loss: %.4f",
                current_iterates,
                epoch,
                best_validation_loss,
            )

        gc.collect()
        torch.cuda.empty_cache()

    accelerator.unwrap_model(model).save_pretrained(log_dir / "last_model")
    logger.info(
        "Iterative Update %s - Training complete. Best epoch: %s with Validation Loss: %.4f",
        current_iterates,
        best_epoch,
        best_validation_loss,
    )
    torch.cuda.empty_cache()
    gc.collect()

    return log_dir / "best_model"
