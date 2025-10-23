"""Train Model

Author: Hikaru Asano
Affiliation: The University of Tokyo
"""

import argparse
import gc
import os
import pathlib
from datetime import datetime

import wandb

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object
from bitsandbytes.optim import PagedAdamW
from constraint_learning_rlhf.data import (
    compute_positive_prior_from_original,
    compute_theta_from_preprocessed,
    get_annotation_metrics_from_preprocessed,
    get_data_loaders,
    load_preprocessed_dataset,
    load_preprocessed_original_dataset,
    prepare_iterative_update_dataset,
)
from constraint_learning_rlhf.evaluation import evaluate_and_save_results
from constraint_learning_rlhf.losses import build_loss_fn
from constraint_learning_rlhf.modeling import (
    initialize_model_and_tokenizer,
    load_best_model,
)
from constraint_learning_rlhf.training import (
    initialize_logger,
    log_metrics,
    save_metrics,
    set_seed,
    train_model,
)
from transformers import (
    get_cosine_schedule_with_warmup,
)

HOME_DIR = pathlib.Path(__file__).parent.parent.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sequence classification model with LLM annotations."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generator.",
    )
    parser.add_argument(
        "--annotate_model_name",
        type=str,
        default="gpt2",
        help="Name of the annotate model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name of the train model.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="safety", help="Name of the dataset."
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Rank of the LoRA matrix.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Alpha parameter for the LoRA matrix.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout rate for the LoRA matrix.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--num_iterate", type=int, default=5, help="Number of iterative updates."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.03, help="Warmup ratio for scheduler."
    )
    parser.add_argument(
        "--loss_type", type=str, default="robust_uu", help="Type of loss function."
    )
    parser.add_argument(
        "--gamma", type=float, default=-0.001, help="Gamma parameter for loss function."
    )
    parser.add_argument(
        "--log_prefix",
        type=str,
        default="test",
        help="Prefix for logging.",
    )
    parser.add_argument(
        "--num_labeled_size",
        type=int,
        default=None,
        help="Number of labeled samples.",
    )
    parser.add_argument(
        "--use_debug",
        action="store_true",
        help="Use debug mode with limited dataset sizes (100 samples each).",
    )
    return parser.parse_args()


def setup_accelerator(
    log_dir: pathlib.Path, args: argparse.Namespace, current_timestamp: str
) -> Accelerator:
    # Check if the user is logged in
    if not wandb.api.api_key:
        # Prompt the user to log in
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs], log_with="wandb", project_dir=log_dir
    )
    name = f"{args.dataset_name}_{args.annotate_model_name}_{args.loss_type}_{args.num_labeled_size}_{args.lr:.1e}_seed{args.seed}_{current_timestamp}"
    accelerator.init_trackers(
        project_name=f"{args.log_prefix}",
        config=args,
        init_kwargs={"wandb": {"entity": os.getenv("WANDB_ENTITY"), "name": name}},
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.init(
            project=f"{args.log_prefix}",
            name=name,
            config=args,
            entity=os.getenv("WANDB_ENTITY"),
        )
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        wandb.define_metric("test/*", step_metric="test/step")
        wandb.define_metric("annotate/*", step_metric="annotate/step")
    return accelerator


def initialize_training_components(
    args: argparse.Namespace, accelerator: Accelerator, theta_1: float, theta_2: float
):
    model, tokenizer = initialize_model_and_tokenizer(
        args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    optimizer = PagedAdamW(model.parameters(), lr=args.lr)

    # Load preprocessed dataset (already formatted, no need for prompt.format)
    train_ds, validate_ds, test_ds = load_preprocessed_dataset(
        args.dataset_name,
        args.annotate_model_name,
    )

    # Limit dataset sizes in debug mode
    if args.use_debug:
        train_ds = train_ds.select(range(min(100, len(train_ds))))
        validate_ds = validate_ds.select(range(min(100, len(validate_ds))))
        test_ds = test_ds.select(range(min(100, len(test_ds))))

    train_loader = get_data_loaders(tokenizer, train_ds, args.batch_size)
    validate_loader = get_data_loaders(tokenizer, validate_ds, args.eval_batch_size)
    test_loader = get_data_loaders(tokenizer, test_ds, args.eval_batch_size)

    total_training_steps = args.num_epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    positive_prior = compute_positive_prior_from_original(args.dataset_name, "train")

    # Compute theta from preprocessed dataset (simplified)
    theta_1, theta_2 = compute_theta_from_preprocessed(
        args.dataset_name,
        args.annotate_model_name,
        args.seed,
        args.num_labeled_size,
    )

    loss_fn = build_loss_fn(
        args.loss_type,
        positive_prior,
        theta_1,
        theta_2,
        gamma=args.gamma,
    )
    (
        model,
        optimizer,
        train_loader,
        validate_loader,
        test_loader,
        scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_loader, validate_loader, test_loader, scheduler
    )
    return (
        model,
        tokenizer,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        validate_loader,
        test_loader,
    )


def evaluate_datasets(
    model,
    tokenizer,
    datasets,
    accelerator,
    log_dir,
    logger,
    current_iterates,
    eval_batch_size: int,
):
    metrics_dict = {}
    for dataset, name in datasets:
        temp_log_dir = log_dir / f"iterative_update_{current_iterates}"
        temp_log_dir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_and_save_results(
            model,
            tokenizer,
            dataset,
            accelerator,
            temp_log_dir / name,
            eval_batch_size=eval_batch_size,
        )
        log_name = "annotate" if name == "train" else "test"
        log_metrics(accelerator, metrics, log_name, current_iterates + 1)
        if accelerator.is_main_process:
            save_metrics(metrics, log_dir / f"{name}_eval.csv", current_iterates + 1)
            for metric_name, metric_value in metrics.items():
                logger.info(f"{metric_name}: {metric_value}")
        if name == "train":
            metrics_dict["theta_1"] = metrics["pred1_label1_rate"]
            metrics_dict["theta_2"] = metrics["pred0_label1_rate"]
    return metrics_dict


def reinitialize_components(
    args: argparse.Namespace,
    tokenizer,
    current_iterates: int,
    log_dir: pathlib.Path,
    theta_1: float,
    theta_2: float,
    accelerator: Accelerator,
):
    model, tokenizer = initialize_model_and_tokenizer(
        args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    # Load dataset from iterative update annotations
    train_ds, validate_ds = prepare_iterative_update_dataset(
        log_dir / f"iterative_update_{current_iterates}" / "train" / "annotation",
        validation_size=0.125,
    )

    # Limit dataset sizes in debug mode
    if args.use_debug:
        train_ds = train_ds.select(range(min(100, len(train_ds))))
        validate_ds = validate_ds.select(range(min(100, len(validate_ds))))

    train_loader = get_data_loaders(tokenizer, train_ds, args.batch_size)
    validate_loader = get_data_loaders(tokenizer, validate_ds, args.eval_batch_size)

    total_training_steps = args.num_epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_training_steps)
    optimizer = PagedAdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    positive_prior = compute_positive_prior_from_original(args.dataset_name, "train")

    # Use theta values passed from evaluate_datasets (computed from model predictions)
    loss_fn = build_loss_fn(
        args.loss_type,
        positive_prior,
        theta_1,
        theta_2,
        gamma=args.gamma,
    )

    model, train_loader, validate_loader, scheduler = accelerator.prepare(
        model, train_loader, validate_loader, scheduler
    )
    return (
        model,
        tokenizer,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        validate_loader,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.model_name == "Llama-2-7b-chat-hf":
        args.eval_batch_size = min(90, args.eval_batch_size)
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_timestamp = gather_object([current_timestamp])[0]
    logger, log_dir = initialize_logger(args, current_timestamp)

    # Get annotation metrics from preprocessed CSV
    annotation_metrics = get_annotation_metrics_from_preprocessed(
        args.dataset_name, args.annotate_model_name
    )

    accelerator = setup_accelerator(log_dir, args, current_timestamp)
    if accelerator.is_main_process:
        log_metrics(accelerator, annotation_metrics["train"], "annotate", 0)
        log_metrics(accelerator, annotation_metrics["test"], "test", 0)
        save_metrics(annotation_metrics["train"], log_dir / "train_eval.csv", 0)
        save_metrics(annotation_metrics["test"], log_dir / "test_eval.csv", 0)

    (
        model,
        tokenizer,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        validate_loader,
        test_loader,
    ) = initialize_training_components(
        args,
        accelerator,
        annotation_metrics["train"]["pred1_label1_rate"],
        annotation_metrics["train"]["pred0_label1_rate"],
    )

    steps_so_far = 0
    for current_iterates in range(args.num_iterate):
        best_model_path = train_model(
            model,
            optimizer,
            train_loader,
            validate_loader,
            scheduler,
            accelerator,
            loss_fn,
            args.num_epochs,
            logger,
            log_dir,
            current_iterates,
            steps_so_far,
        )
        model = load_best_model(args.model_name, best_model_path)
        model.config.pad_token_id = tokenizer.pad_token_id
        model = accelerator.prepare(model)
        steps_so_far += len(train_loader) * args.num_epochs
        # Free memory
        del train_loader, validate_loader, scheduler, loss_fn, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # Load preprocessed dataset for annotation (already formatted)
        train_ds = load_preprocessed_original_dataset(args.dataset_name, "train")
        test_ds = load_preprocessed_original_dataset(args.dataset_name, "test")

        # Limit dataset sizes in debug mode
        if args.use_debug:
            train_ds = train_ds.select(range(min(100, len(train_ds))))
            test_ds = test_ds.select(range(min(100, len(test_ds))))

        metrics = evaluate_datasets(
            model,
            tokenizer,
            [(train_ds, "train"), (test_ds, "test")],
            accelerator,
            log_dir,
            logger,
            current_iterates,
            eval_batch_size=args.eval_batch_size,
        )

        theta_1 = metrics["theta_1"]
        theta_2 = metrics["theta_2"]
        print(f"theta_1: {theta_1}, theta_2: {theta_2}")

        # Reinitialize components for next iteration
        (
            model,
            tokenizer,
            loss_fn,
            optimizer,
            scheduler,
            train_loader,
            validate_loader,
        ) = reinitialize_components(
            args,
            tokenizer,
            current_iterates,
            log_dir,
            theta_1,
            theta_2,
            accelerator,
        )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
