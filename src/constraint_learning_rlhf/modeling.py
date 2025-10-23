"""Model initialization helpers.

Author: Hikaru Asano
Affiliation: The University of Tokyo
"""

import os
from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def initialize_model_and_tokenizer(
    model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float
) -> Tuple:
    """
    Initialize model and tokenizer with LoRA configuration.

    Args:
        model_name: Name of the pretrained model.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout rate.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_configuration = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        quantization_config=quantization_configuration,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    lora_configuration = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_configuration)
    return model, tokenizer


def load_best_model(model_name: str, best_model_path: str):
    """
    Load the best model from checkpoint.

    Args:
        model_name: Name of the base model.
        best_model_path: Path to the best model checkpoint.

    Returns:
        Loaded model with LoRA weights.
    """
    quantization_configuration = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        quantization_config=quantization_configuration,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )
    model = PeftModel.from_pretrained(model, best_model_path)
    return model
