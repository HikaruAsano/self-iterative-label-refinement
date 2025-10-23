"""Dataset Preprocessing Module.

This module provides functionality to download and process datasets from HuggingFace:
1. fake: Download hashed dataset and reconstruct using raw Kaggle data
2. safety: Download complete dataset from HuggingFace
3. saroco: Download complete dataset from HuggingFace
4. protein: Download hashed dataset and reconstruct using binding affinity data

Usage:
    python -m constraint_learning_rlhf.preprocess.format_dataset --dataset fake
    python -m constraint_learning_rlhf.preprocess.format_dataset --dataset safety
    python -m constraint_learning_rlhf.preprocess.format_dataset --dataset saroco
    python -m constraint_learning_rlhf.preprocess.format_dataset --dataset protein
    python -m constraint_learning_rlhf.preprocess.format_dataset --dataset all

Requirements for fake dataset:
    - Raw Kaggle data must be downloaded and placed at:
      - data/fake_news/Fake.csv
      - data/fake_news/True.csv
    - Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Author: Hikaru Asano
Affiliation: The University of Tokyo
"""

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

# Data directory relative to this file
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# HuggingFace dataset repositories
HUGGINGFACE_REPOS = {
    "fake": "HikaruAsano/SILR-fakenews-classification",
    "safety": "HikaruAsano/SILR-safety-classification",
    "saroco": "HikaruAsano/SILR-saroco-classification",
    "protein": "HikaruAsano/SILR-protein-classification",
}

# Base dataset repositories
BINDING_AFFINITY_REPO = "jglaser/binding_affinity"

# Format template for fake news dataset
FAKE_PROMPT = """# Title: {title}
# Content: {text}
"""


def compute_text_hash(text: str) -> Optional[str]:
    """Compute SHA256 hash of text-like input.
    
    Args:
        text: Text to hash
        
    Returns:
        SHA256 hash string, or None if text is None/NaN
    """
    if pd.isna(text):
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_text(title: str, text: str) -> str:
    """Apply the FAKE_PROMPT template to create formatted text.
    
    Args:
        title: Article title
        text: Article content
        
    Returns:
        Formatted text string
    """
    return FAKE_PROMPT.format(title=title, text=text)


def load_kaggle_data(data_dir: Optional[Path] = None) -> Dict[str, str]:
    """Load Kaggle Fake News dataset and create a hash -> text mapping.
    
    Args:
        data_dir: Path to data directory. Defaults to DATA_DIR.
        
    Returns:
        Dictionary mapping text_hash -> formatted_text
        
    Raises:
        FileNotFoundError: If Kaggle data files are not found
    """
    if data_dir is None:
        data_dir = DATA_DIR
        
    fake_path = data_dir / "fake_news" / "Fake.csv"
    true_path = data_dir / "fake_news" / "True.csv"
    
    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            f"Kaggle data not found. Please download from:\n"
            f"https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            f"and place Fake.csv and True.csv in {data_dir / 'fake_news'}/"
        )
    
    hash_to_text = {}
    
    # Load Fake.csv
    print("Loading Fake.csv...")
    df_fake = pd.read_csv(fake_path)
    print(f"  Loaded {len(df_fake)} rows")
    
    for _, row in df_fake.iterrows():
        formatted = format_text(row["title"], row["text"])
        text_hash = compute_text_hash(formatted)
        hash_to_text[text_hash] = formatted
    
    # Load True.csv
    print("Loading True.csv...")
    df_true = pd.read_csv(true_path)
    print(f"  Loaded {len(df_true)} rows")
    
    for _, row in df_true.iterrows():
        formatted = format_text(row["title"], row["text"])
        text_hash = compute_text_hash(formatted)
        hash_to_text[text_hash] = formatted
    
    print(f"Total unique formatted texts: {len(hash_to_text)}")
    return hash_to_text


def load_binding_affinity_data() -> Dict[str, str]:
    """Load binding affinity dataset and create a hash -> smiles mapping.
    
    Returns:
        Dictionary mapping text_hash -> smiles
    """
    print(f"Downloading base dataset from {BINDING_AFFINITY_REPO}...")
    dataset = load_dataset(BINDING_AFFINITY_REPO)
    
    if isinstance(dataset, DatasetDict):
        splits = dataset.items()
    else:
        splits = [("train", dataset)]
    
    hash_to_smiles: Dict[str, str] = {}
    total_rows = 0
    
    for split_name, split in splits:
        df = split.to_pandas()
        if "smiles" not in df.columns:
            raise ValueError(
                f"Column 'smiles' not found in {BINDING_AFFINITY_REPO} "
                f"split '{split_name}'. Available columns: {list(df.columns)}"
            )
        print(f"  Loaded {len(df)} rows from {split_name}")
        total_rows += len(df)
        
        for smiles in df["smiles"]:
            text_hash = compute_text_hash(smiles)
            if text_hash is not None:
                hash_to_smiles[text_hash] = smiles
    
    print(f"Total rows scanned: {total_rows}")
    print(f"Total unique smiles: {len(hash_to_smiles)}")
    return hash_to_smiles


def download_dataset(repo_id: str) -> DatasetDict:
    """Download dataset from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    print(f"Downloading dataset from {repo_id}...")
    dataset = load_dataset(repo_id)
    print(f"  Train: {len(dataset['train'])} rows")
    print(f"  Test: {len(dataset['test'])} rows")
    return dataset


def _select_hash_column(
    df: pd.DataFrame,
    hash_column: Optional[str],
) -> str:
    """Select hash column from the hashed dataset."""
    if hash_column is not None:
        if hash_column not in df.columns:
            raise ValueError(
                f"Hash column '{hash_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return hash_column
    
    if "text_hash" in df.columns:
        return "text_hash"
    if "smiles_hash" in df.columns:
        return "smiles_hash"
    
    candidates = [col for col in df.columns if "hash" in col]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(
            f"Multiple hash columns found: {candidates}. "
            f"Please specify hash_column."
        )
    
    raise ValueError(
        "No hash column found in the hashed dataset. "
        "Please specify hash_column."
    )


def reconstruct_dataset(
    hashed_dataset: Dataset,
    hash_to_text: Dict[str, str],
    hash_column: Optional[str] = None,
    text_column: str = "text",
) -> pd.DataFrame:
    """Reconstruct the original dataset by matching hashes with base data.
    
    Args:
        hashed_dataset: HuggingFace Dataset with hash column
        hash_to_text: Dictionary mapping hash -> original text
        hash_column: Hash column name (auto-detected if None)
        text_column: Output text column name

    Returns:
        DataFrame with reconstructed text column
    """
    df = hashed_dataset.to_pandas()
    hash_column = _select_hash_column(df, hash_column)
    
    # Reconstruct text from hash
    texts = []
    matched = 0
    unmatched = 0
    
    for text_hash in df[hash_column]:
        if pd.isna(text_hash):
            texts.append(None)
            unmatched += 1
        elif text_hash in hash_to_text:
            texts.append(hash_to_text[text_hash])
            matched += 1
        else:
            texts.append(None)
            unmatched += 1
    
    df[text_column] = texts
    
    # Reorder columns: text first, then label, then models (drop hash column)
    model_cols = [
        c for c in df.columns if c not in [hash_column, text_column, "label"]
    ]
    df_reconstructed = df[[text_column, "label"] + model_cols]
    
    print(f"  Matched: {matched}, Unmatched: {unmatched}")
    
    return df_reconstructed


def reconstruct_and_save(
    output_dir: Optional[Path] = None,
    repo_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download hashed dataset and reconstruct using Kaggle data.
    
    This is the main entry point for dataset reconstruction.
    
    Args:
        output_dir: Directory to save reconstructed CSVs. Defaults to DATA_DIR.
        repo_id: HuggingFace repository ID for hashed dataset. 
                 If None, uses HUGGINGFACE_REPOS["fake"]
        
    Returns:
        Tuple of (train_df, test_df) DataFrames
    """
    if output_dir is None:
        output_dir = DATA_DIR
    
    if repo_id is None:
        repo_id = HUGGINGFACE_REPOS["fake"]
    
    # Step 1: Load Kaggle data and create hash mapping
    hash_to_text = load_kaggle_data(output_dir)
    
    # Step 2: Download hashed dataset from HuggingFace
    hashed_dataset = download_dataset(repo_id)
    
    # Step 3: Reconstruct train dataset
    print("\nReconstructing fake_train...")
    train_df = reconstruct_dataset(hashed_dataset["train"], hash_to_text)
    train_output_path = output_dir / "fake_train.csv"
    train_df.to_csv(train_output_path, index=False)
    print(f"  Saved to {train_output_path}")
    
    # Step 4: Reconstruct test dataset
    print("\nReconstructing fake_test...")
    test_df = reconstruct_dataset(hashed_dataset["test"], hash_to_text)
    test_output_path = output_dir / "fake_test.csv"
    test_df.to_csv(test_output_path, index=False)
    print(f"  Saved to {test_output_path}")
    
    # Verification
    train_matched = train_df["text"].notna().sum()
    test_matched = test_df["text"].notna().sum()
    print(f"\n✅ Dataset reconstruction complete!")
    print(f"  Train: {train_matched}/{len(train_df)} texts matched")
    print(f"  Test: {test_matched}/{len(test_df)} texts matched")
    
    return train_df, test_df


def reconstruct_and_save_protein(
    output_dir: Optional[Path] = None,
    repo_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download hashed protein dataset and reconstruct using binding affinity data.
    
    Args:
        output_dir: Directory to save reconstructed CSVs. Defaults to DATA_DIR.
        repo_id: HuggingFace repository ID for hashed dataset.
                 If None, uses HUGGINGFACE_REPOS["protein"]
        
    Returns:
        Tuple of (train_df, test_df) DataFrames
    """
    if output_dir is None:
        output_dir = DATA_DIR
    
    if repo_id is None:
        repo_id = HUGGINGFACE_REPOS["protein"]
    
    # Step 1: Load base dataset and create hash mapping
    hash_to_text = load_binding_affinity_data()
    
    # Step 2: Download hashed dataset from HuggingFace
    hashed_dataset = download_dataset(repo_id)
    
    # Step 3: Reconstruct train dataset
    print("\nReconstructing protein_train...")
    train_df = reconstruct_dataset(hashed_dataset["train"], hash_to_text)
    train_output_path = output_dir / "protein_train.csv"
    train_df.to_csv(train_output_path, index=False)
    print(f"  Saved to {train_output_path}")
    
    # Step 4: Reconstruct test dataset
    print("\nReconstructing protein_test...")
    test_df = reconstruct_dataset(hashed_dataset["test"], hash_to_text)
    test_output_path = output_dir / "protein_test.csv"
    test_df.to_csv(test_output_path, index=False)
    print(f"  Saved to {test_output_path}")
    
    # Verification
    train_matched = train_df["text"].notna().sum()
    test_matched = test_df["text"].notna().sum()
    print(f"\n✅ Dataset reconstruction complete!")
    print(f"  Train: {train_matched}/{len(train_df)} texts matched")
    print(f"  Test: {test_matched}/{len(test_df)} texts matched")
    
    return train_df, test_df


def download_and_save_complete_dataset(
    dataset_name: str,
    output_dir: Optional[Path] = None,
    repo_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download complete dataset from HuggingFace and save as CSV.
    
    This function is used for datasets that are already complete on HuggingFace
    (safety, saroco) and don't require reconstruction.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'safety', 'saroco')
        output_dir: Directory to save CSVs. Defaults to DATA_DIR.
        repo_id: HuggingFace repository ID. If None, uses HUGGINGFACE_REPOS.
        
    Returns:
        Tuple of (train_df, test_df) DataFrames
    """
    if output_dir is None:
        output_dir = DATA_DIR
    
    if repo_id is None:
        if dataset_name not in HUGGINGFACE_REPOS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(HUGGINGFACE_REPOS.keys())}"
            )
        repo_id = HUGGINGFACE_REPOS[dataset_name]
    
    # Download dataset
    dataset = download_dataset(repo_id)
    
    # Convert to DataFrames
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    
    # Save to CSV
    train_output_path = output_dir / f"{dataset_name}_train.csv"
    test_output_path = output_dir / f"{dataset_name}_test.csv"
    
    train_df.to_csv(train_output_path, index=False)
    print(f"  Saved to {train_output_path}")
    
    test_df.to_csv(test_output_path, index=False)
    print(f"  Saved to {test_output_path}")
    
    print(f"\n✅ Dataset download complete!")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    
    return train_df, test_df


def process_dataset(
    dataset_name: str,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single dataset (fake, safety, saroco, or protein).
    
    Args:
        dataset_name: Name of the dataset ('fake', 'safety', 'saroco', or 'protein')
        output_dir: Directory to save CSVs. Defaults to DATA_DIR.
        
    Returns:
        Tuple of (train_df, test_df) DataFrames
    """
    if output_dir is None:
        output_dir = DATA_DIR
    
    if dataset_name == "fake":
        repo_id = HUGGINGFACE_REPOS["fake"]
        return reconstruct_and_save(output_dir, repo_id)
    elif dataset_name == "protein":
        repo_id = HUGGINGFACE_REPOS["protein"]
        return reconstruct_and_save_protein(output_dir, repo_id)
    elif dataset_name in ["safety", "saroco"]:
        return download_and_save_complete_dataset(dataset_name, output_dir)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: ['fake', 'safety', 'saroco', 'protein']"
        )


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess datasets from HuggingFace"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fake", "safety", "saroco", "protein", "all"],
        default="fake",
        help="Dataset to process (default: fake)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (default: data/)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR
    
    print("=" * 60)
    print("Dataset Preprocessing")
    print("=" * 60)
    print(f"\nData directory: {output_dir}")
    print()
    
    datasets_to_process = []
    if args.dataset == "all":
        datasets_to_process = ["fake", "safety", "saroco", "protein"]
    else:
        datasets_to_process = [args.dataset]
    
    results = {}
    
    for dataset_name in datasets_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing {dataset_name} dataset")
        print(f"{'=' * 60}")
        
        try:
            if dataset_name == "fake":
                repo_id = HUGGINGFACE_REPOS["fake"]
                train_df, test_df = reconstruct_and_save(output_dir, repo_id)
            elif dataset_name == "protein":
                repo_id = HUGGINGFACE_REPOS["protein"]
                train_df, test_df = reconstruct_and_save_protein(
                    output_dir, repo_id
                )
            else:
                train_df, test_df = download_and_save_complete_dataset(
                    dataset_name, output_dir
                )
            
            results[dataset_name] = {
                "train": len(train_df),
                "test": len(test_df),
                "status": "success",
            }
            print(f"\n✅ {dataset_name} dataset processed successfully!")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print(f"\nPlease download the required data first.")
            results[dataset_name] = {"status": "error", "message": str(e)}
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results[dataset_name] = {"status": "error", "message": str(e)}
            if args.dataset != "all":
                raise
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for dataset_name, result in results.items():
        if result["status"] == "success":
            print(
                f"  {dataset_name}: "
                f"train={result['train']}, test={result['test']} ✅"
            )
        else:
            print(f"  {dataset_name}: ❌ {result.get('message', 'Failed')}")
    
    # Return 0 if all succeeded, 1 if any failed
    all_succeeded = all(r["status"] == "success" for r in results.values())
    return 0 if all_succeeded else 1


if __name__ == "__main__":
    exit(main())
