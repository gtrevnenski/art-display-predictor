"""
Comprehensive analysis of ALL input column impacts on model predictions.
Combines text column ablation and numeric permutation importance.

Metric: Precision-Recall AUC (Average Precision)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sentence_transformers import SentenceTransformer
import torch


CONFIG = {
    "model_path": "saved_models/catboost_model__MiniLM-L6-v2.cbm",
    "splits_dir": "data/splits_MiniLM-L6-v2",
    "metadata_file": "data/meta_for_model_MiniLM-L6-v2.parquet",
    "full_metadata_file": "data/df_train_clean.parquet",
    "text_columns": ["title", "medium", "classification", "objectName", "culture", "department", "creditLine"],
    "numeric_columns": ["objectEndDate"],
    "categorical_columns": ["department", "country", "cat1", "subcat1", "cat2", "has_country"],
    "sample_size": 1000,  # For speed
    "n_iterations": 3,  # Permutation iterations
}


def load_data_and_model():
    """Load model, test data, and metadata."""
    print("Loading model and data...")

    # Load model
    model = CatBoostClassifier()
    model.load_model(CONFIG["model_path"])

    # Load test data
    splits_dir = Path(CONFIG["splits_dir"])
    test = np.load(splits_dir / "test.npz", allow_pickle=True)
    # Keep as object dtype to preserve categorical features (strings)
    X_test = test["X"]
    y_test = test["y"].astype(int)

    # Load metadata BEFORE sampling
    metadata_df = pd.read_parquet(CONFIG["metadata_file"])
    full_metadata_df = pd.read_parquet(CONFIG["full_metadata_file"])

    # Sample for speed (keep indices to match metadata)
    if len(X_test) > CONFIG["sample_size"]:
        np.random.seed(42)
        indices = np.random.choice(len(X_test), CONFIG["sample_size"], replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
        sample_metadata = metadata_df.iloc[indices].reset_index(drop=True)
        sample_full_metadata = full_metadata_df.iloc[indices].reset_index(drop=True)
    else:
        sample_metadata = metadata_df.iloc[:len(X_test)].reset_index(drop=True)
        sample_full_metadata = full_metadata_df.iloc[:len(X_test)].reset_index(drop=True)

    # Feature columns
    feature_columns = np.load(
        splits_dir / "feature_columns.npy",
        allow_pickle=True
    ).tolist()

    print(f"✓ Loaded {len(X_test)} samples with {X_test.shape[1]} features")
    print(f"First 10 features: {feature_columns[:10]}")
    return model, X_test, y_test, feature_columns, sample_metadata, sample_full_metadata


def get_baseline_performance(model, X, y):
    """Calculate baseline PR AUC (Average Precision)."""
    y_proba = model.predict_proba(X)[:, 1]
    pr_auc = average_precision_score(y, y_proba)
    return pr_auc, y_proba


def analyze_text_columns(
    model, X_test, y_test, feature_columns, full_metadata, baseline_pr_auc
) -> Dict:
    """
    Analyze text columns using ablation:
    Remove each column from text_all, regenerate embeddings, measure PR AUC drop.
    """
    print("\n" + "=" * 70)
    print("ANALYZING TEXT COLUMNS (Ablation Method)")
    print("=" * 70)

    # Load embedding model (must match training model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print(f"Using device: {device}")

    print(f"Sample features: {feature_columns[:10]}")

    # Find embedding column indices (try multiple naming patterns)
    embedding_cols = [i for i, col in enumerate(feature_columns) if col.startswith("e5_")]
    if not embedding_cols:
        embedding_cols = [i for i, col in enumerate(feature_columns) if col.startswith("emb_")]
    if not embedding_cols:
        embedding_cols = [i for i, col in enumerate(feature_columns) if col.isdigit()]

    if not embedding_cols:
        print("⚠ No embedding features found, skipping text column analysis")
        print(f"   Available features: {feature_columns[:20]}")
        return {}

    embedding_start = min(embedding_cols)
    embedding_end = max(embedding_cols) + 1
    print(f"Embedding features: indices {embedding_start}-{embedding_end} ({len(embedding_cols)} dims)")

    results = {}

    for col in CONFIG["text_columns"]:
        if col not in full_metadata.columns:
            print(f"⚠ Column '{col}' not in metadata, skipping")
            continue

        print(f"\nTesting '{col}'...")

        text_columns_to_use = [
            c for c in CONFIG["text_columns"]
            if c != col and c in full_metadata.columns
        ]

        texts_without = full_metadata[text_columns_to_use].fillna("").agg(" ".join, axis=1).tolist()

        # Generate new embeddings
        new_embeddings = embedder.encode(texts_without, show_progress_bar=False)

        # Replace embeddings in X_test
        X_modified = X_test.copy()
        X_modified[:, embedding_start:embedding_end] = new_embeddings

        # Measure performance (PR AUC)
        y_proba = model.predict_proba(X_modified)[:, 1]
        modified_pr_auc = average_precision_score(y_test, y_proba)
        pr_auc_drop = baseline_pr_auc - modified_pr_auc

        results[col] = {
            "method": "ablation",
            "pr_auc_drop": float(pr_auc_drop),
            "pr_auc_without_column": float(modified_pr_auc),
            "percentage_impact": float(pr_auc_drop / baseline_pr_auc * 100) if baseline_pr_auc > 0 else float("nan"),
        }

        pct = (pr_auc_drop / baseline_pr_auc * 100) if baseline_pr_auc > 0 else float("nan")
        print(f"  PR AUC without: {modified_pr_auc:.4f} | Drop: {pr_auc_drop:+.4f} ({pct:+.2f}%)")

    return results


def analyze_categorical_columns(
    model, X_test, y_test, feature_columns, metadata, baseline_pr_auc
) -> Dict:
    """
    Analyze categorical columns using permutation importance:
    Shuffle column values, measure PR AUC drop.
    """
    print("\n" + "=" * 70)
    print("ANALYZING CATEGORICAL COLUMNS (Permutation Method)")
    print("=" * 70)

    results = {}

    for col in CONFIG["categorical_columns"]:
        if col not in metadata.columns:
            print(f"⚠ Column '{col}' not in metadata, skipping")
            continue

        if col not in feature_columns:
            print(f"⚠ Column '{col}' not in features, skipping")
            continue

        col_idx = feature_columns.index(col)
        print(f"\nTesting '{col}' (feature index: {col_idx})...")

        pr_auc_drops = []

        for _ in range(CONFIG["n_iterations"]):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, col_idx])

            y_proba = model.predict_proba(X_permuted)[:, 1]
            permuted_pr_auc = average_precision_score(y_test, y_proba)
            pr_auc_drop = baseline_pr_auc - permuted_pr_auc
            pr_auc_drops.append(pr_auc_drop)

        mean_drop = float(np.mean(pr_auc_drops))
        std_drop = float(np.std(pr_auc_drops))
        pct = (mean_drop / baseline_pr_auc * 100) if baseline_pr_auc > 0 else float("nan")

        results[col] = {
            "method": "permutation",
            "pr_auc_drop": mean_drop,
            "std": std_drop,
            "percentage_impact": pct,
        }

        print(f"  Mean PR AUC drop: {mean_drop:+.4f} ± {std_drop:.4f} ({pct:+.2f}%)")

    return results


def analyze_numeric_columns(
    model, X_test, y_test, feature_columns, metadata, baseline_pr_auc
) -> Dict:
    """
    Analyze numeric columns using permutation importance:
    Shuffle column values, measure PR AUC drop.
    """
    print("\n" + "=" * 70)
    print("ANALYZING NUMERIC COLUMNS (Permutation Method)")
    print("=" * 70)

    results = {}

    for col in CONFIG["numeric_columns"]:
        if col not in metadata.columns:
            print(f"⚠ Column '{col}' not in metadata, skipping")
            continue

        if col not in feature_columns:
            print(f"⚠ Column '{col}' not in features, skipping")
            continue

        col_idx = feature_columns.index(col)
        print(f"\nTesting '{col}' (feature index: {col_idx})...")

        pr_auc_drops = []

        for _ in range(CONFIG["n_iterations"]):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, col_idx])

            y_proba = model.predict_proba(X_permuted)[:, 1]
            permuted_pr_auc = average_precision_score(y_test, y_proba)
            pr_auc_drop = baseline_pr_auc - permuted_pr_auc
            pr_auc_drops.append(pr_auc_drop)

        mean_drop = float(np.mean(pr_auc_drops))
        std_drop = float(np.std(pr_auc_drops))
        pct = (mean_drop / baseline_pr_auc * 100) if baseline_pr_auc > 0 else float("nan")

        results[col] = {
            "method": "permutation",
            "pr_auc_drop": mean_drop,
            "std": std_drop,
            "percentage_impact": pct,
        }

        print(f"  Mean PR AUC drop: {mean_drop:+.4f} ± {std_drop:.4f} ({pct:+.2f}%)")

    return results


def main():
    """Main execution."""
    print("=" * 70)
    print("COMPREHENSIVE COLUMN IMPACT ANALYSIS")
    print("Metric: PR AUC (Average Precision)")
    print("=" * 70)

    model, X_test, y_test, feature_columns, metadata, full_metadata = load_data_and_model()

    baseline_pr_auc, _ = get_baseline_performance(model, X_test, y_test)
    print(f"\n✓ Baseline PR AUC: {baseline_pr_auc:.4f}")

    text_results = analyze_text_columns(
        model, X_test, y_test, feature_columns, full_metadata, baseline_pr_auc
    )

    categorical_results = analyze_categorical_columns(
        model, X_test, y_test, feature_columns, metadata, baseline_pr_auc
    )

    numeric_results = analyze_numeric_columns(
        model, X_test, y_test, feature_columns, metadata, baseline_pr_auc
    )

    all_results = {**text_results, **categorical_results, **numeric_results}

    # Sort by impact
    def impact_key(item):
        col, data = item
        return abs(data.get("pr_auc_drop", 0.0))

    sorted_results = dict(sorted(all_results.items(), key=impact_key, reverse=True))

    print("\n" + "=" * 70)
    print("SUMMARY: ALL INPUT COLUMNS (sorted by impact)")
    print("=" * 70)
    print(f"Baseline PR AUC: {baseline_pr_auc:.4f}")
    print(f"Sample size: {len(X_test)} test samples\n")

    for col, data in sorted_results.items():
        method_tag = "[Ablation]" if data["method"] == "ablation" else "[Permutation]"
        std_str = f" ± {data['std']:.4f}" if "std" in data else ""
        print(f"{col:20s} {method_tag:15s}: {data['pr_auc_drop']:+.4f}{std_str:13s} ({data['percentage_impact']:+.2f}%)")

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "description": "Impact of each input column on model predictions",
        "metric": "average_precision_score (PR AUC)",
        "baseline_pr_auc": float(baseline_pr_auc),
        "sample_size": len(X_test),
        "methods": {
            "ablation": "For text columns: remove from text_all, regenerate embeddings",
            "permutation": "For categorical and numeric: shuffle column values",
        },
        "columns": sorted_results,
    }

    output_file = output_dir / "all_columns_impact_pr_auc.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
