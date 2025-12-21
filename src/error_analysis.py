# src/error_analysis.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


def load_splits(splits_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load validation and test splits."""
    val = np.load(splits_dir / "val.npz", allow_pickle=True)
    test = np.load(splits_dir / "test.npz", allow_pickle=True)

    X_val = val["X"].astype(np.float32)
    y_val = val["y"].astype(int)

    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(int)

    feature_columns = np.load(
        splits_dir / "feature_columns.npy",
        allow_pickle=True
    ).tolist()

    return X_val, y_val, X_test, y_test, feature_columns


def load_model(model_path: Path) -> CatBoostClassifier:
    """Load the trained CatBoost model."""
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def apply_feature_drop(
    X: np.ndarray,
    feature_columns: List[str],
    cols_to_remove: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Remove specified columns from feature matrix and column list."""
    X_updated = X.copy()
    feature_columns_updated = feature_columns.copy()

    cols_to_drop = [col for col in cols_to_remove if col in feature_columns_updated]

    if cols_to_drop:
        drop_indices = [feature_columns_updated.index(col) for col in cols_to_drop]
        drop_indices_sorted = sorted(drop_indices, reverse=True)

        for idx in drop_indices_sorted:
            X_updated = np.delete(X_updated, idx, axis=1)
            feature_columns_updated.pop(idx)

        print(f"Dropped columns: {cols_to_drop}")
    else:
        print("No columns to drop")

    return X_updated, feature_columns_updated


def get_predictions(
    model: CatBoostClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """Get predictions and probabilities from model."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "proba": y_proba,
        "pred": y_pred,
        "true": y_true,
    }


def recreate_splits(
    metadata_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Recreate val/test splits using same logic as training."""
    y = metadata_df["label_isOnView"].astype(int).values
    indices = np.arange(len(metadata_df))

    # Test split
    idx_temp, idx_test = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Train/val split
    _, idx_val = train_test_split(
        idx_temp,
        test_size=val_size,
        stratify=y[idx_temp],
        random_state=random_state,
    )

    return idx_val, idx_test


def create_error_dataframe(
    metadata_df: pd.DataFrame,
    indices: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Create a dataframe linking predictions to metadata for error analysis."""
    split_metadata = metadata_df.iloc[indices].copy().reset_index(drop=True)

    split_metadata["pred_proba"] = predictions["proba"]
    split_metadata["pred"] = predictions["pred"]
    split_metadata["true"] = predictions["true"]

    split_metadata["is_correct"] = split_metadata["pred"] == split_metadata["true"]
    split_metadata["is_false_positive"] = (split_metadata["pred"] == 1) & (split_metadata["true"] == 0)
    split_metadata["is_false_negative"] = (split_metadata["pred"] == 0) & (split_metadata["true"] == 1)

    return split_metadata


def print_error_summary(df_errors: pd.DataFrame, split_name: str) -> None:
    """Print error summary statistics."""
    total = len(df_errors)
    errors = (~df_errors["is_correct"]).sum()
    fp = df_errors["is_false_positive"].sum()
    fn = df_errors["is_false_negative"].sum()

    print(f"\n{split_name.upper()} SET ERROR BREAKDOWN")
    print("=" * 60)
    print(f"Total samples:        {total:,}")
    print(f"Total errors:         {errors:,} ({errors/total*100:.2f}%)")
    print(f"  False Positives:    {fp:,} ({fp/total*100:.2f}%)")
    print(f"  False Negatives:    {fn:,} ({fn/total*100:.2f}%)")
    if errors > 0:
        print(f"\nError composition:")
        print(f"  FP % of errors:      {fp/errors*100:.2f}%")
        print(f"  FN % of errors:      {fn/errors*100:.2f}%")


def analyze_errors_by_category(
    df_errors: pd.DataFrame,
    category_col: str,
    min_samples: int = 100,
    top_n: int = 15
) -> pd.DataFrame:
    """Analyze error rates by a categorical feature."""
    if category_col not in df_errors.columns:
        return pd.DataFrame()

    category_counts = df_errors[category_col].value_counts()
    valid_categories = category_counts[category_counts >= min_samples].index

    df_filtered = df_errors[df_errors[category_col].isin(valid_categories)].copy()

    if len(df_filtered) == 0:
        return pd.DataFrame()

    error_stats = df_filtered.groupby(category_col).agg({
        "is_correct": ["count", "sum"],
        "is_false_positive": "sum",
        "is_false_negative": "sum",
    }).reset_index()

    error_stats.columns = [category_col, "total", "correct", "fp", "fn"]

    error_stats["error_rate"] = 1 - (error_stats["correct"] / error_stats["total"])
    error_stats["fp_rate"] = error_stats["fp"] / error_stats["total"]
    error_stats["fn_rate"] = error_stats["fn"] / error_stats["total"]

    return error_stats.sort_values("error_rate", ascending=False).head(top_n)


def main() -> None:
    # === Configuration ===
    data_dir = Path("../data")
    splits_dir = data_dir / "splits"
    models_dir = Path("../models")
    model_filename = "catboost_model_optimized_parameters2.cbm"
    metadata_file = data_dir / "meta_for_model.parquet"

    COLS_TO_REMOVE = ["isTimelineWork", "isPublicDomain", "accessionYear"]
    THRESHOLD = 0.5

    print("=" * 70)
    print("MODEL ERROR ANALYSIS")
    print("=" * 70)

    # === Load data ===
    print("\nLoading data...")
    X_val, y_val, X_test, y_test, feature_columns = load_splits(splits_dir)

    print(f"Loaded splits:")
    print(f"  Val:   {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples")

    # === Load model ===
    model_path = models_dir / model_filename
    model = load_model(model_path)
    print(f"\nModel loaded from: {model_path}")

    # === Apply feature dropping ===
    print("\nApplying feature preprocessing...")
    X_val_processed, feature_columns_processed = apply_feature_drop(
        X_val, feature_columns, COLS_TO_REMOVE
    )
    X_test_processed, _ = apply_feature_drop(X_test, feature_columns, COLS_TO_REMOVE)

    print(f"Processed features: {len(feature_columns_processed)}")

    # === Get predictions ===
    print("\nGenerating predictions...")
    val_preds = get_predictions(model, X_val_processed, y_val, THRESHOLD)
    test_preds = get_predictions(model, X_test_processed, y_test, THRESHOLD)

    print(f"  Val set - Mean probability: {val_preds['proba'].mean():.4f}")
    print(f"  Test set - Mean probability: {test_preds['proba'].mean():.4f}")

    # === Load metadata ===
    print("\nLoading metadata...")
    metadata_df = pd.read_parquet(metadata_file)
    print(f"Metadata shape: {metadata_df.shape}")

    # === Recreate splits to get indices ===
    idx_val, idx_test = recreate_splits(metadata_df)

    # === Create error dataframes ===
    print("\nCreating error analysis dataframes...")
    df_val_errors = create_error_dataframe(metadata_df, idx_val, val_preds)
    df_test_errors = create_error_dataframe(metadata_df, idx_test, test_preds)

    # === Print error summaries ===
    print_error_summary(df_val_errors, "validation")
    print_error_summary(df_test_errors, "test")

    # === Analyze by category (if full metadata available) ===
    try:
        full_metadata_file = data_dir / "df_train_clean.parquet"
        if full_metadata_file.exists():
            print("\n\nLoading full metadata for detailed analysis...")
            df_full_metadata = pd.read_parquet(full_metadata_file)

            # Merge with test errors
            df_test_errors = df_test_errors.merge(
                df_full_metadata[[col for col in df_full_metadata.columns 
                                 if col not in df_test_errors.columns or col == "objectID"]],
                on="objectID",
                how="left",
                suffixes=("", "_full")
            )

            # Analyze by department
            if "department" in df_test_errors.columns:
                print("\n" + "=" * 70)
                print("ERROR RATES BY DEPARTMENT (Top 15)")
                print("=" * 70)
                dept_errors = analyze_errors_by_category(
                    df_test_errors, "department", min_samples=50, top_n=15
                )
                if len(dept_errors) > 0:
                    print(dept_errors[["department", "total", "error_rate", 
                                      "fp_rate", "fn_rate"]].to_string(index=False))

    except Exception as e:
        print(f"\nCould not perform detailed category analysis: {e}")

    # === Confidence analysis ===
    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    correct_proba = df_test_errors[df_test_errors["is_correct"]]["pred_proba"]
    error_proba = df_test_errors[~df_test_errors["is_correct"]]["pred_proba"]
    fp_proba = df_test_errors[df_test_errors["is_false_positive"]]["pred_proba"]
    fn_proba = df_test_errors[df_test_errors["is_false_negative"]]["pred_proba"]

    print(f"Average confidence (correct predictions): {correct_proba.mean():.4f}")
    print(f"Average confidence (errors):              {error_proba.mean():.4f}")
    print(f"Average confidence (false positives):     {fp_proba.mean():.4f}")
    print(f"Average confidence (false negatives):     {fn_proba.mean():.4f}")

    # === Feature importance ===
    print("\n" + "=" * 70)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 70)

    feature_importance = model.get_feature_importance()
    df_feature_importance = pd.DataFrame({
        "feature": feature_columns_processed,
        "importance": feature_importance,
    }).sort_values("importance", ascending=False)

    print(df_feature_importance.head(20).to_string(index=False))

    print("\n" + "=" * 70)
    print("Error analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

