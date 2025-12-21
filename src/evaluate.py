# src/evaluate.py

import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)


def main() -> None:
    # === Load splits ===
    splits_dir = Path("../data/splits")

    val = np.load(splits_dir / "val.npz", allow_pickle=True)
    test = np.load(splits_dir / "test.npz", allow_pickle=True)

    X_val = val["X"].astype(np.float32)
    y_val = val["y"].astype(int)

    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(int)

    # === Load feature names ===
    feature_columns = np.load(
        splits_dir / "feature_columns.npy",
        allow_pickle=True
    ).tolist()

    # === Drop selected columns ===
    cols_to_remove = ["isTimelineWork", "isPublicDomain", "accessionYear"]

    # Get indices of columns we want to drop
    drop_indices = [feature_columns.index(col) for col in cols_to_remove]
    print("Dropping indices:", drop_indices)

    # Sort descending so deleting works without shifting indices
    drop_indices_sorted = sorted(drop_indices, reverse=True)

    for idx in drop_indices_sorted:
        X_val = np.delete(X_val, idx, axis=1)
        X_test = np.delete(X_test, idx, axis=1)

        # Remove from feature column names
        feature_columns.pop(idx)

    # === Load model ===
    model_path = Path("../models/catboost_model_optimized_parameters2.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # === Make predictions ===
    # Probabilities
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Convert to class predictions
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)

    # === Compute and print metrics ===
    print("\nVAL metrics:")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Precision:", precision_score(y_val, y_val_pred))
    print("Recall:", recall_score(y_val, y_val_pred))
    print("F1:", f1_score(y_val, y_val_pred))
    print("AUC:", roc_auc_score(y_val, y_val_pred_proba))

    print("\nTEST metrics:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall:", recall_score(y_test, y_test_pred))
    print("F1:", f1_score(y_test, y_test_pred))
    print("AUC:", roc_auc_score(y_test, y_test_pred_proba))


if __name__ == "__main__":
    main()

