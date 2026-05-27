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
    splits_dir = Path("../data/splits_MiniLM-L6-v2")

    val = np.load(splits_dir / "val.npz", allow_pickle=True)
    test = np.load(splits_dir / "test.npz", allow_pickle=True)

    # Keep as object dtype to preserve categorical features
    X_val = val["X"]
    y_val = val["y"].astype(int)

    X_test = test["X"]
    y_test = test["y"].astype(int)

    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")

    # === Load model ===
    model_path = Path("../saved_models/catboost_model__MiniLM-L6-v2.cbm")
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

