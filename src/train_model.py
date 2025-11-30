# src/train_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool


def main() -> None:
    # === Load splits ===
    splits_dir = Path("../data/splits")

    train = np.load(splits_dir / "train.npz", allow_pickle=True)
    val   = np.load(splits_dir / "val.npz",   allow_pickle=True)
    test  = np.load(splits_dir / "test.npz",  allow_pickle=True)

    # Cast X to float32, y to int
    X_train = train["X"].astype(np.float32)
    y_train = train["y"].astype(int)

    X_val   = val["X"].astype(np.float32)
    y_val   = val["y"].astype(int)

    X_test  = test["X"].astype(np.float32)
    y_test  = test["y"].astype(int)

    print("X_train dtype:", X_train.dtype, "shape:", X_train.shape)
    print("X_val   dtype:", X_val.dtype,   "shape:", X_val.shape)
    print("X_test  dtype:", X_test.dtype,  "shape:", X_test.shape)

    # === Load feature names ===
    feature_columns = np.load(
        splits_dir / "feature_columns.npy",
        allow_pickle=True
    ).tolist()

    print("Num features (before dropping):", len(feature_columns))

    # === Drop selected columns ===
    cols_to_remove = [
        "isTimelineWork",
        "isPublicDomain",
        "objectEndDate",
        "objectBeginDate",
        "accessionYear",
    ]

    # Get indices of columns we want to drop
    drop_indices = [feature_columns.index(col) for col in cols_to_remove]
    print("Dropping indices:", drop_indices)

    # Sort descending so deleting works without shifting indices
    drop_indices_sorted = sorted(drop_indices, reverse=True)

    for idx in drop_indices_sorted:
        X_train = np.delete(X_train, idx, axis=1)
        X_val   = np.delete(X_val,   idx, axis=1)
        X_test  = np.delete(X_test,  idx, axis=1)

        # Remove from feature column names
        feature_columns.pop(idx)

    print("Num features (after dropping):", len(feature_columns))

    # === Build CatBoost Pools ===
    train_pool = Pool(
        X_train, y_train,
        feature_names=feature_columns,
    )

    val_pool = Pool(
        X_val, y_val,
        feature_names=feature_columns,
    )

    # === Define and train the model ===
    model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        verbose=100,
    )

    model.fit(train_pool, eval_set=val_pool)

    # === Feature importances ===
    importances = model.get_feature_importance()
    df_imp = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\nTop 20 features by importance:")
    print(df_imp.head(20))


if __name__ == "__main__":
    main()
