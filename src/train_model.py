# src/train_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool


def main() -> None:
    # === Load splits ===
    splits_dir = Path("../data/splits_MiniLM-L6-v2")

    train = np.load(splits_dir / "train.npz", allow_pickle=True)
    val   = np.load(splits_dir / "val.npz",   allow_pickle=True)
    test  = np.load(splits_dir / "test.npz",  allow_pickle=True)

    # Keep as object dtype to preserve categorical features
    X_train = train["X"]
    y_train = train["y"].astype(int)

    X_val   = val["X"]
    y_val   = val["y"].astype(int)

    X_test  = test["X"]
    y_test  = test["y"].astype(int)

    print("X_train dtype:", X_train.dtype, "shape:", X_train.shape)
    print("X_val   dtype:", X_val.dtype,   "shape:", X_val.shape)
    print("X_test  dtype:", X_test.dtype,  "shape:", X_test.shape)

    # === Load feature names ===
    feature_columns = np.load(
        splits_dir / "feature_columns.npy",
        allow_pickle=True
    ).tolist()

    print("Num features:", len(feature_columns))

    # === Identify categorical feature indices ===
    categorical_features = ["department", "country", "cat1", "subcat1", "cat2"]
    cat_indices = [i for i, col in enumerate(feature_columns) if col in categorical_features]
    print(f"Categorical features at indices: {cat_indices}")

    # === Build CatBoost Pools ===
    train_pool = Pool(
        X_train, y_train,
        feature_names=feature_columns,
        cat_features=cat_indices,
    )

    val_pool = Pool(
        X_val, y_val,
        feature_names=feature_columns,
        cat_features=cat_indices,
    )

    # === Define and train the model ===
    model = CatBoostClassifier(
        iterations=800,
        depth=8,
        learning_rate=0.08,
        l2_leaf_reg=5,
        border_count=64,
        random_strength=2,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        eval_metric="PRAUC",
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
