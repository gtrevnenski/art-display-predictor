# src/hyperparameter_optimization.py

import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, Pool


def main() -> None:
    # === Load splits ===
    splits_dir = Path("../data/splits")

    train = np.load(splits_dir / "train.npz", allow_pickle=True)
    val = np.load(splits_dir / "val.npz", allow_pickle=True)

    # Cast X to float32, y to int
    X_train = train["X"].astype(np.float32)
    y_train = train["y"].astype(int)

    X_val = val["X"].astype(np.float32)
    y_val = val["y"].astype(int)

    # === Load feature names ===
    feature_columns = np.load(
        splits_dir / "feature_columns.npy",
        allow_pickle=True
    ).tolist()

    # === Combine train + val for tuning ===
    X_tune = np.concatenate([X_train, X_val], axis=0)
    y_tune = np.concatenate([y_train, y_val], axis=0)

    print("Tuning set:", X_tune.shape, y_tune.shape)

    # === Subsample for faster tuning ===
    rng = np.random.default_rng(42)
    n_samples = min(20_000, X_tune.shape[0])  # cap to 20k for speed

    idx_sub = rng.choice(X_tune.shape[0], size=n_samples, replace=False)

    X_sub = X_tune[idx_sub]
    y_sub = y_tune[idx_sub]

    train_pool_sub = Pool(X_sub, y_sub, feature_names=feature_columns)
    print("Subsampled tuning set:", X_sub.shape, y_sub.shape)

    # === Define hyperparameter search space ===
    param_dist = {
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.08],
        "l2_leaf_reg": [1, 3, 5, 7, 10],
        "random_strength": [0.5, 1.0, 2.0],
        "border_count": [64, 128],
        "iterations": [500]
    }

    # === Run randomized search ===
    print("\nStarting randomized search with 3-fold CV...")
    print("This may take a while...\n")

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="CPU",
        bagging_temperature=1.0,
        verbose=False,
    )

    search_result = model.randomized_search(
        param_distributions=param_dist,
        X=train_pool_sub,
        cv=3,  # 3-fold CV on the subsample
        n_iter=25,  # number of random combinations to try
        partition_random_seed=42,
        search_by_train_test_split=False,  # use CV, not a single split
        calc_cv_statistics=True,
        refit=True,  # refit model on all X_sub, y_sub with best params
        shuffle=True,
    )

    # === Print results ===
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 70)
    print("\nBest parameters from search:")
    for param, value in search_result["params"].items():
        print(f"  {param}: {value}")

    best_auc = max(search_result["cv_results"]["test-AUC-mean"])
    print(f"\nBest CV AUC: {best_auc:.6f}")
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()

