# src/data_split.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main() -> None:
    # Base data directory (relative to this script, same as in the notebook)
    data_dir = Path("../data")

    # 1) Load embeddings and metadata
    embeddings_path = data_dir / "embeddings_text_all_mpnet.npy"
    meta_path = data_dir / "meta_for_model.parquet"

    embeddings = np.load(embeddings_path)
    df_meta = pd.read_parquet(meta_path)

    print("Embeddings:", embeddings.shape)
    print("Meta:", df_meta.shape)

    # 2) Build dataframe with embedding columns
    emb_dim = embeddings.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols)

    # Combine meta + embeddings
    df_features = pd.concat([df_meta, df_emb], axis=1)

    print("Combined features shape:", df_features.shape)
    print(df_features.head())

    # 3) Target vector
    y = df_features["label_isOnView"].astype(int).values

    # 4) Feature matrix with names
    feature_columns = df_features.drop(columns=["objectID", "label_isOnView"]).columns.tolist()
    X = df_features[feature_columns].values

    print("X:", X.shape)
    print("y:", y.shape)
    print("First 5 feature names:", feature_columns[:5])

    # 5) Train/val/test split via indices
    indices = np.arange(len(X))

    # Test split
    idx_temp, idx_test = train_test_split(
        indices,
        test_size=0.15,
        stratify=y,
        random_state=42,
    )

    # Train/val split
    idx_train, idx_val = train_test_split(
        idx_temp,
        test_size=0.15,
        stratify=y[idx_temp],
        random_state=42,
    )

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape,   y_val.shape)
    print("Test: ", X_test.shape,  y_test.shape)

    # 6) Save splits and feature names
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.savez(splits_dir / "train.npz", X=X_train, y=y_train)
    np.savez(splits_dir / "val.npz",   X=X_val,   y=y_val)
    np.savez(splits_dir / "test.npz",  X=X_test,  y=y_test)

    np.save(splits_dir / "feature_columns.npy", np.array(feature_columns))

    print(f"Saved splits and feature names to: {splits_dir}")


if __name__ == "__main__":
    main()
