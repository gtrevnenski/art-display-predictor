# src/make_embeddings.py

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


def main() -> None:
    # === Load cleaned training dataset ===
    df_path = Path("../data/df_train_clean.parquet")
    df_train = pd.read_parquet(df_path)
    print("Loaded training dataframe:", df_train.shape)
    print(df_train.head())

    # === Initialize embedding model on GPU/CPU ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = SentenceTransformer("all-mpnet-base-v2", device=device)

    # === Extract text for embeddings ===
    texts = df_train["text_all"].tolist()

    # === Encode into embeddings ===
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Embeddings shape:", embeddings.shape)

    # === Output directory ===
    out_dir = Path("../data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1) Save embeddings only ===
    emb_out_path = out_dir / "embeddings_text_all_mpnet.npy"
    np.save(emb_out_path, embeddings)

    # === 2) Save metadata needed for training ===
    NUMERIC = ["objectBeginDate", "objectEndDate", "accessionYear"]
    BOOLEAN = ["isTimelineWork", "isPublicDomain"]
    meta_cols = ["objectID", "label_isOnView"] + NUMERIC + BOOLEAN

    df_meta = df_train[meta_cols].reset_index(drop=True)

    meta_out_path = out_dir / "meta_for_model.parquet"
    df_meta.to_parquet(meta_out_path, index=False)

    print("Saved:")
    print(" - Embeddings →", emb_out_path)
    print(" - Metadata   →", meta_out_path)


if __name__ == "__main__":
    main()
