#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
04_facility_location_greedy_k.py
============================================================

Greedy Facility-Location selection of k representative images
per POI using cosine similarity (CLIP embeddings).

- Robust against NaN / Inf / degenerate embeddings
- Logs invalid images (name + POI + emb_index)
- Paper-ready implementation

Input:
- L2-normalized CLIP embeddings (memmap)
- embeddings_index.csv with business_id → emb_index mapping

Output:
- CSV with exactly k representative images per POI
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
EMB_DIM = 512
K = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMB_PATH = PROJECT_ROOT / "Dataset/representations/clip_embeddings_analyse/embeddings_fp16_l2.memmap"
INDEX_CSV = PROJECT_ROOT / "Dataset/representations/clip_embeddings_analyse/embeddings_index.csv"

OUT_CSV = PROJECT_ROOT / "Dataset/representations/poi_representative_images_k5.csv"
INVALID_LOG = PROJECT_ROOT / "Dataset/representations/clip_embeddings_analyse/invalid_embeddings.log"


# ============================================================
def greedy_facility_location(S: np.ndarray, k: int):
    """
    Greedy maximization of the facility-location objective.

    S: similarity matrix (n x n)
    k: number of representatives

    Returns indices of selected items (size k).
    """
    n = S.shape[0]
    selected = []
    current_max = np.zeros(n, dtype=np.float32)

    for _ in range(min(k, n)):
        gains = np.maximum(0.0, S - current_max[None, :]).sum(axis=1)
        best = int(np.argmax(gains))
        selected.append(best)
        current_max = np.maximum(current_max, S[best])

    return selected


# ============================================================
def main():
    t0 = time.time()

    if not EMB_PATH.exists():
        raise FileNotFoundError(EMB_PATH)
    if not INDEX_CSV.exists():
        raise FileNotFoundError(INDEX_CSV)

    index_df = pd.read_csv(INDEX_CSV)

    required = {"emb_index", "business_id", "local_image_path"}
    if not required.issubset(index_df.columns):
        raise ValueError(f"Index CSV must contain columns: {required}")

    # Group images by POI
    grouped = index_df.groupby("business_id")

    # Load embeddings (memmap)
    n_total = index_df["emb_index"].max() + 1
    X_all = np.memmap(
        EMB_PATH,
        dtype=np.float16,
        mode="r",
        shape=(n_total, EMB_DIM),
    )

    results = []

    # prepare invalid log
    INVALID_LOG.parent.mkdir(parents=True, exist_ok=True)
    invalid_logged = False

    for poi_id, g in tqdm(grouped, desc="Facility-Location per POI"):
        emb_indices = g["emb_index"].to_numpy()
        paths = g["local_image_path"].to_numpy()

        if len(emb_indices) == 0:
            continue

        # Load POI embeddings (fp32 for numerical stability)
        X = X_all[emb_indices].astype(np.float32)

        # ----------------------------------------------------
        # Detect and log invalid embeddings
        # ----------------------------------------------------
        mask = np.isfinite(X).all(axis=1)

        if not np.all(mask):
            bad_rows = np.where(~mask)[0]

            with open(INVALID_LOG, "a", encoding="utf-8") as f:
                for bi in bad_rows:
                    f.write(
                        f"{poi_id},{int(emb_indices[bi])},{paths[bi]}\n"
                    )

            # Filter invalid embeddings
            X = X[mask]
            emb_indices = emb_indices[mask]
            paths = paths[mask]

            invalid_logged = True

        if X.shape[0] == 0:
            continue

        # ----------------------------------------------------
        # Similarity matrix (Cosine = dot product)
        # ----------------------------------------------------
        S = X @ X.T  # shape (n, n)

        selected_idx = greedy_facility_location(S, K)

        for rank, idx in enumerate(selected_idx):
            results.append({
                "business_id": poi_id,
                "rank": rank + 1,
                "emb_index": int(emb_indices[idx]),
                "local_image_path": paths[idx],
            })

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)

    runtime = time.time() - t0

    print("\n✔ Facility-location selection finished")
    print(f"✔ Output CSV : {OUT_CSV}")
    print(f"✔ k          : {K}")
    print(f"✔ Runtime    : {runtime:.2f} seconds")

    if invalid_logged:
        print(f"⚠️ Invalid embeddings logged in: {INVALID_LOG}")
    else:
        print("✔ No invalid embeddings detected")


if __name__ == "__main__":
    main()
