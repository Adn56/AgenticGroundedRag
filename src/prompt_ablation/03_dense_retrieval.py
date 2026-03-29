#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPORT TOP-500 RAG RETRIEVAL CANDIDATES

- Reads:
    hypotheses_seed2028_embeddings.csv
    hypotheses_seed2028.jsonl
- Uses:
    FAISS dense index
- Retrieves:
    Top-500 nearest POIs per hypothesis embedding
- Writes (OVERWRITE):
    rag_seed2028_top500.csv

Guarantees:
- deterministic FAISS retrieval
- no reranking
- one row per (user_id, retrieved_rank)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
import csv

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

EMB_CSV = PROJECT_ROOT  / "prompt_ablation" / "hypothesis_test_eval_1000_withTime_embeddings.csv"
HYP_JSONL = PROJECT_ROOT  / "prompt_ablation" / "hypothesis_test_eval_1000_withTime.jsonl"

FAISS_INDEX = DATASET_DIR  / "representations" / "poi_text_index.faiss"
FAISS_META  = DATASET_DIR  / "representations" / "poi_text_meta.jsonl"

OUTPUT_CSV = PROJECT_ROOT  / "prompt_ablation" / "rag_hypothesis_test_eval_1000_withTime_top50.csv"

# ============================================================
# CONFIG
# ============================================================

RAG_K = 200

# ============================================================
# LOAD META (business_id mapping)
# ============================================================

def load_meta(path):
    bids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            bids.append(int(json.loads(line)["business_id"]))
    return bids

print("Loading FAISS index...")
index = faiss.read_index(str(FAISS_INDEX))

print("Loading meta mapping...")
meta_bids = load_meta(FAISS_META)

print("Loading embeddings...")
df_emb = pd.read_csv(EMB_CSV)

emb_map = {
    str(r["user_id"]): np.array(json.loads(r["embedding"]), dtype=np.float32)
    for _, r in df_emb.iterrows()
}

print(f"Loaded {len(emb_map)} embeddings")

# ============================================================
# RETRIEVE
# ============================================================

rows_out = []

for i, (uid, emb) in enumerate(emb_map.items(), start=1):

    D, I = index.search(emb.reshape(1, -1), RAG_K)

    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):

        business_id = meta_bids[idx]

        rows_out.append({
            "user_id": uid,
            "retrieval_rank": rank,
            "business_id": business_id,
            "faiss_distance": float(dist)
        })

    if i % 10 == 0 or i == len(emb_map):
        print(f"Processed {i}/{len(emb_map)} hypotheses")

# ============================================================
# WRITE OUTPUT (OVERWRITE)
# ============================================================

print("\nWriting output...")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "user_id",
            "retrieval_rank",
            "business_id",
            "faiss_distance"
        ]
    )
    writer.writeheader()
    writer.writerows(rows_out)

print("Done.")
print(f"Wrote {len(rows_out)} rows to:")
print(OUTPUT_CSV)
