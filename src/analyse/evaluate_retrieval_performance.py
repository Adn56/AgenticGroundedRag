#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISS RETRIEVAL TARGET ANALYSIS

Computes full retrieval ranking using FAISS.

Outputs:
- Rank distribution (1–15)
- MISS
- Mean / Median / Min / Max rank
- Accuracy@K
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import statistics
import faiss

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

TEST_PATH = DATASET_DIR / "splits" / "test_eval_1000.jsonl"

EMB_CSV = PROJECT_ROOT / "results" / "hypothesis_test_eval_1000_embeddings.csv"

FAISS_INDEX = DATASET_DIR / "representations" / "poi_text_index.faiss"
FAISS_META  = DATASET_DIR / "representations" / "poi_text_meta.jsonl"


# ============================================================
# LOAD TARGETS
# ============================================================

targets = {}

with open(TEST_PATH, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        targets[str(obj["user_id"])] = int(obj["target"]["business_id"])

print("Targets loaded:", len(targets))


# ============================================================
# LOAD META
# ============================================================

def load_meta(path):

    bids = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            bids.append(int(json.loads(line)["business_id"]))

    return bids


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")
index = faiss.read_index(str(FAISS_INDEX))

# Jetzt erst kann man die Größe bestimmen
RETRIEVAL_K = index.ntotal

print("Total POIs in index:", RETRIEVAL_K)

print("Loading POI meta mapping...")
meta_bids = load_meta(FAISS_META)


# ============================================================
# LOAD HYPOTHESIS EMBEDDINGS
# ============================================================

print("Loading hypothesis embeddings...")

df_emb = pd.read_csv(EMB_CSV)

emb_map = {
    str(r["user_id"]): np.array(json.loads(r["embedding"]), dtype=np.float32)
    for _, r in df_emb.iterrows()
}

print("Embeddings loaded:", len(emb_map))


# ============================================================
# RETRIEVAL ANALYSIS
# ============================================================

target_ranks = {}

for i, (uid, emb) in enumerate(emb_map.items(), start=1):

    if uid not in targets:
        continue

    target = targets[uid]

    D, I = index.search(emb.reshape(1, -1), RETRIEVAL_K)

    found_rank = None

    for rank, idx in enumerate(I[0], start=1):

        bid = meta_bids[idx]

        if bid == target:
            found_rank = rank
            break

    if found_rank is not None:
        target_ranks[uid] = found_rank

    if i % 50 == 0 or i == len(emb_map):
        print(f"Processed {i}/{len(emb_map)}")


# ============================================================
# ANALYSIS
# ============================================================

rank_distribution = {i:0 for i in range(1,16)}

all_ranks = []
miss = 0

for uid in targets:

    if uid not in target_ranks:

        miss += 1
        all_ranks.append(RETRIEVAL_K)

        continue

    r = target_ranks[uid]

    all_ranks.append(r)

    if r <= 15:
        rank_distribution[r] += 1
    else:
        miss += 1


# ============================================================
# ACCURACY
# ============================================================

ks = [25,50,100,200,300,600,900]

accuracy = {}

for k in ks:

    hits = sum(1 for r in all_ranks if r <= k)

    accuracy[k] = hits / len(targets)


# ============================================================
# OUTPUT
# ============================================================

print("\n==============================================")
print("RAG TARGET RANK DISTRIBUTION")
print("==============================================")

print(f"{'Rank':<6}{'Count':<10}")

for r in range(1,16):
    print(f"{r:<6}{rank_distribution[r]:<10}")

print("----------------------------------------------")
print(f"{'MISS':<6}{miss:<10}")

print("\n==============================================")
print("RANK STATISTICS")
print("==============================================")

print("Mean rank:", round(sum(all_ranks)/len(all_ranks),3))
print("Median rank:", statistics.median(all_ranks))
print("Min rank:", min(all_ranks))
print("Max rank:", max(all_ranks))

print("\n==============================================")
print("ACCURACY @K")
print("==============================================")

for k in ks:
    print(f"Accuracy@{k}: {round(accuracy[k],4)}")