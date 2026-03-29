#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BM25 RE-RANKING (K=500) + STORE TOP-15 PER USER

For each user_id:
- Dense Retrieval (K=500)
- BM25 re-ranking
- Store Top-15 business_ids

Output:
- rag_seed2027_bm25_top15.csv
- Summary statistics printed
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
import csv

# ============================================================
# PATHS
# ============================================================

BASE = Path("/Users/adrianhoffmann/Documents/Master/Projektarbeit/AgenticGroundedRagClone")
DATA = BASE / "Dataset"

HYP_JSONL = DATA / "validation" / "hypotheses_seed2026_without_cluster.jsonl"
EMB_CSV   = DATA / "validation" / "hypotheses_seed2026_without_cluster_embeddings.csv"

POI_STATES = DATA / "core" / "poi_semantic_states.jsonl"
FAISS_INDEX = DATA / "representations" / "poi_text_index.faiss"
FAISS_META  = DATA / "representations" / "poi_text_meta.jsonl"

OUTPUT_CSV = DATA / "validation_rag" / "rag_seed2026_bm25_top15.csv"




# ============================================================
# CONFIG
# ============================================================

RAG_K = 500
TOP_EVAL = 15

# ============================================================
# HELPERS
# ============================================================

def load_poi_states(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("status") == "ok":
                out[int(obj["business_id"])] = obj["attributes"]
    return out

def load_meta(path):
    bids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            bids.append(int(json.loads(line)["business_id"]))
    return bids

def build_doc(attr):
    tokens = []
    for k in [
        "spatial_configuration",
        "style",
        "atmosphere",
        "target_group",
        "distinctive_features",
    ]:
        tokens.extend(attr.get(k, []))
    return [t.lower() for t in tokens if isinstance(t, str)]

# ============================================================
# LOAD
# ============================================================

print("Loading data...")

poi_states = load_poi_states(POI_STATES)
meta_bids = load_meta(FAISS_META)
index = faiss.read_index(str(FAISS_INDEX))

rows = []
with open(HYP_JSONL, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
df = pd.DataFrame(rows)

df_emb = pd.read_csv(EMB_CSV)
emb_map = {
    str(r["user_id"]): np.array(json.loads(r["embedding"]), dtype=np.float32)
    for _, r in df_emb.iterrows()
}

# ============================================================
# ANALYSIS + STORAGE
# ============================================================

total_cases = 0
hits_top15 = 0
rank_distribution = {r: 0 for r in range(1, TOP_EVAL + 1)}

rows_out = []

for _, r in df.iterrows():

    uid = str(r["user_id"])
    target = int(r["target"])

    if uid not in emb_map:
        continue

    q_emb = emb_map[uid]

    # Dense Retrieval
    _, I = index.search(q_emb.reshape(1, -1), RAG_K)
    candidates = [meta_bids[i] for i in I[0]]

    corpus_tokens, bids = [], []
    for bid in candidates:
        attr = poi_states.get(bid)
        if not attr:
            continue
        tokens = build_doc(attr)
        if tokens:
            corpus_tokens.append(tokens)
            bids.append(bid)

    hyp_tokens = build_doc(r["hypothesis"])
    if not hyp_tokens:
        continue

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(hyp_tokens)

    ranked = sorted(
        zip(bids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    total_cases += 1

    # Store Top-15
    for rank, (bid, score) in enumerate(ranked[:TOP_EVAL], start=1):

        is_target = int(bid == target)

        rows_out.append({
            "user_id": uid,
            "rerank_rank": rank,
            "business_id": bid,
            "bm25_score": float(score),
            "is_target": is_target
        })

        if is_target:
            hits_top15 += 1
            rank_distribution[rank] += 1

# ============================================================
# WRITE OUTPUT
# ============================================================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "user_id",
            "rerank_rank",
            "business_id",
            "bm25_score",
            "is_target"
        ]
    )
    writer.writeheader()
    writer.writerows(rows_out)

# ============================================================
# SUMMARY
# ============================================================

print("\n=== BM25 Re-Ranking @15 (K=500) ===\n")

print(f"Total eligible cases: {total_cases}")
print(f"Hits @15: {hits_top15}")
print(f"Hit-Rate @15: {round(hits_top15 / total_cases, 4) if total_cases else 0.0}")

print("\nRank Distribution (1–15):")
for r in range(1, TOP_EVAL + 1):
    print(f"Rank {r}: {rank_distribution[r]}")

print(f"\nSaved Top-15 per user to:")
print(OUTPUT_CSV)
