#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BM25 RE-RANKING USING STORED RAG TOP-500

Pipeline per user:
1) Load Top-500 business_ids from rag CSV
2) Fetch semantic attributes from poi_semantic_states.jsonl
3) Tokenize hypothesis
4) Run BM25 over the 500 candidates
5) Store Top-15
"""

import json
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi
import csv

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

HYP_JSONL = PROJECT_ROOT / "results" / "hypothesis_test_eval_1000.jsonl"
RAG_CSV   = PROJECT_ROOT / "results" / "rag_hypothesis_test_eval_1000_top100.csv"
POI_STATES = DATASET_DIR / "core" / "poi_semantic_states.jsonl"

OUTPUT_CSV = PROJECT_ROOT / "results" / "rag_hypothesis_test_eval_1000_bm25_top15.csv"

TOP_EVAL = 15

# ============================================================
# LOAD POI STATES INTO MEMORY
# ============================================================

print("Loading POI semantic states...")

poi_states = {}
with open(POI_STATES, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("status") == "ok":
            poi_states[int(obj["business_id"])] = obj["attributes"]

print(f"Loaded {len(poi_states)} POI states")

# ============================================================
# LOAD RAG RETRIEVAL OUTPUT
# ============================================================

print("Loading RAG Top-500...")
df_rag = pd.read_csv(RAG_CSV)

# Gruppierung → user_id → Liste der business_ids
rag_grouped = (
    df_rag
    .sort_values(["user_id", "retrieval_rank"])
    .groupby("user_id")["business_id"]
    .apply(list)
    .to_dict()
)

print(f"Loaded RAG candidates for {len(rag_grouped)} users")

# ============================================================
# LOAD HYPOTHESES
# ============================================================

print("Loading hypotheses...")

hypotheses = {}
targets = {}

with open(HYP_JSONL, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            uid = str(obj["user_id"])
            hypotheses[uid] = obj["hypothesis"]
            targets[uid] = int(obj["target"])

print(f"Loaded {len(hypotheses)} hypotheses")

# ============================================================
# TOKEN HELPER
# ============================================================

def attributes_to_tokens(attr_dict):
    tokens = []
    for key in [
        "spatial_configuration",
        "style",
        "atmosphere",
        "target_group",
        "distinctive_features",
    ]:
        tokens.extend(attr_dict.get(key, []))
    return [t.lower() for t in tokens if isinstance(t, str)]

# ============================================================
# BM25 RE-RANKING
# ============================================================

print("Running BM25 re-ranking...")

rows_out = []
total_cases = 0
hits_top15 = 0
rank_distribution = {r: 0 for r in range(1, TOP_EVAL + 1)}

for uid, candidate_ids in rag_grouped.items():

    uid_str = str(uid)

    if uid_str not in hypotheses:
        continue

    if uid_str not in targets:
        continue

    hyp_attr = hypotheses[uid_str]
    target = targets[uid_str]

    hyp_tokens = attributes_to_tokens(hyp_attr)

    if not hyp_tokens:
        continue

    corpus_tokens = []
    valid_bids = []

    # --- STEP 1: fetch attributes for the 500 candidates ---
    for bid in candidate_ids:

        bid_int = int(bid)

        if bid_int not in poi_states:
            continue

        attr = poi_states[bid_int]
        tokens = attributes_to_tokens(attr)

        if tokens:
            corpus_tokens.append(tokens)
            valid_bids.append(bid_int)

    if not corpus_tokens:
        continue

    # --- STEP 2: BM25 ---
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(hyp_tokens)

    ranked = sorted(
        zip(valid_bids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    total_cases += 1

    # --- STEP 3: Store Top-15 ---
    for rank, (bid, score) in enumerate(ranked[:TOP_EVAL], start=1):

        is_target = int(bid == target)

        rows_out.append({
            "user_id": uid_str,
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

print("Writing output...")

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

print("\n=== BM25 Re-Ranking @15 (from stored RAG Top-500) ===\n")
print(f"Total eligible cases: {total_cases}")
print(f"Hits @15: {hits_top15}")
print(f"Hit-Rate @15: {round(hits_top15 / total_cases, 4) if total_cases else 0.0}")

print("\nRank Distribution (1–15):")
for r in range(1, TOP_EVAL + 1):
    print(f"Rank {r}: {rank_distribution[r]}")

print(f"\nSaved to: {OUTPUT_CSV}")