#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PIPELINE PERFORMANCE EVALUATION
Reasoning-First Agentic RAG

Metrics reported over ALL sequences:

- Hit@1
- Hit@5
- Hit@10
- Hit@15
- nDCG@5
- nDCG@10
- nDCG@15
- MRR
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_PATH = DATASET_DIR / "splits" / "test_eval_1000.jsonl"
RANK_PATH = RESULTS_DIR / "final_llm_ranking_test_eval_1000.csv"


# ============================================================
# LOAD TARGETS
# ============================================================

targets = {}

with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        targets[obj["user_id"]] = obj["target"]["business_id"]

print("Test sequences:", len(targets))


# ============================================================
# LOAD RANKINGS
# ============================================================

df = pd.read_csv(RANK_PATH)


# ============================================================
# EXTRACT TARGET RANKS
# ============================================================

rank_dict = {}

for user, group in df.groupby("user_id"):

    target = targets[user]

    row = group[group["business_id"] == target]

    if len(row) == 0:
        rank_dict[user] = None
    else:
        rank_dict[user] = int(row.iloc[0]["rank"])

N = len(rank_dict)

print("Users evaluated:", N)


# ============================================================
# HIT@K
# ============================================================

def hit_at_k(k):

    hits = 0

    for r in rank_dict.values():
        if r is not None and r <= k:
            hits += 1

    return hits / N


# ============================================================
# NDCG@K
# ============================================================

def ndcg_at_k(k):

    values = []

    for r in rank_dict.values():

        if r is None or r > k:
            values.append(0)

        else:
            values.append(1 / np.log2(r + 1))

    return np.mean(values)


# ============================================================
# MRR
# ============================================================

reciprocal_ranks = []

for r in rank_dict.values():

    if r is None:
        reciprocal_ranks.append(0)
    else:
        reciprocal_ranks.append(1 / r)

mrr = np.mean(reciprocal_ranks)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==============================================")
print("PIPELINE PERFORMANCE")
print("Reasoning-First Agentic-RAG")
print("==============================================")

print("\nHit@K")
print("---------------------------")

print(f"Hit@1  : {hit_at_k(1):.4f}")
print(f"Hit@5  : {hit_at_k(5):.4f}")
print(f"Hit@10 : {hit_at_k(10):.4f}")
print(f"Hit@20 : {hit_at_k(20):.4f}")


print("\nnDCG@K")
print("---------------------------")

print(f"nDCG@5  : {ndcg_at_k(5):.4f}")
print(f"nDCG@10 : {ndcg_at_k(10):.4f}")
print(f"nDCG@20 : {ndcg_at_k(20):.4f}")


print("\nMRR")
print("---------------------------")

print(f"MRR : {mrr:.4f}")

print("\nSequences evaluated:", N)