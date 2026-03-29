#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RANK DISTRIBUTION ANALYSIS

Evaluates where the true target POI appears within ranks 1-15 across
three pipeline stages:

1. Hypothesis / embedding retrieval
2. Distance reranking
3. LLM reranking

Additional metrics:
- Accuracy@1
- Accuracy@5
- Accuracy@10
- Accuracy@15
- Accuracy@20

Also computes:
- rank distributions (1-20 + MISS)
- rank-shift statistics between stages
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_PATH = DATASET_DIR / "splits" / "test_eval_1000.jsonl"

RAG_PATH = RESULTS_DIR / "rag_hypothesis_test_eval_1000_top100.csv"
BM25_PATH = RESULTS_DIR / "distance_reranked_top20.csv"
LLM_PATH = RESULTS_DIR / "final_llm_ranking_test_eval_1000.csv"

TOP_K = 20


# ============================================================
# LOAD TARGETS
# ============================================================

targets = {}

with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        user_id = obj["user_id"]
        target = obj["target"]["business_id"]
        targets[user_id] = target

print("Targets loaded:", len(targets))


# ============================================================
# LOAD HYPOTHESIS / RAG
# ============================================================

df_rag = pd.read_csv(RAG_PATH)

hyp_ranks = defaultdict(lambda: None)

for _, row in df_rag.iterrows():
    user = row["user_id"]
    bid = row["business_id"]
    rank = row["retrieval_rank"]

    if user in targets and bid == targets[user]:
        if hyp_ranks[user] is None or rank < hyp_ranks[user]:
            hyp_ranks[user] = rank


# ============================================================
# LOAD BM25
# ============================================================

df_bm25 = pd.read_csv(BM25_PATH)

df_bm25 = df_bm25.rename(columns={
    "rerank_rank": "rank"
})

bm25_ranks = defaultdict(lambda: None)

for _, row in df_bm25.iterrows():
    user = row["user_id"]
    bid = row["business_id"]
    rank = row["rank"]

    if user in targets and bid == targets[user]:
        if bm25_ranks[user] is None or rank < bm25_ranks[user]:
            bm25_ranks[user] = rank


# ============================================================
# LOAD LLM
# ============================================================

df_llm = pd.read_csv(LLM_PATH)

llm_ranks = defaultdict(lambda: None)

for _, row in df_llm.iterrows():
    user = row["user_id"]
    bid = row["business_id"]
    rank = row["rank"]

    if user in targets and bid == targets[user]:
        if llm_ranks[user] is None or rank < llm_ranks[user]:
            llm_ranks[user] = rank


# ============================================================
# COUNT DISTRIBUTION
# ============================================================

hyp_counts = {i: 0 for i in range(1, TOP_K + 1)}
bm25_counts = {i: 0 for i in range(1, TOP_K + 1)}
llm_counts = {i: 0 for i in range(1, TOP_K + 1)}

hyp_miss = 0
bm25_miss = 0
llm_miss = 0


# ============================================================
# RANK SHIFT STATS
# ============================================================

hyp_to_bm25_plus = 0
hyp_to_bm25_minus = 0
hyp_to_bm25_sum = 0

bm25_to_llm_plus = 0
bm25_to_llm_minus = 0
bm25_to_llm_sum = 0

hyp_bm25_shifts = []
bm25_llm_shifts = []


# ============================================================
# TOP-K ENTRY / EXIT TRACKING
# ============================================================

entered_hyp_to_bm25 = []
left_hyp_to_bm25 = []

entered_bm25_to_llm = []
left_bm25_to_llm = []


# ============================================================
# PROCESS USERS
# ============================================================

for user in targets:

    r_hyp = hyp_ranks[user]
    r_bm25 = bm25_ranks[user]
    r_llm = llm_ranks[user]

    # Hypothesis distribution
    if r_hyp is None or r_hyp > TOP_K:
        hyp_miss += 1
    else:
        hyp_counts[int(r_hyp)] += 1

    # BM25 distribution
    if r_bm25 is None or r_bm25 > TOP_K:
        bm25_miss += 1
    else:
        bm25_counts[int(r_bm25)] += 1

    # LLM distribution
    if r_llm is None or r_llm > TOP_K:
        llm_miss += 1
    else:
        llm_counts[int(r_llm)] += 1

    # Top-K entry/exit Hyp -> BM25
    hyp_topk = (r_hyp is not None and r_hyp <= TOP_K)
    bm25_topk = (r_bm25 is not None and r_bm25 <= TOP_K)

    if (not hyp_topk) and bm25_topk:
        entered_hyp_to_bm25.append((user, r_hyp, r_bm25))

    if hyp_topk and (not bm25_topk):
        left_hyp_to_bm25.append((user, r_hyp, r_bm25))

    # Top-K entry/exit BM25 -> LLM
    llm_topk = (r_llm is not None and r_llm <= TOP_K)

    if (not bm25_topk) and llm_topk:
        entered_bm25_to_llm.append((user, r_bm25, r_llm))

    if bm25_topk and (not llm_topk):
        left_bm25_to_llm.append((user, r_bm25, r_llm))

    # Rank shifts Hyp -> BM25
    if (
            r_hyp is not None and r_bm25 is not None and
            r_hyp <= TOP_K and r_bm25 <= TOP_K
    ):
        shift = r_hyp - r_bm25

        if shift != 0:
            hyp_bm25_shifts.append((shift, user, r_hyp, r_bm25))

        if shift > 0:
            hyp_to_bm25_plus += 1
        elif shift < 0:
            hyp_to_bm25_minus += 1

        hyp_to_bm25_sum += shift

    # Rank shifts BM25 -> LLM
    if (
            r_bm25 is not None and r_llm is not None and
            r_bm25 <= TOP_K and r_llm <= TOP_K
    ):
        shift = r_bm25 - r_llm

        if shift != 0:
            bm25_llm_shifts.append((shift, user, r_bm25, r_llm))

        if shift > 0:
            bm25_to_llm_plus += 1
        elif shift < 0:
            bm25_to_llm_minus += 1

        bm25_to_llm_sum += shift


# ============================================================
# PRINT RANK DISTRIBUTION TABLE
# ============================================================

print("\n==============================================")
print("TARGET RANK DISTRIBUTION")
print("==============================================")

print(f"{'Rank':<6}{'Hyp':<10}{'BM25':<10}{'LLM':<10}")

for i in range(1, TOP_K + 1):
    print(f"{i:<6}{hyp_counts[i]:<10}{bm25_counts[i]:<10}{llm_counts[i]:<10}")

print("----------------------------------------------")
print(f"{'MISS':<6}{hyp_miss:<10}{bm25_miss:<10}{llm_miss:<10}")
print("\nTotal sequences:", len(targets))


# ============================================================
# ACCURACY METRICS
# ============================================================

def compute_accuracy(counts, total):

    acc1 = counts[1] / total
    acc5 = sum(counts[i] for i in range(1,6)) / total
    acc10 = sum(counts[i] for i in range(1,11)) / total
    acc15 = sum(counts[i] for i in range(1,16)) / total
    acc20 = sum(counts[i] for i in range(1,21)) / total

    return acc1, acc5, acc10, acc15, acc20


total = len(targets)

hyp_acc = compute_accuracy(hyp_counts, total)
bm25_acc = compute_accuracy(bm25_counts, total)
llm_acc = compute_accuracy(llm_counts, total)

print("\n==============================================")
print("ACCURACY METRICS")
print("==============================================")

print(f"{'Stage':<10}{'Acc@1':<10}{'Acc@5':<10}{'Acc@10':<10}{'Acc@15':<10}{'Acc@20':<10}")
print(f"{'Hyp':<10}{hyp_acc[0]:<10.3f}{hyp_acc[1]:<10.3f}{hyp_acc[2]:<10.3f}{hyp_acc[3]:<10.3f}{hyp_acc[4]:<10.3f}")
print(f"{'Distance':<10}{bm25_acc[0]:<10.3f}{bm25_acc[1]:<10.3f}{bm25_acc[2]:<10.3f}{bm25_acc[3]:<10.3f}{bm25_acc[4]:<10.3f}")
print(f"{'LLM':<10}{llm_acc[0]:<10.3f}{llm_acc[1]:<10.3f}{llm_acc[2]:<10.3f}{llm_acc[3]:<10.3f}{llm_acc[4]:<10.3f}")