#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RANK DISTRIBUTION ANALYSIS

Evaluates where the true target POI appears within ranks 1-15 across
two pipeline stages:

1. Hypothesis / embedding retrieval (RAG)
2. BM25 reranking

Runs analysis for:
- CLUSTER pipeline
- TIME pipeline
- REFERENCE pipeline

Metrics:
- Recall@1 / @5 / @10 / @15 (before and after reranking)
- MRR
- nDCG@5 / @10 / @15
"""

import json
import pandas as pd
import math
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "prompt_ablation"

TEST_PATH = DATASET_DIR / "splits" / "test_eval_1000.jsonl"

TOP_K = 15


PIPELINES = {

    "CLUSTER": {
        "RAG": RESULTS_DIR / "rag_hypothesis_with_predicted_cluster_top100.csv",
        "BM25": RESULTS_DIR / "rag_hypothesis_test_eval_1000_bm25_top15.csv"
    },

    "TIME": {
        "RAG": RESULTS_DIR / "rag_hypothesis_test_eval_1000_withTime_top50.csv",
        "BM25": RESULTS_DIR / "rag_hypothesis_test_eval_1000_withTime_top50_bm25_top15.csv"
    },

    "REFERENCE": {
        "RAG": PROJECT_ROOT / "results" / "rag_hypothesis_test_eval_1000_top100.csv",
        "BM25": PROJECT_ROOT / "results" / "rag_hypothesis_test_eval_1000_bm25_top15.csv"
    }
}


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
# METRIC FUNCTIONS
# ============================================================

def reciprocal_rank(rank):

    if rank is None:
        return 0

    return 1 / rank


def ndcg(rank, k):

    if rank is None or rank > k:
        return 0

    return 1 / math.log2(rank + 1)


# ============================================================
# PIPELINE ANALYSIS FUNCTION
# ============================================================

def run_pipeline(name, rag_path, bm25_path):

    print("\n\n==============================================================")
    print("PIPELINE:", name)
    print("==============================================================")

    # ============================================================
    # LOAD RAG
    # ============================================================

    df_rag = pd.read_csv(rag_path)

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

    df_bm25 = pd.read_csv(bm25_path)

    bm25_ranks = defaultdict(lambda: None)

    for _, row in df_bm25.iterrows():

        user = row["user_id"]
        rank = row["rerank_rank"]
        is_target = row["is_target"]

        if user not in targets:
            continue

        if is_target == 1:

            if bm25_ranks[user] is None or rank < bm25_ranks[user]:
                bm25_ranks[user] = rank


    # ============================================================
    # COUNT DISTRIBUTION
    # ============================================================

    hyp_counts = {i: 0 for i in range(1, TOP_K + 1)}
    bm25_counts = {i: 0 for i in range(1, TOP_K + 1)}

    hyp_miss = 0
    bm25_miss = 0


    # ============================================================
    # METRICS
    # ============================================================

    recall1_hyp = 0
    recall5_hyp = 0
    recall10_hyp = 0
    recall15_hyp = 0

    recall1_bm25 = 0
    recall5_bm25 = 0
    recall10_bm25 = 0
    recall15_bm25 = 0

    mrr_sum = 0
    ndcg5_sum = 0
    ndcg10_sum = 0
    ndcg15_sum = 0


    # ============================================================
    # RANK SHIFT STATS
    # ============================================================

    hyp_to_bm25_plus = 0
    hyp_to_bm25_minus = 0
    hyp_to_bm25_sum = 0

    hyp_bm25_shifts = []


    # ============================================================
    # ENTRY / EXIT
    # ============================================================

    entered_hyp_to_bm25 = []
    left_hyp_to_bm25 = []


    # ============================================================
    # PROCESS USERS
    # ============================================================

    for user in targets:

        r_hyp = hyp_ranks[user]
        r_bm25 = bm25_ranks[user]


        # ============================================================
        # RETRIEVAL RECALL
        # ============================================================

        if r_hyp is not None:

            if r_hyp == 1:
                recall1_hyp += 1

            if r_hyp <= 5:
                recall5_hyp += 1

            if r_hyp <= 10:
                recall10_hyp += 1

            if r_hyp <= 15:
                recall15_hyp += 1


        # Hypothesis distribution
        if r_hyp is None or r_hyp > TOP_K:
            hyp_miss += 1
        else:
            hyp_counts[int(r_hyp)] += 1


        # ============================================================
        # BM25 METRICS
        # ============================================================

        if r_bm25 is not None:

            if r_bm25 == 1:
                recall1_bm25 += 1

            if r_bm25 <= 5:
                recall5_bm25 += 1

            if r_bm25 <= 10:
                recall10_bm25 += 1

            if r_bm25 <= 15:
                recall15_bm25 += 1

            mrr_sum += reciprocal_rank(r_bm25)

            ndcg5_sum += ndcg(r_bm25, 5)
            ndcg10_sum += ndcg(r_bm25, 10)
            ndcg15_sum += ndcg(r_bm25, 15)


        # BM25 distribution
        if r_bm25 is None or r_bm25 > TOP_K:
            bm25_miss += 1
        else:
            bm25_counts[int(r_bm25)] += 1


        # ============================================================
        # TOP-K ENTRY / EXIT
        # ============================================================

        hyp_topk = (r_hyp is not None and r_hyp <= TOP_K)
        bm25_topk = (r_bm25 is not None and r_bm25 <= TOP_K)

        if (not hyp_topk) and bm25_topk:
            entered_hyp_to_bm25.append((user, r_hyp, r_bm25))

        if hyp_topk and (not bm25_topk):
            left_hyp_to_bm25.append((user, r_hyp, r_bm25))


        # ============================================================
        # RANK SHIFTS
        # ============================================================

        if (
                r_hyp is not None and
                r_bm25 is not None and
                r_hyp <= TOP_K and
                r_bm25 <= TOP_K
        ):

            shift = r_hyp - r_bm25

            if shift != 0:
                hyp_bm25_shifts.append((shift, user, r_hyp, r_bm25))

            if shift > 0:
                hyp_to_bm25_plus += 1
            elif shift < 0:
                hyp_to_bm25_minus += 1

            hyp_to_bm25_sum += shift


    total = len(targets)


    # ============================================================
    # PRINT DISTRIBUTION
    # ============================================================

    print("\nTARGET RANK DISTRIBUTION")
    print("----------------------------------------------")
    print(f"{'Rank':<6}{'Hyp':<10}{'BM25':<10}")

    for i in range(1, TOP_K + 1):
        print(f"{i:<6}{hyp_counts[i]:<10}{bm25_counts[i]:<10}")

    print("----------------------------------------------")
    print(f"{'MISS':<6}{hyp_miss:<10}{bm25_miss:<10}")


    # ============================================================
    # RECALL METRICS
    # ============================================================

    print("\nRECALL METRICS")
    print("----------------------------------------------")

    print("Stage        Recall@1   Recall@5   Recall@10   Recall@15")
    print("--------------------------------------------------------")

    print(f"Retrieval    {recall1_hyp/total:.4f}     {recall5_hyp/total:.4f}     {recall10_hyp/total:.4f}     {recall15_hyp/total:.4f}")
    print(f"BM25         {recall1_bm25/total:.4f}     {recall5_bm25/total:.4f}     {recall10_bm25/total:.4f}     {recall15_bm25/total:.4f}")

    print()

    print("MRR     :", round(mrr_sum / total, 4))
    print("nDCG@5  :", round(ndcg5_sum / total, 4))
    print("nDCG@10 :", round(ndcg10_sum / total, 4))
    print("nDCG@15 :", round(ndcg15_sum / total, 4))


    # ============================================================
    # SHIFT ANALYSIS
    # ============================================================

    print("\nRANK SHIFT ANALYSIS")
    print("----------------------------------------------")

    print("Improved (+):", hyp_to_bm25_plus)
    print("Worse (-):", hyp_to_bm25_minus)
    print("Sum shift:", hyp_to_bm25_sum)


    # ============================================================
    # ENTRY EXIT
    # ============================================================

    print("\nTOP-15 ENTRY / EXIT")
    print("----------------------------------------------")

    print("Entered Top-15:", len(entered_hyp_to_bm25))
    print("Left Top-15:", len(left_hyp_to_bm25))


    # ============================================================
    # SORTED SHIFTS
    # ============================================================

    print("\nSORTED SHIFTS HYP -> BM25")
    print("----------------------------------------------")

    for shift, user, r1, r2 in sorted(hyp_bm25_shifts, key=lambda x: x[0], reverse=True):
        print(f"user {user} | {r1} -> {r2} | shift {shift}")


# ============================================================
# RUN PIPELINES
# ============================================================

for name, paths in PIPELINES.items():

    run_pipeline(
        name,
        paths["RAG"],
        paths["BM25"]
    )