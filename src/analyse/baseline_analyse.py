#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BASELINE EVALUATION

Supports:
- JSONL baselines (LLM, KNN)
- CSV baseline (Distance)

Metrics:
- Hit@1
- Hit@5
- Hit@10
- Hit@15
- Hit@20
- MRR
- nDCG@5
- nDCG@10
- nDCG@15
- nDCG@20
"""

import json
import math
import pandas as pd
from pathlib import Path

# ============================================================
# FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_DIR = PROJECT_ROOT / "baseline"
TEST_FILE = PROJECT_ROOT / "Dataset" / "splits" / "test_eval_1000.jsonl"

FILES = {
    "Agentic-Grounded-Rag": PROJECT_ROOT /"results"/ "final_llm_ranking_test_eval_1000.csv",
    "LLM_ZERO_SHOT": BASELINE_DIR / "baseline_llm_zero_shot_predictions.jsonl",
    "KNN_BASELINE": BASELINE_DIR / "knn_predictions.jsonl",
    "DISTANCE_BASELINE": PROJECT_ROOT / "results" / "distance_baseline_top20.csv"
}

# ============================================================
# LOAD TARGETS (für CSV)
# ============================================================

targets = {}
with open(TEST_FILE, "r") as f:
    for line in f:
        row = json.loads(line)
        targets[row["user_id"]] = str(row["target"]["business_id"])

# ============================================================
# METRIC HELPERS
# ============================================================

def reciprocal_rank(target, preds):
    if target in preds:
        rank = preds.index(target) + 1
        return 1 / rank
    return 0


def ndcg_at_k(target, preds, k):
    if target not in preds[:k]:
        return 0
    rank = preds.index(target) + 1
    return 1 / math.log2(rank + 1)

# ============================================================
# JSONL EVALUATION
# ============================================================

def evaluate_jsonl(file_path):

    total = 0

    hits_at1 = hits_at5 = hits_at10 = hits_at15 = hits_at20 = 0
    mrr_sum = 0

    ndcg5_sum = ndcg10_sum = ndcg15_sum = ndcg20_sum = 0

    with open(file_path, "r") as f:
        for line in f:
            row = json.loads(line)

            target = str(row["target"])
            preds = [str(p) for p in row["prediction"]]

            total += 1

            if len(preds) > 0 and preds[0] == target:
                hits_at1 += 1
            if target in preds[:5]:
                hits_at5 += 1
            if target in preds[:10]:
                hits_at10 += 1
            if target in preds[:15]:
                hits_at15 += 1
            if target in preds[:20]:
                hits_at20 += 1

            mrr_sum += reciprocal_rank(target, preds)

            ndcg5_sum += ndcg_at_k(target, preds, 5)
            ndcg10_sum += ndcg_at_k(target, preds, 10)
            ndcg15_sum += ndcg_at_k(target, preds, 15)
            ndcg20_sum += ndcg_at_k(target, preds, 20)

    return {
        "samples": total,
        "Hit@1": hits_at1 / total,
        "Hit@5": hits_at5 / total,
        "Hit@10": hits_at10 / total,
        "Hit@15": hits_at15 / total,
        "Hit@20": hits_at20 / total,
        "MRR": mrr_sum / total,
        "nDCG@5": ndcg5_sum / total,
        "nDCG@10": ndcg10_sum / total,
        "nDCG@15": ndcg15_sum / total,
        "nDCG@20": ndcg20_sum / total
    }

# ============================================================
# CSV EVALUATION (DISTANCE)
# ============================================================

def evaluate_csv(file_path):

    df = pd.read_csv(file_path)

    total = 0

    hits_at1 = hits_at5 = hits_at10 = hits_at15 = hits_at20 = 0
    mrr_sum = 0

    ndcg5_sum = ndcg10_sum = ndcg15_sum = ndcg20_sum = 0

    grouped = df.sort_values("rank").groupby("user_id")

    for user_id, group in grouped:

        preds = group.sort_values("rank")["business_id"].astype(str).tolist()
        target = targets.get(user_id)

        if target is None:
            continue

        total += 1

        if len(preds) > 0 and preds[0] == target:
            hits_at1 += 1
        if target in preds[:5]:
            hits_at5 += 1
        if target in preds[:10]:
            hits_at10 += 1
        if target in preds[:15]:
            hits_at15 += 1
        if target in preds[:20]:
            hits_at20 += 1

        mrr_sum += reciprocal_rank(target, preds)

        ndcg5_sum += ndcg_at_k(target, preds, 5)
        ndcg10_sum += ndcg_at_k(target, preds, 10)
        ndcg15_sum += ndcg_at_k(target, preds, 15)
        ndcg20_sum += ndcg_at_k(target, preds, 20)

    return {
        "samples": total,
        "Hit@1": hits_at1 / total,
        "Hit@5": hits_at5 / total,
        "Hit@10": hits_at10 / total,
        "Hit@15": hits_at15 / total,
        "Hit@20": hits_at20 / total,
        "MRR": mrr_sum / total,
        "nDCG@5": ndcg5_sum / total,
        "nDCG@10": ndcg10_sum / total,
        "nDCG@15": ndcg15_sum / total,
        "nDCG@20": ndcg20_sum / total
    }

# ============================================================
# RUN EVALUATION
# ============================================================

print("\n==============================")
print("BASELINE EVALUATION RESULTS")
print("==============================\n")

for name, path in FILES.items():

    if str(path).endswith(".csv"):
        r = evaluate_csv(path)
    else:
        r = evaluate_jsonl(path)

    print(name)

    print("Samples:", r["samples"])
    print("Hit@1 :", round(r["Hit@1"], 4))
    print("Hit@5 :", round(r["Hit@5"], 4))
    print("Hit@10:", round(r["Hit@10"], 4))
    print("Hit@15:", round(r["Hit@15"], 4))
    print("Hit@20:", round(r["Hit@20"], 4))
    print("MRR   :", round(r["MRR"], 4))
    print("nDCG@5 :", round(r["nDCG@5"], 4))
    print("nDCG@10:", round(r["nDCG@10"], 4))
    print("nDCG@15:", round(r["nDCG@15"], 4))
    print("nDCG@20:", round(r["nDCG@20"], 4))
    print()