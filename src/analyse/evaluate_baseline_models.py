#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BASELINE EVALUATION

Metrics:
- Hit@1
- Hit@5
- Hit@10
- MRR
- nDCG@5
- nDCG@10
"""

import json
import math
from pathlib import Path

# ============================================================
# FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


BASELINE_DIR = PROJECT_ROOT / "baseline"

FILES = {
    "LLM_ZERO_SHOT": BASELINE_DIR / "baseline_llm_zero_shot_predictions.jsonl",
    "KNN_BASELINE": BASELINE_DIR / "knn_predictions.jsonl"
}

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

    dcg = 1 / math.log2(rank + 1)
    idcg = 1

    return dcg / idcg


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(file_path):

    total = 0

    hits_at1 = 0
    hits_at5 = 0
    hits_at10 = 0

    mrr_sum = 0
    ndcg5_sum = 0
    ndcg10_sum = 0

    with open(file_path, "r") as f:

        for line in f:

            row = json.loads(line)

            target = str(row["target"])
            preds = [str(p) for p in row["prediction"]]

            total += 1

            # HIT
            if len(preds) > 0 and preds[0] == target:
                hits_at1 += 1

            if target in preds[:5]:
                hits_at5 += 1

            if target in preds[:10]:
                hits_at10 += 1

            # MRR
            mrr_sum += reciprocal_rank(target, preds)

            # nDCG
            ndcg5_sum += ndcg_at_k(target, preds, 5)
            ndcg10_sum += ndcg_at_k(target, preds, 10)

    results = {
        "samples": total,
        "Hit@1": hits_at1 / total,
        "Hit@5": hits_at5 / total,
        "Hit@10": hits_at10 / total,
        "MRR": mrr_sum / total,
        "nDCG@5": ndcg5_sum / total,
        "nDCG@10": ndcg10_sum / total
    }

    return results


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n==============================")
print("BASELINE EVALUATION RESULTS")
print("==============================\n")

for name, path in FILES.items():

    r = evaluate(path)

    print(name)
    print("Samples:", r["samples"])
    print("Hit@1:", round(r["Hit@1"], 4))
    print("Hit@5:", round(r["Hit@5"], 4))
    print("Hit@10:", round(r["Hit@10"], 4))
    print("MRR:", round(r["MRR"], 4))
    print("nDCG@5:", round(r["nDCG@5"], 4))
    print("nDCG@10:", round(r["nDCG@10"], 4))
    print()