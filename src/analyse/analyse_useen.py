#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
from collections import defaultdict
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_DIR = PROJECT_ROOT / "baseline"

TEST_JSONL = DATASET_DIR / "splits" / "test_eval_1000.jsonl"

LLM_RANKING_CSV = RESULTS_DIR / "final_llm_ranking_test_eval_1000.csv"

LLM_BASELINE_JSONL = BASELINE_DIR / "baseline_llm_zero_shot_predictions.jsonl"
KNN_BASELINE_JSONL = BASELINE_DIR / "knn_predictions.jsonl"

DISTANCE_CSV = BASELINE_DIR / "distance_baseline_top20.csv"

# ============================================================
# STEP 1
# unseen targets sammeln
# ============================================================

unseen_targets = {}  # user_id -> target

with open(TEST_JSONL, "r") as f:
    for line in f:
        row = json.loads(line)

        if row["target_seen_in_last_9"] is False:
            user_id = str(row["user_id"])
            target = str(row["target"]["business_id"])

            unseen_targets[user_id] = target

print("Unseen targets:", len(unseen_targets))


# ============================================================
# FUNCTION
# metrics berechnen
# ============================================================

def compute_metrics(rank_list, total):

    hits1 = sum(1 for r in rank_list if r == 1)
    hits5 = sum(1 for r in rank_list if r <= 5)
    hits10 = sum(1 for r in rank_list if r <= 10)
    hits15 = sum(1 for r in rank_list if r <= 15)
    hits20 = sum(1 for r in rank_list if r <= 20)

    mean_rank = None
    if rank_list:
        mean_rank = sum(rank_list) / len(rank_list)

    print("Hit@1 :", round(hits1 / total, 4))
    print("Hit@5 :", round(hits5 / total, 4))
    print("Hit@10:", round(hits10 / total, 4))
    print("Hit@15:", round(hits15 / total, 4))
    print("Hit@20:", round(hits20 / total, 4))

    if mean_rank:
        print("MeanRank:", round(mean_rank, 2))


# ============================================================
# STEP 2
# AGENTIC RAG / LLM RERANK
# ============================================================

user_predictions = defaultdict(list)

with open(LLM_RANKING_CSV, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:

        user_id = row["user_id"]

        if user_id not in unseen_targets:
            continue

        business = row["business_id"]
        rank = int(row["rank"])

        user_predictions[user_id].append((rank, business))


ranks = []

for user_id, target in unseen_targets.items():

    preds = sorted(user_predictions[user_id])

    found_rank = None

    for rank, business in preds:
        if business == target:
            found_rank = rank
            break

    if found_rank:
        ranks.append(found_rank)


print("\n==============================")
print("AGENTIC RAG (LLM RERANK)")
print("==============================")

compute_metrics(ranks, len(unseen_targets))


# ============================================================
# STEP 3
# LLM ZERO SHOT BASELINE
# ============================================================

ranks = []

with open(LLM_BASELINE_JSONL, "r") as f:

    for line in f:
        row = json.loads(line)

        user_id = str(row["user_id"])

        if user_id not in unseen_targets:
            continue

        target = str(row["target"])

        preds = [str(p) for p in row["prediction"]]

        found_rank = None

        for i, p in enumerate(preds):
            if p == target:
                found_rank = i + 1
                break

        if found_rank:
            ranks.append(found_rank)


print("\n==============================")
print("LLM ZERO SHOT BASELINE")
print("==============================")

compute_metrics(ranks, len(unseen_targets))


# ============================================================
# STEP 4
# KNN BASELINE
# ============================================================

ranks = []

with open(KNN_BASELINE_JSONL, "r") as f:

    for line in f:
        row = json.loads(line)

        user_id = str(row["user_id"])

        if user_id not in unseen_targets:
            continue

        target = str(row["target"])

        preds = [str(p) for p in row["prediction"]]

        found_rank = None

        for i, p in enumerate(preds):

            if p == target:
                found_rank = i + 1
                break

        if found_rank:
            ranks.append(found_rank)


print("\n==============================")
print("KNN BASELINE")
print("==============================")

compute_metrics(ranks, len(unseen_targets))


# ============================================================
# STEP 5
# DISTANCE BASELINE
# ============================================================

user_predictions = defaultdict(list)

with open(DISTANCE_CSV, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:

        user_id = row["user_id"]

        if user_id not in unseen_targets:
            continue

        business = row["business_id"]
        rank = int(row["rank"])

        user_predictions[user_id].append((rank, business))


ranks = []

for user_id, target in unseen_targets.items():

    preds = sorted(user_predictions[user_id])

    found_rank = None

    for rank, business in preds:
        if business == target:
            found_rank = rank
            break

    if found_rank:
        ranks.append(found_rank)


print("\n==============================")
print("DISTANCE BASELINE")
print("(Spatial Proximity)")
print("==============================")

compute_metrics(ranks, len(unseen_targets))