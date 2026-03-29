#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM SCORE AGGREGATION + FINAL RANKING

Steps:
1. Load LLM behavioural scores
2. Compute mean score: (s_dist + s_time + s_sem) / 3
3. Rank candidates by mean score
4. Tie-break using BM25 rank
5. Save final ranking
"""

import json
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

SCORING_PATH = RESULTS_DIR / "scoring_test_eval_1000.jsonl"
RERANK_PATH = RESULTS_DIR / "distance_reranked_top20.csv"

OUTPUT_PATH = RESULTS_DIR / "final_llm_ranking_test_eval_1000.csv"


# ============================================================
# LOAD LLM SCORES
# ============================================================

records = []

with open(SCORING_PATH, "r", encoding="utf-8") as f:
    for line in f:

        obj = json.loads(line)

        user_id = obj["user_id"]

        scores = obj.get("scores")

        if not scores:
            continue

        for entry in scores:

            bid = entry["business_id"]

            s_dist = entry["s_dist"]
            s_time = entry["s_time"]
            s_sem = entry["s_sem"]

            mean_score = (s_dist + s_time + s_sem) / 3

            records.append({
                "user_id": user_id,
                "business_id": bid,
                "score": mean_score
            })

df_scores = pd.DataFrame(records)


# ============================================================
# LOAD BM25 RANKS
# ============================================================

df_rerank = pd.read_csv(RERANK_PATH)

df_rerank = df_rerank.rename(columns={
    "rank": "distance_rank"
})


# ============================================================
# MERGE
# ============================================================

df = df_scores.merge(
    df_rerank[["user_id", "business_id", "distance_rank"]],
    on=["user_id", "business_id"],
    how="left"
)


# ============================================================
# SORTING
# ============================================================

df = df.sort_values(
    by=["user_id", "score", "distance_rank"],
    ascending=[True, False, True]
)


# ============================================================
# FINAL RANK ASSIGNMENT
# ============================================================

df["rank"] = df.groupby("user_id").cumcount() + 1


# ============================================================
# SAVE RESULT
# ============================================================

df = df[[
    "user_id",
    "business_id",
    "rank",
    "score",
    "distance_rank"
]]

df.to_csv(OUTPUT_PATH, index=False)

print("Final ranking written to:")
print(OUTPUT_PATH)