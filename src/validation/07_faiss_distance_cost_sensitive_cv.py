#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 Analysis:
 Can the Top1 FAISS distance predict
 whether the target is contained in the Distance Top-20?
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from scipy.stats import gaussian_kde

# ============================================================
# CONFIG
# ============================================================

SEEDS = ["2026", "2027", "2028"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "top1_predicts_distance20"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

all_rows = []

for seed in SEEDS:

    HYP_JSONL = DATASET_DIR / "validation" / f"hypotheses_seed{seed}_without_cluster.jsonl"
    RAG_CSV   = DATASET_DIR / "validation_rag" / f"rag_seed{seed}_without_cluster_top200.csv"
    TOP20_CSV  = DATASET_DIR / "validation_rag" / f"rag_{seed}_without_cluster_distance_top20.csv"

    df_rag = pd.read_csv(RAG_CSV)
    df_bm25 = pd.read_csv(TOP20_CSV)

    df_rag["user_id"] = df_rag["user_id"].astype(str)
    df_bm25["user_id"] = df_bm25["user_id"].astype(str)

    # Determine Top1 per user
    df_top1 = (
        df_rag.sort_values(["user_id", "faiss_distance"])
        .groupby("user_id")
        .first()
        .reset_index()
    )

    # Load target
    targets = {}
    with open(HYP_JSONL, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            targets[str(obj["user_id"])] = int(obj["target"])

    for uid, target in targets.items():

        top1_row = df_top1[df_top1["user_id"] == uid]
        if top1_row.empty:
            continue

        top1_distance = float(top1_row["faiss_distance"].values[0])

        bm25_user = df_bm25[df_bm25["user_id"] == uid]
        bm25_hit = int((bm25_user["business_id"] == target).any())

        all_rows.append({
            "top1_distance": top1_distance,
            "bm25_hit": bm25_hit
        })

df = pd.DataFrame(all_rows)

print("Total cases:", len(df))
print("Min distance:", df["top1_distance"].min())
print("Max distance:", df["top1_distance"].max())

# ============================================================
# ROC
# lower distance = better
# ============================================================

scores = -df["top1_distance"]

fpr, tpr, _ = roc_curve(df["bm25_hit"], scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC: Top1 Distance → BM25 Hit")
plt.legend()
plt.savefig(OUTPUT_DIR / "roc_top1.png", dpi=300)
plt.close()

print("ROC-AUC:", round(roc_auc, 4))

# ============================================================
# SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10,6))

df_hit  = df[df["bm25_hit"] == 1]["top1_distance"].values
df_miss = df[df["bm25_hit"] == 0]["top1_distance"].values

kde_hit  = gaussian_kde(df_hit)
kde_miss = gaussian_kde(df_miss)

x_vals = np.linspace(df["top1_distance"].min(),
                     df["top1_distance"].max(), 500)

plt.fill_between(x_vals, kde_hit(x_vals), alpha=0.5, label="BM25 Hit")
plt.fill_between(x_vals, kde_miss(x_vals), alpha=0.5, label="BM25 Miss")

plt.xlabel("Top1 FAISS Distance")
plt.ylabel("Density")
plt.title("Score Distribution: Top1 Distance")
plt.legend()

plt.savefig(OUTPUT_DIR / "score_distribution_top1.png", dpi=300)
plt.close()

print("Saved plots to:", OUTPUT_DIR)
print("==============================================")
