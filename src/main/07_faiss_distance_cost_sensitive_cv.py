#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leave-One-Seed-Out Evaluation (Cost-Sensitive)
+ Final Production Threshold Training
+ Score Distribution Plot
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# ============================================================
# CONFIG
# ============================================================

SEEDS = ["2026", "2027", "2028"]

C_FP = 2.0   # cost of false loop
C_FN = 1.0   # cost of lost hit

BASE = Path("/Users/adrianhoffmann/Documents/Master/Projektarbeit/AgenticGroundedRagClone")
DATA = BASE / "Dataset"

OUTPUT_DIR = BASE / "analysis_outputs" / "faiss_threshold_cv_cost_sensitive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

all_rows = []

for seed in SEEDS:

    HYP_JSONL = DATA / "validation" / f"hypotheses_seed{seed}_without_cluster.jsonl"
    RAG_CSV   = DATA / "validation_rag" / f"rag_seed{seed}_without_cluster_top100.csv"
    BM25_CSV  = DATA / "validation_rag" / f"rag_seed{seed}_without_cluster_bm25_top15.csv"

    df_rag = pd.read_csv(RAG_CSV)
    df_bm25 = pd.read_csv(BM25_CSV)

    df_rag["user_id"] = df_rag["user_id"].astype(str)
    df_bm25["user_id"] = df_bm25["user_id"].astype(str)

    targets = {}

    with open(HYP_JSONL, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            targets[str(obj["user_id"])] = int(obj["target"])

    for uid, target in targets.items():

        rag_user = df_rag[df_rag["user_id"] == uid]
        if rag_user.empty:
            continue

        target_row = rag_user[rag_user["business_id"] == target]
        if target_row.empty:
            continue

        faiss_dist = float(target_row["faiss_distance"].values[0])
        bm25_user = df_bm25[df_bm25["user_id"] == uid]
        bm25_hit = int((bm25_user["business_id"] == target).any())

        all_rows.append({
            "seed": seed,
            "faiss_distance": faiss_dist,
            "bm25_hit": bm25_hit
        })

df = pd.DataFrame(all_rows)

print("\nTotal usable cases:", len(df))

# ============================================================
# LEAVE-ONE-SEED-OUT CV
# ============================================================

results = []
all_test_labels = []
all_test_scores = []

for test_seed in SEEDS:

    print(f"\n=== TEST SEED: {test_seed} ===")

    train_df = df[df["seed"] != test_seed]
    test_df  = df[df["seed"] == test_seed]

    thresholds = np.unique(train_df["faiss_distance"])
    best_cost = np.inf
    best_threshold = None

    for t in thresholds:

        y_pred = (train_df["faiss_distance"] >= t).astype(int)
        y_true = train_df["bm25_hit"].values

        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        cost = C_FP * FP + C_FN * FN

        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    print("Train Threshold:", round(best_threshold, 6))
    print("Train Cost:", best_cost)

    # Test evaluation
    y_true_test = test_df["bm25_hit"].values
    y_pred_test = (test_df["faiss_distance"] >= best_threshold).astype(int)

    TP = np.sum((y_pred_test == 1) & (y_true_test == 1))
    FP = np.sum((y_pred_test == 1) & (y_true_test == 0))
    FN = np.sum((y_pred_test == 0) & (y_true_test == 1))
    TN = np.sum((y_pred_test == 0) & (y_true_test == 0))

    test_cost = C_FP * FP + C_FN * FN

    print("TP:", TP, "FP:", FP, "FN:", FN, "TN:", TN)
    print("Test Cost:", test_cost)

    all_test_labels.extend(y_true_test.tolist())
    all_test_scores.extend(test_df["faiss_distance"].values.tolist())

    results.append({
        "test_seed": test_seed,
        "threshold": best_threshold,
        "test_cost": test_cost
    })

# ============================================================
# POOLED ROC (OUT-OF-FOLD)
# ============================================================

fpr_all, tpr_all, _ = roc_curve(all_test_labels, all_test_scores)
auc_all = auc(fpr_all, tpr_all)

plt.figure()
plt.plot(fpr_all, tpr_all, label=f"AUC = {auc_all:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(OUTPUT_DIR / "roc_pooled_cv.png", dpi=300)
plt.close()

print("\nPooled CV ROC-AUC:", round(auc_all, 6))

# ============================================================
# FINAL PRODUCTION THRESHOLD (ALL DATA)
# ============================================================

print("\n=== FINAL PRODUCTION TRAINING (ALL DATA) ===")

thresholds = np.unique(df["faiss_distance"])
best_cost = np.inf
best_threshold = None

for t in thresholds:

    y_pred = (df["faiss_distance"] >= t).astype(int)
    y_true = df["bm25_hit"].values

    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    cost = C_FP * FP + C_FN * FN

    if cost < best_cost:
        best_cost = cost
        best_threshold = t

print("Final Production Threshold:", round(best_threshold, 6))
print("Final Production Cost:", best_cost)

# ============================================================
# SCORE DISTRIBUTION (KDE STYLE - LIKE PAPER FIGURE)
# ============================================================

from scipy.stats import gaussian_kde

plt.figure(figsize=(10, 6))

df_hit  = df[df["bm25_hit"] == 1]["faiss_distance"].values
df_miss = df[df["bm25_hit"] == 0]["faiss_distance"].values

# KDE estimation
kde_hit  = gaussian_kde(df_hit)
kde_miss = gaussian_kde(df_miss)

x_vals = np.linspace(min(df["faiss_distance"]),
                     max(df["faiss_distance"]), 500)

y_hit  = kde_hit(x_vals)
y_miss = kde_miss(x_vals)

# Plot filled curves
plt.fill_between(x_vals, y_hit,  alpha=0.5, label="BM25 Hit")
plt.plot(x_vals, y_hit)

plt.fill_between(x_vals, y_miss, alpha=0.5, label="BM25 Miss")
plt.plot(x_vals, y_miss)

# Optional: decision boundary
plt.axvline(best_threshold,
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {best_threshold:.3f}")

plt.title("Score Distributions (All Seeds)")
plt.xlabel("FAISS Distance")
plt.ylabel("Density")
plt.legend()

plt.savefig(OUTPUT_DIR / "score_distribution_pooled.png", dpi=300)
plt.close()

print("Saved score_distribution_pooled.png")

# ============================================================
# SAVE THRESHOLD
# ============================================================

with open(OUTPUT_DIR / "final_production_threshold.json", "w") as f:
    json.dump({
        "threshold": float(best_threshold),
        "C_FP": C_FP,
        "C_FN": C_FN
    }, f, indent=4)

print("\nSaved final threshold to JSON.")
print("\n==============================================\n")