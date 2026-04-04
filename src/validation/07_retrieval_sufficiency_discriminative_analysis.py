#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RETRIEVAL SUFFICIENCY – DISCRIMINATIVE ANALYSIS PROTOCOL
---------------------------------------------------------

Outputs:
- ROC-AUC
- Max Youden J + τ
- KS statistic
- Distribution statistics
- Saved plots:
    * roc_curve.png / .pdf
    * precision_recall_curve.png / .pdf
    * score_distributions.png / .pdf
    * youden_curve.png / .pdf
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
from scipy.stats import ks_2samp

# ============================================================
# CONFIG
# ============================================================

SEEDS = ["2026", "2027", "2028"]
TAU_GRID = np.linspace(0, 1, 5001)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "Dataset"
OUTPUT_DIR = PROJECT_ROOT / "analysis_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

def load_seed(seed):

    llm_file = DATA / "scoring" / f"outputs_seed{seed}.jsonl"
    bm25_file = DATA / "validation_rag" / f"rag_{SEED_TAG}_without_cluster_distance_top20.csv

    bm25 = pd.read_csv(bm25_file)
    positives = set(bm25[bm25["is_target"] == 1]["user_id"].values)

    rows = []

    with llm_file.open() as f:
        for line in f:
            obj = json.loads(line)
            uid = obj.get("user_id")
            if uid is None:
                continue

            score = float(obj.get("retrieval_sufficiency", 0.0))
            y = 1 if uid in positives else 0
            rows.append((score, y))

    return pd.DataFrame(rows, columns=["score", "y"])

df = pd.concat([load_seed(s) for s in SEEDS], ignore_index=True)

scores = df["score"].values
y = df["y"].values

print("\n===================================================")
print("RETRIEVAL SUFFICIENCY – DISCRIMINATIVE ANALYSIS")
print("===================================================")

print(f"\nSample size N = {len(df)}")
print(f"Positive class rate = {np.mean(y):.4f}")

# ============================================================
# 1) ROC-AUC
# ============================================================

auc = roc_auc_score(y, scores)
print(f"\nROC-AUC = {auc:.6f}")

# ============================================================
# 2) Youden Index
# ============================================================

best_youden = -np.inf
best_tau = None

for tau in TAU_GRID:
    y_pred = (scores < tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = recall + specificity - 1

    if youden > best_youden:
        best_youden = youden
        best_tau = tau

print(f"Max Youden J = {best_youden:.6f}")
print(f"τ (Youden max) = {best_tau:.6f}")

# ============================================================
# 3) KS-Test
# ============================================================

loop_scores = scores[y == 1]
accept_scores = scores[y == 0]

ks_stat, ks_p = ks_2samp(loop_scores, accept_scores)

print(f"\nKolmogorov–Smirnov statistic = {ks_stat:.6f}")
print(f"KS p-value = {ks_p:.6f}")

# ============================================================
# 4) Distribution statistics
# ============================================================

print("\nScore statistics:")
print("LOOP mean =", np.mean(loop_scores))
print("ACCEPT mean =", np.mean(accept_scores))
print("LOOP std =", np.std(loop_scores))
print("ACCEPT std =", np.std(accept_scores))


# ------------------------------------------------------------
# 5) Raw Score Scatter (Class-wise)
# ------------------------------------------------------------

plt.figure(figsize=(8, 3))

# Jitter
y_jitter = y + np.random.normal(0, 0.02, size=len(y))

plt.scatter(scores[y == 0], y_jitter[y == 0],
            alpha=0.6, label="ACCEPT")

plt.scatter(scores[y == 1], y_jitter[y == 1],
            alpha=0.6, label="LOOP")

plt.yticks([0, 1], ["ACCEPT", "LOOP"])
plt.xlabel("retrieval_sufficiency")
plt.title("Raw Score Distribution by Class")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "score_scatter.png", dpi=300)
plt.savefig(OUTPUT_DIR / "score_scatter.pdf")
plt.close()


# ============================================================
# PLOTS (SAVED)
# ============================================================

sns.set(style="whitegrid")

# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(y, scores)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300)
plt.savefig(OUTPUT_DIR / "roc_curve.pdf")
plt.close()

# ------------------------------------------------------------
# Precision-Recall Curve
# ------------------------------------------------------------

precision, recall_curve, _ = precision_recall_curve(y, scores)

plt.figure(figsize=(6,6))
plt.plot(recall_curve, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=300)
plt.savefig(OUTPUT_DIR / "precision_recall_curve.pdf")
plt.close()

# ------------------------------------------------------------
# Score Distributions
# ------------------------------------------------------------

plt.figure(figsize=(7,5))
sns.kdeplot(loop_scores, label="LOOP", fill=True, alpha=0.4)
sns.kdeplot(accept_scores, label="ACCEPT", fill=True, alpha=0.4)
plt.xlabel("retrieval_sufficiency")
plt.ylabel("Density")
plt.title("Score Distributions")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "score_distributions.png", dpi=300)
plt.savefig(OUTPUT_DIR / "score_distributions.pdf")
plt.close()

# ------------------------------------------------------------
# Youden Curve
# ------------------------------------------------------------

youden_values = []

for tau in TAU_GRID:
    y_pred = (scores < tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden_values.append(recall_val + specificity - 1)

plt.figure(figsize=(7,5))
plt.plot(TAU_GRID, youden_values)
plt.xlabel("τ")
plt.ylabel("Youden J")
plt.title("Youden Index over τ")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "youden_curve.png", dpi=300)
plt.savefig(OUTPUT_DIR / "youden_curve.pdf")
plt.close()

print(f"\nPlots saved to: {OUTPUT_DIR}")
print("\n===================================================")
print("ANALYSIS COMPLETE")
print("===================================================")
