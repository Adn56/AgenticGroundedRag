#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid Search using PRECOMPUTED TRAIN prefix stats (CSV)

This script is now BIT-IDENTICAL in logic to the direct grid-search.
"""

import csv
import json
from pathlib import Path

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "Dataset"

PREFIX_STATS = DATASET / "Variable_order_Markov_transition_rules" / "sequential_prefix_stats_k9.csv"
VAL_PATH     = DATASET / "splits" / "validation_cluster_only.jsonl"

# =========================
# CONFIG
# =========================
K_MAX = 9
MIN_K_ALLOWED = 2

SUPPORT_CANDIDATES = [1, 2, 5, 10, 20, 40, 60, 80, 100]
ENTROPY_GRID = [0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00]
ACC_TOL = 0.002

# =========================
# LOADERS
# =========================
def load_prefix_stats(path: Path):
    support = {}
    dH = {}
    pred = {}

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            p = tuple(map(int, r["prefix"].split("-")))
            support[p] = int(r["support"])
            dH[p] = float(r["dH"])
            pred[p] = int(r["pred_next"])

    return support, dH, pred

def load_validation(path: Path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            inp = obj.get("input", [])
            tgt = obj.get("target", None)
            if tgt is not None and len(inp) >= MIN_K_ALLOWED:
                samples.append((tuple(inp), int(tgt)))
    return samples

# =========================
# EVALUATION
# =========================
def evaluate(prefix_support, prefix_dH, prefix_pred, val_samples, min_support, eps):
    correct = 0
    covered = 0
    used_k = []

    for inp, true_next in val_samples:
        for k in range(min(K_MAX, len(inp)), MIN_K_ALLOWED - 1, -1):
            prefix = inp[-k:]

            if prefix_support.get(prefix, 0) < min_support:
                continue
            if prefix_dH.get(prefix, 0.0) < eps:
                continue

            pred = prefix_pred[prefix]
            covered += 1
            used_k.append(k)
            if pred == true_next:
                correct += 1
            break

    acc = correct / covered if covered else 0.0
    cov = covered / len(val_samples) if val_samples else 0.0
    avg_k = sum(used_k) / len(used_k) if used_k else 0.0
    return acc, cov, avg_k

def better(a, b):
    if b is None:
        return True

    sa, ea, acca, cova, avga = a
    sb, eb, accb, covb, avgb = b

    if acca > accb + ACC_TOL:
        return True
    if abs(acca - accb) <= ACC_TOL:
        if cova != covb:
            return cova > covb
        if sa != sb:
            return sa > sb
        if ea != eb:
            return ea > eb
        return avga < avgb
    return False

# =========================
# MAIN
# =========================
def main():
    prefix_support, prefix_dH, prefix_pred = load_prefix_stats(PREFIX_STATS)
    val_samples = load_validation(VAL_PATH)

    print(f"Loaded prefixes: {len(prefix_support)}")
    print(f"Validation samples: {len(val_samples)}\n")
    print("min_support,entropy_eps,accuracy,coverage,avg_order")

    best = None

    for s in SUPPORT_CANDIDATES:
        for e in ENTROPY_GRID:
            acc, cov, avgk = evaluate(prefix_support, prefix_dH, prefix_pred, val_samples, s, e)
            row = (s, e, acc, cov, avgk)
            print(f"{s},{e:.2f},{acc:.3f},{cov:.3f},{avgk:.2f}")
            if better(row, best):
                best = row

    s, e, acc, cov, avgk = best
    print("\nRECOMMENDED:")
    print(f"min_support={s}, entropy_eps={e:.2f}, acc={acc:.3f}, cov={cov:.3f}, avg_k={avgk:.2f}")

if __name__ == "__main__":
    main()
