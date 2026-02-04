#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract FINAL sequential patterns after hyperparameter selection.

Input:
- sequential_prefix_stats_k9.csv (TRAIN-only statistics)

Final configuration (selected on VALIDATION):
- min_support = 20
- entropy_eps = 0.40
- avg_k ≈ 2.73
- acc = 0.587
- cov = 1.000

This script FILTERS prefixes that satisfy:
- support >= min_support
- dH >= entropy_eps
- k >= MIN_K_ALLOWED

No re-evaluation. No validation data touched.
Reviewer-safe.
"""

import csv
from pathlib import Path

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "Dataset"

IN_CSV  = DATASET / "Variable_order_Markov_transition_rules" / "sequential_prefix_stats_k9.csv"
OUT_CSV = DATASET / "core" / "sequential_patterns_FINAL.csv"

# =========================
# FINAL CONFIG (LOCKED)
# =========================
MIN_SUPPORT = 20
ENTROPY_EPS = 0.40
MIN_K_ALLOWED = 2

# =========================
# MAIN
# =========================
def main():
    kept = []

    with open(IN_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"prefix", "k", "support", "dH", "pred_next"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Missing columns in input CSV: {sorted(missing)}")

        for r in reader:
            k = int(r["k"])
            support = int(r["support"])
            dH = float(r["dH"])

            if k < MIN_K_ALLOWED:
                continue
            if support < MIN_SUPPORT:
                continue
            if dH < ENTROPY_EPS:
                continue

            kept.append(r)

    # Sort for readability / analysis:
    # 1) shorter contexts first (model simplicity)
    # 2) higher support
    # 3) higher dH
    kept.sort(
        key=lambda r: (
            int(r["k"]),
            -int(r["support"]),
            -float(r["dH"]),
        )
    )

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prefix", "k", "support", "dH", "pred_next"]
        )
        writer.writeheader()
        for r in kept:
            writer.writerow(r)

    print("FINAL PATTERN EXTRACTION")
    print("------------------------")
    print(f"Input CSV : {IN_CSV}")
    print(f"Output CSV: {OUT_CSV}")
    print(f"Patterns kept: {len(kept)}")
    print(f"Constraints: support ≥ {MIN_SUPPORT}, dH ≥ {ENTROPY_EPS}, k ≥ {MIN_K_ALLOWED}")

if __name__ == "__main__":
    main()
