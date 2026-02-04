#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POI → Cluster Assignment Export (Reviewer-clean)

This script:
- loads normalized POI text embeddings
- performs KMeans clustering
- writes one CSV with (poi_id, cluster_id)
- prints cluster statistics to console (English)

No analysis, no metrics, no heuristics.
Pure structural transparency.
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REP_DIR = PROJECT_ROOT / "Dataset" / "representations"

EMB_PATH =  REP_DIR / "poi_text_embeddings.jsonl"
OUT_DIR = REP_DIR / "Cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "poi_cluster_assignment.csv"

# =========================
# CONFIG
# =========================
K = 16
RANDOM_STATE = 42

# =========================
# IO
# =========================
def load_embeddings(path):
    poi_ids, X = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            poi_ids.append(int(obj["business_id"]))
            X.append(obj["embedding"])
    X = normalize(np.asarray(X, dtype=np.float64))
    return poi_ids, X

# =========================
# MAIN
# =========================
def main():
    poi_ids, X = load_embeddings(EMB_PATH)

    km = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=10
    )
    labels = km.fit_predict(X)

    # -------------------------
    # Write CSV
    # -------------------------
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("poi_id,cluster_id\n")
        for pid, lab in zip(poi_ids, labels):
            f.write(f"{pid},{lab}\n")

    # -------------------------
    # Console statistics
    # -------------------------
    counts = Counter(labels)

    print("\nCluster assignment summary")
    print("--------------------------")
    print(f"Number of clusters: {len(counts)}")
    print(f"Total number of POIs: {len(poi_ids)}\n")

    print("POIs per cluster:")
    for cluster_id in sorted(counts):
        print(f"  Cluster {cluster_id}: {counts[cluster_id]} POIs")

    print(f"\nCSV written to: {OUT_CSV}")

# =========================
if __name__ == "__main__":
    main()
