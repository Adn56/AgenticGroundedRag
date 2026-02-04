#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CREATE PREFIX STATS (TRAIN) – REVIEWER-SAFE

Stores EXACTLY the same dH definition as used in the direct grid-search:

dH(prefix) =
- if k == MIN_K_ALLOWED: H(prefix)
- if k > MIN_K_ALLOWED:
    - if suffix exists: H(suffix) - H(prefix)
    - else: H(prefix)

Output CSV columns:
prefix,k,support,dH,pred_next
"""

import json
import math
from pathlib import Path
from collections import Counter, defaultdict

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "Dataset"

CLUSTER_PATH = DATASET / "representations" / "Cluster" / "poi_cluster_assignment.csv"
TRAIN_PATH   = DATASET / "splits" / "train.jsonl"

OUT_DIR = DATASET / "representations" / "Cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "sequential_prefix_stats_k9.csv"

# =========================
# CONFIG
# =========================
K_MAX = 9
MIN_K_ALLOWED = 2

# =========================
# HELPERS
# =========================
def entropy(dist):
    return -sum(p * math.log2(p) for p in dist.values())

def load_cluster_mapping(path: Path):
    m = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            poi, cluster = line.strip().split(",")
            m[int(poi)] = int(cluster)
    return m

def load_train_sequences(path: Path, poi_to_cluster: dict):
    seqs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            traj = obj.get("trajectory", [])
            clusters = [
                poi_to_cluster[int(step["business_id"])]
                for step in traj
                if "business_id" in step and int(step["business_id"]) in poi_to_cluster
            ]
            if len(clusters) >= MIN_K_ALLOWED + 1:
                seqs.append(clusters)
    return seqs

# =========================
# MAIN
# =========================
def main():
    poi_to_cluster = load_cluster_mapping(CLUSTER_PATH)
    sequences = load_train_sequences(TRAIN_PATH, poi_to_cluster)
    print(f"Loaded TRAIN sequences: {len(sequences)}")

    prefix_count = Counter()
    transition_count = Counter()

    for seq in sequences:
        L = len(seq)
        for k in range(MIN_K_ALLOWED, K_MAX + 1):
            for i in range(L - k):
                prefix = tuple(seq[i:i+k])
                nxt = seq[i+k]
                prefix_count[prefix] += 1
                transition_count[(prefix, nxt)] += 1

    # conditional distributions P(next | prefix)
    P = defaultdict(dict)
    for (p, n), c in transition_count.items():
        P[p][n] = c
    for p in P:
        Z = sum(P[p].values())
        for n in P[p]:
            P[p][n] /= Z

    rows = []

    for prefix in P:
        k = len(prefix)
        supp = prefix_count[prefix]
        H_p = entropy(P[prefix])

        if k == MIN_K_ALLOWED:
            dH = H_p
        else:
            suffix = prefix[1:]
            if suffix in P:
                dH = entropy(P[suffix]) - H_p
            else:
                dH = H_p

        pred_next = max(P[prefix], key=P[prefix].get)

        rows.append((
            "-".join(map(str, prefix)),
            k,
            supp,
            f"{dH:.6f}",
            pred_next
        ))

    rows.sort(key=lambda r: (r[1], r[2]), reverse=True)

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("prefix,k,support,dH,pred_next\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")

    print(f"Written {len(rows)} prefixes → {OUT_CSV}")

if __name__ == "__main__":
    main()
