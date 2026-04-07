#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GLOBAL LOG-LIKELIHOOD EVALUATION FOR MARKOV ORDER SELECTION

- TRAIN: estimate transition probabilities
- VALIDATION: compute global log-likelihood
- Purpose: select maximum useful order k (model-level, not per sequence)

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

TRAIN_PATH = DATASET / "splits" / "train.jsonl"
VAL_PATH   = DATASET / "splits" / "validation_cluster_only.jsonl"
CLUSTER_PATH = DATASET / "representations" / "Cluster" / "poi_cluster_assignment.csv"

# =========================
# CONFIG
# =========================
K_MAX = 9
MIN_K_ALLOWED = 2
EPS = 1e-12   # numerical safety (log(0))

# =========================
# IO
# =========================
def load_cluster_mapping(path):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            poi, cluster = line.strip().split(",")
            mapping[int(poi)] = int(cluster)
    return mapping

def load_train_sequences(path, poi_to_cluster):
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

def load_validation_samples(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            inp = obj.get("input", [])
            tgt = obj.get("target", None)
            if tgt is not None and len(inp) >= MIN_K_ALLOWED:
                samples.append((inp, tgt))
    return samples

# =========================
# TRAINING
# =========================
def train_markov(seqs):
    prefix_count = Counter()
    transition_count = Counter()

    for seq in seqs:
        L = len(seq)
        for k in range(MIN_K_ALLOWED, K_MAX + 1):
            for i in range(L - k):
                prefix = tuple(seq[i:i+k])
                nxt = seq[i+k]
                prefix_count[prefix] += 1
                transition_count[(prefix, nxt)] += 1

    P = defaultdict(dict)
    for (p, n), c in transition_count.items():
        P[p][n] = c

    for p in P:
        Z = sum(P[p].values())
        for n in P[p]:
            P[p][n] /= Z

    return P

# =========================
# LOG-LIKELIHOOD
# =========================
def log_likelihood(P, val_samples, k_max):
    ll = 0.0
    used = 0

    for inp, true_next in val_samples:
        matched = False

        for k in range(min(k_max, len(inp)), MIN_K_ALLOWED - 1, -1):
            prefix = tuple(inp[-k:])
            if prefix in P and true_next in P[prefix]:
                p = P[prefix][true_next]
                ll += math.log(p + EPS)
                used += 1
                matched = True
                break

        if not matched:
            # unseen transition → ignored (standard in LL comparison)
            pass

    return ll, used

# =========================
# MAIN
# =========================
def main():
    poi_to_cluster = load_cluster_mapping(CLUSTER_PATH)
    train_seqs = load_train_sequences(TRAIN_PATH, poi_to_cluster)
    val_samples = load_validation_samples(VAL_PATH)

    print(f"TRAIN sequences: {len(train_seqs)}")
    print(f"VALIDATION samples: {len(val_samples)}\n")

    P = train_markov(train_seqs)

    print("k,log_likelihood,used_transitions")

    prev_ll = None

    for k in range(MIN_K_ALLOWED, K_MAX + 1):
        ll, used = log_likelihood(P, val_samples, k)
        delta = ll - prev_ll if prev_ll is not None else 0.0
        print(f"{k},{ll:.2f},{used}  ΔLL={delta:.2f}")
        prev_ll = ll

if __name__ == "__main__":
    main()
