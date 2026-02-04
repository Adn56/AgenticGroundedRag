#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
k-sensitivity analysis for sequential transitions.

Key properties (NO hidden restrictions):
- No Top-N filtering
- No greedy matching
- No intersection-only JS
- All transition patterns included
- Global optimal cluster alignment (Hungarian)
- Explicit mass-loss measurement

This script measures *true* k-sensitivity.
"""

import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / "Dataset"

EMB_PATH = DATASET / "representations" / "poi_text_embeddings.jsonl"
SEQ_PATH = DATASET / "splits" / "train.jsonl"

OUT_DIR = DATASET / "representations" / "Cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "k_transition_metrics_full.csv"
PAIRWISE_CSV = OUT_DIR / "k_pairwise_stability_full.csv"

# =========================
# CONFIG
# =========================
K_VALUES = [8, 12, 16, 20, 30]
RANDOM_STATE = 42

# =========================
# IO
# =========================
def load_embeddings(path):
    ids, X = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ids.append(int(o["business_id"]))
            X.append(o["embedding"])
    X = normalize(np.asarray(X, dtype=np.float64))
    return ids, X

def load_sequences(path, valid_ids):
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            traj = obj.get("trajectory", [])
            # business_ids extrahieren
            s = [
                int(step["business_id"])
                for step in traj
                if "business_id" in step and int(step["business_id"]) in valid_ids
            ]

            if len(s) >= 2:
                seqs.append(s)

    return seqs

# =========================
# CLUSTER UTILITIES
# =========================
def clusters_from_labels(ids, labels):
    cl = defaultdict(set)
    for pid, lab in zip(ids, labels):
        cl[int(lab)].add(pid)
    return cl

def jaccard_set(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0

def hungarian_jaccard_matching(cl1, cl2):
    I = list(cl1.keys())
    J = list(cl2.keys())

    cost = np.zeros((len(I), len(J)))
    for i, ci in enumerate(I):
        for j, cj in enumerate(J):
            cost[i, j] = 1.0 - jaccard_set(cl1[ci], cl2[cj])

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[J[c]] = I[r]

    # unmatched clusters in cl2 -> best overlap (many-to-one, explicitly allowed)
    for cj in J:
        if cj not in mapping:
            best_i = max(I, key=lambda ci: jaccard_set(cl1[ci], cl2[cj]))
            mapping[cj] = best_i

    return mapping  # cl2_label -> cl1_label

# =========================
# SEQUENCES → REGIONS
# =========================
def region_sequences(seqs, poi_to_label):
    out = []
    for s in seqs:
        r = []
        for p in s:
            if p in poi_to_label:
                lab = poi_to_label[p]
                if not r or r[-1] != lab:
                    r.append(lab)
        if len(r) >= 2:
            out.append(r)
    return out

# =========================
# TRANSITIONS
# =========================
def extract_pair_counts(region_seqs):
    pair = Counter()
    left = Counter()
    right = Counter()
    total = 0
    for s in region_seqs:
        for a, b in zip(s[:-1], s[1:]):
            pair[(a, b)] += 1
            left[a] += 1
            right[b] += 1
            total += 1
    return pair, left, right, total

def transition_metrics(region_seqs):
    pair, left, right, total = extract_pair_counts(region_seqs)

    lifts, counts = [], []
    self_loops = 0
    for (a, b), c in pair.items():
        lift = (c / total) / max((left[a] / total) * (right[b] / total), 1e-12)
        lifts.append(lift)
        counts.append(c)
        if a == b:
            self_loops += c

    counts = np.asarray(counts, dtype=np.float64)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12)) / math.log(len(probs))

    out_degree = Counter()
    for (a, b) in pair:
        out_degree[a] += 1

    n_nodes = max(len(left), 1)
    sparsity = len(pair) / (n_nodes ** 2)

    return {
        "pair_counts": pair,
        "pattern_count": len(pair),
        "lift_median": float(np.median(lifts)),
        "lift_q25": float(np.percentile(lifts, 25)),
        "lift_q75": float(np.percentile(lifts, 75)),
        "coverage": float(counts.sum() / total),
        "entropy": float(entropy),
        "self_loop_ratio": float(self_loops / counts.sum()),
        "sparsity": float(sparsity),
        "out_degree_median": float(np.median(list(out_degree.values())))
    }

# =========================
# STABILITY METRICS (FULL)
# =========================
def weighted_jaccard_full(a, b):
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    den = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return num / den if den > 0 else 0.0

def mass_loss(a, b):
    total_a = sum(a.values())
    total_b = sum(b.values())
    shared = sum(min(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b))
    return 1.0 - shared / max(total_a, total_b)

def spearman_all(a, b):
    keys = list(set(a) & set(b))
    if len(keys) < 5:
        return None
    return float(spearmanr([a[k] for k in keys], [b[k] for k in keys])[0])

def transition_distributions(pair_counts):
    by_left = defaultdict(Counter)
    for (a, b), c in pair_counts.items():
        by_left[a][b] += c

    dist = {}
    for a, ctr in by_left.items():
        s = sum(ctr.values())
        dist[a] = {b: c / s for b, c in ctr.items()}
    return dist

def js_divergence_union(d1, d2):
    keys = set(d1) | set(d2)
    vals = []
    for k in keys:
        p = d1.get(k, {})
        q = d2.get(k, {})
        all_b = sorted(set(p) | set(q))
        pv = np.array([p.get(b, 0.0) for b in all_b])
        qv = np.array([q.get(b, 0.0) for b in all_b])
        if pv.sum() == 0 or qv.sum() == 0:
            vals.append(1.0)
        else:
            vals.append(float(jensenshannon(pv, qv)))
    return float(np.median(vals)) if vals else None

# =========================
# MAIN
# =========================
def main():
    poi_ids, X = load_embeddings(EMB_PATH)
    seqs = load_sequences(SEQ_PATH, set(poi_ids))

    per_k = {}

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(
            "k,PatternCount,LiftMedian,LiftQ25,LiftQ75,"
            "Coverage,Entropy,SelfLoopRatio,Sparsity,OutDegreeMedian\n"
        )

        for k in K_VALUES:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X)
            poi_to_lab = dict(zip(poi_ids, labels))
            clusters = clusters_from_labels(poi_ids, labels)

            rseqs = region_sequences(seqs, poi_to_lab)
            m = transition_metrics(rseqs)

            per_k[k] = {
                "clusters": clusters,
                "pair_counts": m["pair_counts"]
            }

            f.write(
                f"{k},{m['pattern_count']},{m['lift_median']},"
                f"{m['lift_q25']},{m['lift_q75']},{m['coverage']},"
                f"{m['entropy']},{m['self_loop_ratio']},"
                f"{m['sparsity']},{m['out_degree_median']}\n"
            )

    with open(PAIRWISE_CSV, "w", encoding="utf-8") as f:
        f.write("k1,k2,WeightedJaccard,MassLoss,SpearmanAll,JS_Divergence\n")

        for k1, k2 in combinations(K_VALUES, 2):
            cl1 = per_k[k1]["clusters"]
            cl2 = per_k[k2]["clusters"]

            mapping = hungarian_jaccard_matching(cl1, cl2)

            p1 = per_k[k1]["pair_counts"]
            p2_raw = per_k[k2]["pair_counts"]
            p2 = Counter()
            for (a, b), c in p2_raw.items():
                p2[(mapping[a], mapping[b])] += c

            wj = weighted_jaccard_full(p1, p2)
            ml = mass_loss(p1, p2)
            rho = spearman_all(p1, p2)
            js = js_divergence_union(
                transition_distributions(p1),
                transition_distributions(p2)
            )

            f.write(f"{k1},{k2},{wj},{ml},{rho},{js}\n")

    print("Finished k-sensitivity analysis.")

if __name__ == "__main__":
    main()
