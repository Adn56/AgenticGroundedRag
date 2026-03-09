#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-NN BASELINE FOR NEXT-POI PREDICTION

Works with trajectory-style train.jsonl

Train trajectory:
[POI1 POI2 POI3 ... POIn]

Converted to training samples:
[POI1..POI9] -> POI10

Similarity:
Normalized positional overlap (sequence matching)
sim(A,B) = (1/L) * sum_{i=1..L} I(A_i == B_i)
"""

import json
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

K = 50
TOP_L = 10
SEQ_LEN = 9

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

TRAIN_PATH = DATASET_DIR / "splits" / "train.jsonl"
TEST_PATH  = DATASET_DIR / "splits" / "test_eval_1000.jsonl"

OUT_PATH = PROJECT_ROOT / "baseline" / "knn_predictions.jsonl"


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def similarity(seq_a, seq_b):
    """
    Position-based sequence similarity.

    Computes normalized positional overlap between
    two sequences of equal length.

    sim(A,B) = (1/L) * sum_{i=1..L} I(A_i == B_i)
    """

    L = min(len(seq_a), len(seq_b))

    if L == 0:
        return 0.0

    matches = 0

    for i in range(L):
        if seq_a[i] == seq_b[i]:
            matches += 1

    return matches / L


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train_data = load_jsonl(TRAIN_PATH)
test_data  = load_jsonl(TEST_PATH)

# ============================================================
# BUILD TRAIN SEQUENCES
# ============================================================

train_sequences = []

for user in train_data:

    traj = user["trajectory"]

    poi_ids = [int(v["business_id"]) for v in traj]

    if len(poi_ids) <= SEQ_LEN:
        continue

    for i in range(len(poi_ids) - SEQ_LEN):

        seq = poi_ids[i:i+SEQ_LEN]
        target = poi_ids[i+SEQ_LEN]

        train_sequences.append((seq, target))

print("Generated training samples:", len(train_sequences))
print("Test sequences:", len(test_data))


# ============================================================
# KNN
# ============================================================

with open(OUT_PATH, "w", encoding="utf-8") as out:

    for seq in test_data:

        user_id = seq["user_id"]
        test_seq = [int(v["business_id"]) for v in seq["input"]]

        scores = []

        for train_seq, train_target in train_sequences:

            sim = similarity(test_seq, train_seq)

            scores.append((sim, train_target))

        scores.sort(reverse=True)

        neighbours = scores[:K]

        targets = [t for _, t in neighbours]

        counts = Counter(targets)

        ranked = [t for t, _ in counts.most_common(TOP_L)]

        out.write(json.dumps({
            "user_id": user_id,
            "prediction": ranked,
            "target": int(seq["target"]["business_id"])
        }) + "\n")

        print("processed user:", user_id)

print("\nDONE")