#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DISTANCE-BASED RE-RANKING (TOP-15 from stored RAG candidates)

Pipeline per user:
1) Load RAG candidates
2) Take last POI from sequence
3) Compute Haversine distance to all candidates
4) Rank by distance (ascending)
5) Store Top-15
"""

import json
from pathlib import Path
import pandas as pd
import math
import csv

# ============================================================
# PATHS (UNCHANGED FROM YOUR ORIGINAL SCRIPT)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

HYP_JSONL = PROJECT_ROOT / "prompt_ablation" / "hypothesis_test_eval_1000_withTime.jsonl"
RAG_CSV   = PROJECT_ROOT / "prompt_ablation" / "rag_hypothesis_test_eval_1000_withTime_top50.csv"
POI_STATES = DATASET_DIR / "core" / "poi_semantic_states.jsonl"

OUTPUT_CSV = PROJECT_ROOT / "prompt_ablation" / "rag_hypothesis_test_eval_1000_withTime_top50_distance_top20.csv"

TOP_K = 20

# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))

# ============================================================
# LOAD POI COORDINATES
# ============================================================

print("Loading POI coordinates...")

poi_coords = {}

with open(POI_STATES, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("status") == "ok":
            lat = obj.get("latitude")
            lon = obj.get("longitude")

            if lat is not None and lon is not None:
                poi_coords[int(obj["business_id"])] = (lat, lon)

print(f"Loaded {len(poi_coords)} POIs with coordinates")

# ============================================================
# LOAD RAG RETRIEVAL OUTPUT
# ============================================================

print("Loading RAG candidates...")

df_rag = pd.read_csv(RAG_CSV)

rag_grouped = (
    df_rag
    .sort_values(["user_id", "retrieval_rank"])
    .groupby("user_id")["business_id"]
    .apply(list)
    .to_dict()
)

print(f"Loaded RAG candidates for {len(rag_grouped)} users")

# ============================================================
# LOAD SEQUENCES (FROM HYP FILE)
# ============================================================

print("Loading sequences...")

sequences = {}

with open(HYP_JSONL, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            uid = str(obj["user_id"])
            sequences[uid] = obj["sequence"]

print(f"Loaded sequences for {len(sequences)} users")

# ============================================================
# DISTANCE RE-RANKING
# ============================================================

print("Running distance re-ranking...")

rows_out = []

for uid, candidate_ids in rag_grouped.items():

    uid_str = str(uid)

    if uid_str not in sequences:
        continue

    sequence = sequences[uid_str]

    if not sequence:
        continue

    last_poi = sequence[-1]

    if last_poi not in poi_coords:
        continue

    lat1, lon1 = poi_coords[last_poi]

    distances = []

    # --- compute distances ---
    for bid in candidate_ids:

        bid_int = int(bid)

        if bid_int not in poi_coords:
            continue

        lat2, lon2 = poi_coords[bid_int]
        dist = haversine(lat1, lon1, lat2, lon2)

        distances.append((bid_int, dist))

    if not distances:
        continue

    # --- sort ascending (closest first) ---
    ranked = sorted(distances, key=lambda x: x[1])

    # --- store top K ---
    for rank, (bid, dist) in enumerate(ranked[:TOP_K], start=1):
        rows_out.append({
            "user_id": uid_str,
            "rank": rank,
            "business_id": bid,
            "distance_km": float(dist)
        })

# ============================================================
# SAVE OUTPUT
# ============================================================

print("Writing output...")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "user_id",
            "rank",
            "business_id",
            "distance_km"
        ]
    )
    writer.writeheader()
    writer.writerows(rows_out)

print(f"\nSaved to: {OUTPUT_CSV}")