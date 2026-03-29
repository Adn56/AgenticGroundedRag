#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DISTANCE-BASED RE-RANKING (STORE TOP-20 PER USER)

Output:
CSV mit:
- user_id
- rank
- business_id
- distance_km
"""

import json
from pathlib import Path
import pandas as pd
import math

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

HYP_JSONL = PROJECT_ROOT / "results" / "hypothesis_test_eval_1000.jsonl"
RAG_CSV   = PROJECT_ROOT / "results" / "rag_hypothesis_test_eval_1000_top100.csv"
POI_STATES = DATASET_DIR / "core" / "poi_semantic_states.jsonl"

OUTPUT_CSV = PROJECT_ROOT / "results" / "distance_reranked_top20.csv"

TOP_K = 20

# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# ============================================================
# LOAD POI COORDINATES
# ============================================================

poi_coords = {}

with open(POI_STATES, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("status") == "ok":
            poi_coords[int(obj["business_id"])] = (
                obj["latitude"],
                obj["longitude"]
            )

# ============================================================
# LOAD RAG OUTPUT
# ============================================================

df_rag = pd.read_csv(RAG_CSV)

rag_grouped = (
    df_rag
    .sort_values(["user_id", "retrieval_rank"])
    .groupby("user_id")["business_id"]
    .apply(list)
    .to_dict()
)

# ============================================================
# LOAD SEQUENCES + TARGETS
# ============================================================

sequences = {}

with open(HYP_JSONL, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            uid = str(obj["user_id"])
            sequences[uid] = obj["sequence"]

# ============================================================
# DISTANCE RE-RANKING + STORE
# ============================================================

rows = []

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

    # --- sort ascending ---
    ranked = sorted(distances, key=lambda x: x[1])

    # --- store top 20 ---
    for rank, (bid, dist) in enumerate(ranked[:TOP_K], start=1):
        rows.append({
            "user_id": uid,
            "rank": rank,
            "business_id": bid,
            "distance_km": dist
        })

# ============================================================
# SAVE CSV
# ============================================================

df_out = pd.DataFrame(rows)
df_out.to_csv(OUTPUT_CSV, index=False)