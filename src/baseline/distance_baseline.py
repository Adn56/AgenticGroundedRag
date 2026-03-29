#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================


BASE= Path(__file__).resolve().parents[2]

POI_FILE = BASE / "Dataset" / "core" / "poi_semantic_states.jsonl"
TEST_FILE = BASE / "Dataset" / "splits" / "test_eval_1000.jsonl"

OUTPUT_CSV = BASE / "baseline" / "distance_baseline_top20.csv"

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
# LOAD POI COORDS
# ============================================================

poi_coords = {}

with open(POI_FILE, "r") as f:
    for line in f:
        obj = json.loads(line)
        bid = obj["business_id"]
        lat = obj.get("latitude")
        lon = obj.get("longitude")

        if lat is not None and lon is not None:
            poi_coords[bid] = (lat, lon)

print(f"Loaded {len(poi_coords)} POIs with coordinates")


# ============================================================
# PROCESS USERS
# ============================================================

rows = []

with open(TEST_FILE, "r") as f:
    for line in tqdm(f, desc="Processing users"):
        obj = json.loads(line)

        user_id = obj["user_id"]
        sequence = obj["input"]

        if not sequence:
            continue

        # 👉 letzter POI
        last_poi = sequence[-1]["business_id"]

        if last_poi not in poi_coords:
            continue

        lat1, lon1 = poi_coords[last_poi]

        distances = []

        for poi_id, (lat2, lon2) in poi_coords.items():
            if poi_id == last_poi:
                continue

            dist = haversine(lat1, lon1, lat2, lon2)
            distances.append((poi_id, dist))

        # 👉 sortieren nach Distanz (aufsteigend)
        distances.sort(key=lambda x: x[1])

        # 👉 Top-K
        top_k = distances[:TOP_K]

        for rank, (poi_id, dist) in enumerate(top_k, start=1):
            rows.append({
                "user_id": user_id,
                "rank": rank,
                "business_id": poi_id,
                "distance_km": dist
            })


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved to: {OUTPUT_CSV}")