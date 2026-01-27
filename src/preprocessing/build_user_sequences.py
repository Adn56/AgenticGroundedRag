#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 01 – USER TRAJECTORY CONSTRUCTION (PREPROCESSING)

- loads raw check-in data
- sorts POI visits per user by timestamp
- constructs explicit user trajectories
- writes one trajectory per user (JSONL)

INPUT:
Dataset/output_poi_preprocessing/checkin.csv

OUTPUT:
Dataset/output_poi_preprocessing/user_trajectories.jsonl
"""

import pandas as pd
import json
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "Dataset"

INPUT_CSV = DATASET_DIR / "output_poi_preprocessing" / "checkin.csv"
OUTPUT_JSONL = DATASET_DIR / "output_poi_preprocessing" / "user_trajectories.jsonl"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

# Parse timestamp
df["date"] = pd.to_datetime(df["date"], utc=True)

# Sort per user
df = df.sort_values(by=["user_id", "date"])

# ============================================================
# BUILD TRAJECTORIES
# ============================================================

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for user_id, group in df.groupby("user_id"):
        trajectory = [
            {
                "business_id": int(row["business_id"]),
                "date": row["date"].isoformat()
            }
            for _, row in group.iterrows()
        ]

        record = {
            "user_id": int(user_id),
            "trajectory": trajectory
        }

        f.write(json.dumps(record) + "\n")

print(f"✅ Wrote user trajectories to {OUTPUT_JSONL}")
