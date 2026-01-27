#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USER TRAJECTORY SPLIT: TRAIN / VALIDATION / TEST

Rules:
- per user: last POI -> TEST
- per validation user: second-to-last POI -> VALIDATION
- TRAIN never contains validation or test POIs
- validation users:
    * exactly N_VALIDATION users
    * sequence length > MIN_SEQ_LEN
    * stratified by sequence length (quantiles)
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

N_VALIDATION = 200
MIN_SEQ_LEN = 10
N_STRATA = 4
SEED = 2026

np.random.seed(SEED)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "Dataset"

INPUT_CSV = DATASET_DIR / "output_poi_preprocessing" / "checkin.csv"
OUTPUT_DIR = DATASET_DIR / "splits"

TRAIN_OUT = OUTPUT_DIR / "train.jsonl"
VAL_OUT = OUTPUT_DIR / "validation.jsonl"
TEST_OUT = OUTPUT_DIR / "test.jsonl"

# ============================================================
# LOAD & SORT DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values(by=["user_id", "date"])

# ============================================================
# BUILD USER TRAJECTORIES
# ============================================================

users = []

for user_id, group in df.groupby("user_id"):
    traj = [
        {
            "business_id": int(row["business_id"]),
            "date": row["date"].isoformat()
        }
        for _, row in group.iterrows()
    ]

    if len(traj) < 2:
        continue

    users.append({
        "user_id": int(user_id),
        "trajectory": traj,
        "seq_len": len(traj)
    })

users_df = pd.DataFrame(users)

# ============================================================
# SELECT VALIDATION USERS (STRATIFIED)
# ============================================================

eligible = users_df[users_df["seq_len"] > MIN_SEQ_LEN].copy()

eligible["stratum"] = pd.qcut(
    eligible["seq_len"],
    q=N_STRATA,
    labels=False,
    duplicates="drop"
)

val_parts = []

for _, stratum_df in eligible.groupby("stratum"):
    n = int(round(N_VALIDATION * len(stratum_df) / len(eligible)))
    val_parts.append(
        stratum_df.sample(
            n=min(n, len(stratum_df)),
            random_state=SEED
        )
    )

val_df = pd.concat(val_parts).sample(
    n=min(N_VALIDATION, len(eligible)),
    random_state=SEED
)

val_user_ids = set(val_df["user_id"])

# ============================================================
# WRITE SPLITS
# ============================================================

with open(TRAIN_OUT, "w", encoding="utf-8") as f_train, \
        open(VAL_OUT, "w", encoding="utf-8") as f_val, \
        open(TEST_OUT, "w", encoding="utf-8") as f_test:

    for u in users:
        traj = u["trajectory"]

        test_item = traj[-1]
        train_items = traj[:-1]

        # TEST (always)
        f_test.write(json.dumps({
            "user_id": u["user_id"],
            "trajectory": train_items,
            "target": test_item
        }) + "\n")

        # VALIDATION (only selected users)
        if u["user_id"] in val_user_ids:
            val_item = train_items[-1]
            train_items = train_items[:-1]

            f_val.write(json.dumps({
                "user_id": u["user_id"],
                "trajectory": train_items,
                "target": val_item
            }) + "\n")

        # TRAIN (clean context only)
        f_train.write(json.dumps({
            "user_id": u["user_id"],
            "trajectory": train_items
        }) + "\n")

print("✅ Finished splitting trajectories")
print(f"Train users: {len(users)}")
print(f"Validation users: {len(val_user_ids)}")
print(f"Seed used: {SEED}")
