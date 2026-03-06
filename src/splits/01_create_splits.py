#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USER TRAJECTORY SPLIT: TRAIN / VALIDATION / TEST

- per user: last POI -> TEST
- per validation user: second-to-last POI -> VALIDATION
- TRAIN never contains validation or test POIs
- exactly N_VALIDATION validation users
- stratified by sequence length (quantiles)
- deterministic, leakage-free
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

N_VALIDATION = 1655
MIN_SEQ_LEN = 10
N_STRATA = 4
SEED = 2026

rng = np.random.default_rng(SEED)

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD & SORT
# ============================================================

df = pd.read_csv(INPUT_CSV)
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values(["user_id", "date"])

# ============================================================
# BUILD USER TRAJECTORIES
# ============================================================

users = []

for uid, g in df.groupby("user_id"):
    traj = [
        {"business_id": int(r.business_id), "date": r.date.isoformat()}
        for r in g.itertuples()
    ]
    if len(traj) >= 2:
        users.append({
            "user_id": int(uid),
            "trajectory": traj,
            "seq_len": len(traj)
        })

users_df = pd.DataFrame(users)

# ============================================================
# SELECT VALIDATION USERS (STRATIFIED, EXACT)
# ============================================================

eligible = users_df[users_df.seq_len > MIN_SEQ_LEN].copy()

eligible["stratum"] = pd.qcut(
    eligible.seq_len,
    q=N_STRATA,
    labels=False,
    duplicates="drop"
)

# ideal fractional allocation
stratum_sizes = eligible.groupby("stratum").size()
fractions = N_VALIDATION * stratum_sizes / stratum_sizes.sum()

base = np.floor(fractions).astype(int)
remainder = fractions - base

missing = N_VALIDATION - base.sum()

# distribute remainder deterministically
order = remainder.sort_values(ascending=False).index.tolist()
for s in order[:missing]:
    base.loc[s] += 1

# sample per stratum
val_parts = []
for s, n in base.items():
    pool = eligible[eligible.stratum == s]
    assert n <= len(pool)
    val_parts.append(
        pool.sample(n=n, random_state=SEED)
    )

val_df = pd.concat(val_parts)
assert len(val_df) == N_VALIDATION

val_user_ids = set(val_df.user_id)

# ============================================================
# WRITE SPLITS
# ============================================================

with open(TRAIN_OUT, "w") as f_train, \
        open(VAL_OUT, "w") as f_val, \
        open(TEST_OUT, "w") as f_test:

    for u in users:
        traj = u["trajectory"]

        # TEST
        f_test.write(json.dumps({
            "user_id": u["user_id"],
            "trajectory": traj[:-1],
            "target": traj[-1]
        }) + "\n")

        train_items = traj[:-1]

        # VALIDATION
        if u["user_id"] in val_user_ids:
            f_val.write(json.dumps({
                "user_id": u["user_id"],
                "trajectory": train_items[:-1],
                "target": train_items[-1]
            }) + "\n")
            train_items = train_items[:-1]

        # TRAIN
        f_train.write(json.dumps({
            "user_id": u["user_id"],
            "trajectory": train_items
        }) + "\n")

# ============================================================
# SUMMARY
# ============================================================

print("Finished splitting trajectories")
print(f"Train users      : {len(users)}")
print(f"Validation users : {len(val_user_ids)} (exact)")
print(f"Seed             : {SEED}")
