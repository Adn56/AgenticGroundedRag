#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DETERMINISTIC SUBSAMPLING FOR EVALUATION / DEVELOPMENT (LLM INPUT LEVEL)

- Test  -> fixed evaluation subset (N_TEST_EVAL)
- Val   -> fixed development subset (N_VAL_DEV)
- Stratified by INPUT sequence length (quantiles) IF POSSIBLE
- Deterministic (fixed seed)
- Leakage-free
- Reviewer-safe

IMPORTANT:
Sampling is performed on *_last9_input.jsonl (model input representation),
NOT on raw trajectory splits.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

SEED = 2026
N_STRATA = 4

N_TEST_EVAL = 1000
N_VAL_DEV  = 100

rng = np.random.default_rng(SEED)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "Dataset" / "splits"

TEST_IN = SPLIT_DIR / "test_last9_input.jsonl"
VAL_IN  = SPLIT_DIR / "validation_last9_input.jsonl"

TEST_OUT = SPLIT_DIR / "test_eval_1000.jsonl"
VAL_OUT  = SPLIT_DIR / "validation_dev_100.jsonl"

# ============================================================
# IO
# ============================================================

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ============================================================
# DATAFRAME PREPARATION (LLM INPUT LEVEL)
# ============================================================

def to_df(data):
    return pd.DataFrame({
        "idx": range(len(data)),
        "seq_len": [len(d["input"]) for d in data]
    })

# ============================================================
# SAMPLING (ROBUST)
# ============================================================

def stratified_sample(df: pd.DataFrame, n_samples: int, n_strata: int):
    """
    Deterministic sampling.
    Falls back to uniform random sampling if stratification is impossible.
    """

    # --------------------------------------------------------
    # CASE 1: No variance in sequence length -> uniform sampling
    # --------------------------------------------------------
    if df.seq_len.nunique() == 1:
        sampled = rng.choice(df.idx, size=n_samples, replace=False)
        return sorted(sampled.tolist())

    # --------------------------------------------------------
    # CASE 2: Stratified sampling by seq_len quantiles
    # --------------------------------------------------------
    df = df.copy()

    df["stratum"] = pd.qcut(
        df.seq_len,
        q=n_strata,
        labels=False,
        duplicates="drop"
    )

    sizes = df.groupby("stratum").size()
    fractions = n_samples * sizes / sizes.sum()

    base = np.floor(fractions).astype(int)
    remainder = fractions - base

    missing = n_samples - base.sum()

    for s in remainder.sort_values(ascending=False).index[:missing]:
        base.loc[s] += 1

    sampled_idx = []

    for s, n in base.items():
        pool = df[df.stratum == s]
        sampled = rng.choice(pool.idx, size=n, replace=False)
        sampled_idx.extend(sampled.tolist())

    # Defensive check
    if len(sampled_idx) != n_samples:
        raise RuntimeError(
            f"Sampling failed: expected {n_samples}, got {len(sampled_idx)}"
        )

    return sorted(sampled_idx)

# ============================================================
# RUN
# ============================================================

def main():
    print("\nDETERMINISTIC SUBSAMPLING (LLM INPUT LEVEL)\n")

    test_data = load_jsonl(TEST_IN)
    val_data  = load_jsonl(VAL_IN)

    print(f"[OK] Loaded test inputs       : {len(test_data)}")
    print(f"[OK] Loaded validation inputs : {len(val_data)}\n")

    test_df = to_df(test_data)
    val_df  = to_df(val_data)

    test_eval_idx = stratified_sample(test_df, N_TEST_EVAL, N_STRATA)
    val_dev_idx   = stratified_sample(val_df,  N_VAL_DEV,  N_STRATA)

    write_jsonl(TEST_OUT, [test_data[i] for i in test_eval_idx])
    write_jsonl(VAL_OUT,  [val_data[i]  for i in val_dev_idx])

    print("[DONE]")
    print(f"Test evaluation subset : {len(test_eval_idx)} → {TEST_OUT.name}")
    print(f"Validation dev subset  : {len(val_dev_idx)}  → {VAL_OUT.name}")
    print(f"Seed                  : {SEED}\n")

if __name__ == "__main__":
    main()
