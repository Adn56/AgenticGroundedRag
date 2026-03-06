#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DETERMINISTIC SUBSAMPLING FOR EVALUATION / DEVELOPMENT (LLM INPUT LEVEL)

- Test  -> fixed evaluation subset (N_TEST_EVAL), single seed
- Val   -> multiple fixed development subsets (N_VAL_DEV), multiple seeds
- Uniform deterministic subsampling on fixed-length model inputs
- Leakage-free

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

TEST_SEED = 2026
VAL_SEEDS = [2026, 2027, 2028]   # <-- 3 dev subsamples for Top-K stability

N_TEST_EVAL = 1000
N_VAL_DEV  = 100

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "Dataset" / "splits"

TEST_IN = SPLIT_DIR / "test_last9_input.jsonl"
VAL_IN  = SPLIT_DIR / "validation_last9_input.jsonl"

TEST_OUT = SPLIT_DIR / "test_eval_1000.jsonl"
VAL_OUT_TEMPLATE = SPLIT_DIR / "validation_dev_100_seed{seed}.jsonl"

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
    # All inputs have fixed length (=9), kept only for indexing
    return pd.DataFrame({
        "idx": range(len(data))
    })

# ============================================================
# SAMPLING
# ============================================================

def uniform_sample(df: pd.DataFrame, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    sampled = rng.choice(df.idx, size=n_samples, replace=False)
    return sorted(sampled.tolist())

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

    # --------------------------------------------------------
    # TEST SUBSET (single, fixed)
    # --------------------------------------------------------
    test_eval_idx = uniform_sample(test_df, N_TEST_EVAL, TEST_SEED)
    write_jsonl(TEST_OUT, [test_data[i] for i in test_eval_idx])

    print(f"[TEST]  {len(test_eval_idx)} samples → {TEST_OUT.name}")
    print(f"[TEST]  Seed: {TEST_SEED}\n")

    # --------------------------------------------------------
    # VALIDATION DEV SUBSETS (multiple seeds)
    # --------------------------------------------------------
    for seed in VAL_SEEDS:
        val_dev_idx = uniform_sample(val_df, N_VAL_DEV, seed)
        out_path = Path(str(VAL_OUT_TEMPLATE).format(seed=seed))
        write_jsonl(out_path, [val_data[i] for i in val_dev_idx])

        print(f"[VAL]   {len(val_dev_idx)} samples → {out_path.name} (seed={seed})")

    print("\n[DONE]")

if __name__ == "__main__":
    main()
