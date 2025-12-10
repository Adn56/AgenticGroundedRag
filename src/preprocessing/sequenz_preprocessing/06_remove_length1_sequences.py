# src/preprocessing/poi_preprocessing/06_remove_length1_sequences.py

import pandas as pd
import ast
import numpy as np
from pathlib import Path
import logging
from utils.logging_config import setup_logging
import sys

BASE_DIR = Path(__file__).resolve().parents[3]

OUT_DIR = BASE_DIR / "Dataset" / "processed" / "sequenz_processed"
INPUT_FILE = OUT_DIR / "05_user_sequences_speed_cleaned.csv"
OUTPUT_FILE = OUT_DIR / "06_user_sequences_2018_2019_filtered_len2plus.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "Dataset" / "logs" / "preprocessing"
LOG_FILE = LOG_DIR / "02_log_sequenz_preprocessing.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    format="%(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

finish_logging = setup_logging("06_remove_length1_sequences")

try:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["sequence"] = df["sequence"].apply(ast.literal_eval)

    before = len(df)

    seq_lengths_before = [len(s) for s in df["sequence"]]
    avg_before = np.mean(seq_lengths_before) if before > 0 else 0
    median_before = np.median(seq_lengths_before) if before > 0 else 0

    df = df[df["sequence"].apply(lambda s: len(s) > 1)]
    after = len(df)

    seq_lengths_after = [len(s) for s in df["sequence"]]
    avg_after = np.mean(seq_lengths_after) if after > 0 else 0
    median_after = np.median(seq_lengths_after) if after > 0 else 0

    removed = before - after

    df.to_csv(OUTPUT_FILE, index=False)

    logging.info(
        f"[STEP 06] seq_before={before:,} "
        f"seq_after={after:,} "
        f"removed={removed:,} "
        f"avg_len={avg_before:.2f}->{avg_after:.2f} "
        f"median_len={median_before:.2f}->{median_after:.2f} "
        f"saved={OUTPUT_FILE.name}"
    )

except Exception:
    logging.exception("ERROR in STEP 06")
    sys.exit(1)

finish_logging()
print("Script: 06_remove_length1_sequences.py executed")