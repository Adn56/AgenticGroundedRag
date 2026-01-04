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

    before_sequences = len(df)
    total_entries_before = int(np.sum([len(seq) for seq in df["sequence"]]))

    df = df[df["sequence"].apply(lambda seq: len(seq) > 1)]

    after_sequences = len(df)
    total_entries_after = int(np.sum([len(seq) for seq in df["sequence"]]))

    removed_sequences = before_sequences - after_sequences
    removed_entries = total_entries_before - total_entries_after

    df.to_csv(OUTPUT_FILE, index=False)

    logging.info(
        f"[STEP 06] sequences_before={before_sequences:,} "
        f"sequences_after={after_sequences:,} "
        f"removed_sequences={removed_sequences:,} "
        f"entries_before={total_entries_before:,} "
        f"entries_after={total_entries_after:,} "
        f"removed_entries={removed_entries:,} "
        f"saved={OUTPUT_FILE.name}"
    )

except Exception:
    logging.exception("ERROR in STEP 06")
    sys.exit(1)

finish_logging()
print("Script: 06_remove_length1_sequences.py executed")