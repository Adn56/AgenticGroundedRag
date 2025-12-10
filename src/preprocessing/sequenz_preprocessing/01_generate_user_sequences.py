# src/preprocessing/poi_preprocessing/01_create_user_sequences.py

import pandas as pd
from pathlib import Path
import logging
from utils.logging_config import setup_logging
import sys

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / "Dataset" / "processed" / "sequenz_processed"

INPUT_FILE  = OUT_DIR / "00_media_2018_2019.csv"
OUTPUT_FILE = OUT_DIR / "01_user_sequences_2018_2019.csv"

# =====================================================
# SHARED LOGFILE
# =====================================================
LOG_DIR  = BASE_DIR / "Dataset" / "logs" / "preprocessing"
LOG_FILE = LOG_DIR / "02_log_sequenz_preprocessing.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    format="%(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

finish_logging = setup_logging("01_create_user_sequences")

try:
    # =====================================================
    # LOAD CSV (FAST)
    # =====================================================
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    input_rows = len(df)

    # =====================================================
    # SELECT ONLY NEEDED COLUMNS
    # =====================================================
    df = df[["owner_id", "location_id", "id", "taken_at_unix_timestamp"]]

    # convert to correct types (FAST)
    df["location_id"] = df["location_id"].astype(str)
    df["id"] = df["id"].astype(str)
    df["taken_at_unix_timestamp"] = df["taken_at_unix_timestamp"].astype(int)

    # =====================================================
    # SORT (FAST)
    # =====================================================
    df = df.sort_values(
        by=["owner_id", "taken_at_unix_timestamp"],
        ascending=True
    )

    # =====================================================
    # BUILD TRIPLE SEQUENCES (ULTRA FAST!)
    # Format: [poi_id, timestamp, media_id]
    # =====================================================
    def build_sequence(g):
        return list(zip(
            g["location_id"],             # poi_id
            g["taken_at_unix_timestamp"], # timestamp
            g["id"],                      # media_id
        ))

    user_sequences = (
        df.groupby("owner_id")[["location_id", "taken_at_unix_timestamp", "id"]]
        .apply(build_sequence)
        .reset_index()
        .rename(columns={0: "sequence"})
    )

    seq_users = len(user_sequences)

    # =====================================================
    # SAVE
    # =====================================================
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    user_sequences.to_csv(OUTPUT_FILE, index=False)

    # =====================================================
    # LOG ENTRY
    # =====================================================
    logging.info(
        f"[STEP 01] input={input_rows:,} "
        f"users={seq_users:,} "
        f"saved={OUTPUT_FILE.name}"
    )

except Exception:
    logging.exception("ERROR in STEP 01")
    sys.exit(1)

finish_logging()

print("Script: 01_create_user_sequences executed")
