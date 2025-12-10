# src/preprocessing/poi_preprocessing/04_sequence_time_cleaning.py

import pandas as pd
import ast
from pathlib import Path
import logging
import time
from utils.logging_config import setup_logging
import sys

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / "Dataset" / "processed" / "sequenz_processed"

SEQ_FILE = OUT_DIR / "03_user_sequences_2018_2019_canonical.csv"
OUT_FILE = OUT_DIR / "04_user_sequences_2018_2019_1h_fast_cleaned.csv"

# =====================================================
# LOGGING
# =====================================================
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

finish_logging = setup_logging("04_sequence_time_cleaning")


# =====================================================
# PARAMETERS
# =====================================================
THRESH_SECONDS = 3600  # 1 hour

start_time = time.time()

try:
    # -------------------------------------------------
    # LOAD SEQUENCES
    # -------------------------------------------------
    df = pd.read_csv(SEQ_FILE, low_memory=False)
    df["sequence"] = df["sequence"].apply(ast.literal_eval)

    seq_before = df["sequence"].apply(len).sum()

    # -------------------------------------------------
    # ULTRA-FAST CLEANING with canonical_poi_id
    # -------------------------------------------------
    def clean_sequence(sequence):
        if len(sequence) <= 1:
            return sequence

        cleaned = [sequence[0]]

        for i in range(1, len(sequence)):
            poi_prev, ts_prev, media_prev, canon_prev = cleaned[-1]
            poi_curr, ts_curr, media_curr, canon_curr = sequence[i]

            # 🔥 NEU: canonical vergleichen, nicht poi_id
            if canon_prev != canon_curr:
                cleaned.append(sequence[i])
                continue

            # gleicher Canonical → Zeit prüfen
            if (ts_curr - ts_prev) > THRESH_SECONDS:
                cleaned.append(sequence[i])
            # sonst löschen

        return cleaned

    cleaned_rows = []
    for _, row in df.iterrows():
        cleaned_seq = clean_sequence(row["sequence"])
        cleaned_rows.append([row["owner_id"], row["sequence_id"], cleaned_seq])

    df_out = pd.DataFrame(cleaned_rows, columns=["owner_id", "sequence_id", "sequence"])

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------
    seq_after = df_out["sequence"].apply(len).sum()
    removed = seq_before - seq_after

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_FILE, index=False)

    runtime = round(time.time() - start_time, 2)

    logging.info(
        f"[STEP 03_FAST_CANON] before={seq_before:,} after={seq_after:,} removed={removed:,}"
    )

except Exception:
    logging.exception("ERROR in STEP 04 FAST CANONICAL CLEANING")
    sys.exit(1)

finish_logging()
print("Script: 04_sequence_time_duplicat_cleaning.py executed")
