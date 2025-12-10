# src/preprocessing/poi_preprocessing/00_filter_media_years.py

import pandas as pd
from pathlib import Path
from utils.logging_config import setup_logging
import logging
import sys

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[3]

MEDIA_FILE = BASE_DIR / "Dataset" / "raw" / "media.csv"

OUT_DIR = BASE_DIR / "Dataset" / "processed" / "sequenz_processed"
OUT_FILE = OUT_DIR / "00_media_2018_2019.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# SHARED LOG FILE  (unique for sequence preprocessing)
# =====================================================
LOG_DIR = BASE_DIR / "Dataset" / "logs" / "preprocessing"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "02_log_sequenz_preprocessing.log"

# overwrite at STEP 00
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("All preprocessing output files are stored in: Dataset/processed/sequenz_processed\n")
    f.write("--------------------------------------------------------------------------------\n\n")

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    format="%(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

# scientific execution header
finish_logging = setup_logging("00_filter_media_years")

try:
    # =====================================================
    # LOAD
    # =====================================================
    df = pd.read_csv(MEDIA_FILE, low_memory=False)
    total_before = len(df)

    # =====================================================
    # TIMESTAMP PARSE
    # =====================================================
    df["timestamp_dt"] = pd.to_datetime(df["taken_at_timestamp"], errors="coerce", utc=True)
    df["year"] = df["timestamp_dt"].dt.year

    # =====================================================
    # FILTER 2018 + 2019
    # =====================================================
    df_out = df[df["year"].isin([2018, 2019])]
    total_after = len(df_out)

    reduction = (1 - total_after / total_before)

    # =====================================================
    # REMOVE TEMP COLS
    # =====================================================
    df_out = df_out.drop(columns=["timestamp_dt", "year"])

    # =====================================================
    # SAVE
    # =====================================================
    df_out.to_csv(OUT_FILE, index=False)

    # =====================================================
    # STEP 00 LOG ENTRY
    # =====================================================
    logging.info(
        f"[STEP 00] input={total_before:,} "
        f"filtered={total_after:,} "
        f"reduction={reduction:.2%} "
        f"saved={OUT_FILE.name}"
    )

except Exception:
    logging.exception("ERROR in STEP 00")
    sys.exit(1)

finish_logging()

print("Script: 00_filter_media_years executed")
