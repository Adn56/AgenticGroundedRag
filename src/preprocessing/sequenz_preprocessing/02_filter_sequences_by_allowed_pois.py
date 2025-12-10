# src/preprocessing/poi_preprocessing/02_filter_sequences_allowed_pois.py

import pandas as pd
from pathlib import Path
import ast
import logging
from utils.logging_config import setup_logging
import sys

# ========================================
# PATHS
# ========================================
BASE_DIR = Path(__file__).resolve().parents[3]

OUT_DIR = BASE_DIR / "Dataset" / "processed" / "sequenz_processed"

RAW_SEQ_FILE     = OUT_DIR / "01_user_sequences_2018_2019.csv"
ALLOWED_POI_FILE = BASE_DIR / "Dataset" / "processed" / "poi_processed" / "03_poi_names_cleaned.csv"
OUTPUT_FILE      = OUT_DIR / "02_user_sequences_2018_2019_filtered.csv"

# ========================================
# LOGGING
# ========================================
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

finish_logging = setup_logging("02_filter_sequences_allowed_pois")


# ========================================
# PROCESSING
# ========================================
try:
    # -------------------------------------------------
    # LOAD SEQUENCES
    # -------------------------------------------------
    df_seq = pd.read_csv(RAW_SEQ_FILE, low_memory=False)
    seq_before = len(df_seq)

    # FAST PARSE
    df_seq["sequence"] = df_seq["sequence"].apply(ast.literal_eval)

    # -------------------------------------------------
    # LOAD ALLOWED POIs
    # -------------------------------------------------
    df_allowed = pd.read_csv(ALLOWED_POI_FILE, low_memory=False)

    allowed_set = set(df_allowed["location_id"].astype(str))

    # -------------------------------------------------
    # FILTER & SPLIT (ULTRA FAST)
    # -------------------------------------------------
    rows = []

    for idx, row in df_seq.iterrows():
        owner = row["owner_id"]
        seq   = row["sequence"]

        current_seq = []
        seq_id = 1

        for poi, ts, media in seq:
            poi = str(poi)

            if poi in allowed_set:
                current_seq.append((poi, ts, media))
            else:
                if current_seq:
                    rows.append([owner, seq_id, current_seq])
                    seq_id += 1
                    current_seq = []

        if current_seq:
            rows.append([owner, seq_id, current_seq])

    df_out = pd.DataFrame(rows, columns=["owner_id", "sequence_id", "sequence"])
    seq_after = len(df_out)

    # -------------------------------------------------
    # POI STATISTICS
    # -------------------------------------------------
    pois_before = sum(len(s) for s in df_seq["sequence"])
    pois_after  = sum(len(s) for s in df_out["sequence"])
    pois_removed = pois_before - pois_after
    reduction_pct = pois_removed / pois_before * 100 if pois_before > 0 else 0

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    # -------------------------------------------------
    # LOG ENTRY
    # -------------------------------------------------
    logging.info(
        f"[STEP 02] input_seq={seq_before:,} "
        f"input_pois={pois_before:,} "
        f"allowed_pois={len(allowed_set):,} "
        f"output_seq={seq_after:,} "
        f"output_pois={pois_after:,} "
        f"removed_pois={pois_removed:,} "
        f"reduction_pct={reduction_pct:.2f}% "
        f"saved={OUTPUT_FILE.name}"
    )

except Exception:
    logging.exception("ERROR in STEP 02")
    sys.exit(1)

finish_logging()
print("Script: 02_filter_sequences_allowed_pois executed")
