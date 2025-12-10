# src/preprocessing/poi_preprocessing/03_add_canonical_ids_to_sequences.py

import pandas as pd
import ast
from pathlib import Path
import logging
from utils.logging_config import setup_logging
import sys

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[3]

OUT_DIR = BASE_DIR / "Dataset" / "processed" / "sequenz_processed"

INPUT_FILE = OUT_DIR / "02_user_sequences_2018_2019_filtered.csv"
CANONICAL_FILE = BASE_DIR / "Dataset" / "processed" / "poi_processed" / "04_poi_names_cleaned_with_canonical.csv"
OUTPUT_FILE = OUT_DIR / "03_user_sequences_2018_2019_canonical.csv"

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

finish_logging = setup_logging("03_add_canonical_ids")


try:
    # -------------------------------------------------
    # LOAD SEQUENCES
    # -------------------------------------------------
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["sequence"] = df["sequence"].apply(ast.literal_eval)

    # -------------------------------------------------
    # LOAD CANONICAL MAP
    # -------------------------------------------------
    df_can = pd.read_csv(CANONICAL_FILE, low_memory=False)

    # Wichtig: POI-ID = location_id
    df_can["location_id"] = df_can["location_id"].astype(str)
    df_can["canonical_poi_id"] = df_can["canonical_poi_id"].astype(str)

    canonical_map = dict(
        zip(df_can["location_id"], df_can["canonical_poi_id"])
    )

    # -------------------------------------------------
    # ADD canonical_poi_id to each triplet
    # -------------------------------------------------
    out_rows = []

    for _, row in df.iterrows():
        owner = row["owner_id"]
        seq_id = row["sequence_id"]
        seq = row["sequence"]

        new_seq = []

        for poi, ts, media in seq:
            poi_str = str(poi)
            canonical = canonical_map.get(poi_str)

            new_seq.append((poi_str, ts, media, canonical))

        out_rows.append([owner, seq_id, new_seq])

    df_out = pd.DataFrame(out_rows, columns=["owner_id", "sequence_id", "sequence"])

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    logging.info(
        f"[STEP 02.5] canonical IDs added for {len(df_out):,} sequences → {OUTPUT_FILE.name}"
    )

except Exception:
    logging.exception("ERROR in STEP 02.5 canonical mapping")
    sys.exit(1)

finish_logging()
print("Script: 03_add_canonical_ids_to_sequences executed")
