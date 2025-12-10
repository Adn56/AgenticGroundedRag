# src/preprocessing/poi_preprocessing/05_speed_based_cleaning.py

import pandas as pd
import ast
from pathlib import Path
import math
import logging
from utils.logging_config import setup_logging

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[3]

SEQ_FILE = BASE_DIR / "Dataset" / "processed" / "sequenz_processed" / "04_user_sequences_2018_2019_1h_fast_cleaned.csv"
POI_FILE = BASE_DIR / "Dataset" / "processed" / "poi_processed" / "04_poi_names_cleaned_with_canonical.csv"

OUT_FILE = BASE_DIR / "Dataset" / "processed" / "sequenz_processed" / "05_user_sequences_speed_cleaned.csv"

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

finish_logging = setup_logging("05_speed_cleaning")


# =====================================================
# HAVERSINE FUNCTION
# =====================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


# =====================================================
# LOAD POI COORDS using location_id
# =====================================================
poi_df = pd.read_csv(POI_FILE, low_memory=False)

poi_df["location_id"] = poi_df["location_id"].astype(str)

poi_coord = dict(
    zip(
        poi_df["location_id"],
        zip(poi_df["lat"], poi_df["lon"])
    )
)


# =====================================================
# LOAD SEQUENCES
# =====================================================
df = pd.read_csv(SEQ_FILE, low_memory=False)
df["sequence"] = df["sequence"].apply(ast.literal_eval)

seq_before = df["sequence"].apply(len).sum()


# =====================================================
# PARAMETERS
# =====================================================
MAX_SPEED = 150  # km/h


# =====================================================
# SPEED CLEANING FUNCTION
# =====================================================
def clean_speed(sequence):
    if len(sequence) <= 1:
        return sequence

    cleaned = [sequence[0]]

    for i in range(1, len(sequence)):
        prev = cleaned[-1]
        curr = sequence[i]

        # Struktur:
        # (location_id, ts, media_id, canonical)
        loc_prev, ts_prev, media_prev, canon_prev = prev
        loc_curr, ts_curr, media_curr, canon_curr = curr

        loc_prev = str(loc_prev)
        loc_curr = str(loc_curr)

        # Zeit
        dt = ts_curr - ts_prev
        if dt <= 0:
            cleaned.append(curr)
            continue

        dt_h = dt / 3600

        # Koordinaten lookup
        if loc_prev not in poi_coord or loc_curr not in poi_coord:
            cleaned.append(curr)
            continue

        lat1, lon1 = poi_coord[loc_prev]
        lat2, lon2 = poi_coord[loc_curr]

        dist = haversine(lat1, lon1, lat2, lon2)
        speed = dist / dt_h

        # Too fast → remove
        if speed > MAX_SPEED:
            # remove curr
            continue

        cleaned.append(curr)

    return cleaned


# =====================================================
# APPLY CLEANING
# =====================================================
cleaned_rows = []
for _, row in df.iterrows():
    cleaned_seq = clean_speed(row["sequence"])
    cleaned_rows.append([row["owner_id"], row["sequence_id"], cleaned_seq])

df_out = pd.DataFrame(cleaned_rows, columns=["owner_id", "sequence_id", "sequence"])

seq_after = df_out["sequence"].apply(len).sum()
removed = seq_before - seq_after

# SAVE
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT_FILE, index=False)

logging.info(f"[STEP 05] before={seq_before:,} after={seq_after:,} removed={removed:,}")

finish_logging()

print("Script: 05_speed_based_cleaning.py executed")

