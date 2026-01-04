# src/preprocessing/poi_preprocessing/04_canonical_merge.py

import pandas as pd
from pathlib import Path
from utils.logging_config import setup_logging
import logging

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[3]

INPUT = BASE_DIR / "Dataset" / "processed" / "poi_processed" / "03_poi_names_cleaned.csv"
OUTPUT_CANONICAL = BASE_DIR / "Dataset" / "processed" / "poi_processed" / "04_poi_names_cleaned_with_canonical.csv"
OUTPUT_MAPPING = BASE_DIR / "Dataset" / "processed" / "poi_processed" / "04_poi_canonical_mapping.csv"

OUTPUT_DIR = BASE_DIR / "Dataset" / "processed" / "poi_processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Shared central logfile
# ============================================================
LOG_DIR = BASE_DIR / "Dataset" / "logs" / "preprocessing"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "poi_preprocessing.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    format="%(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

finish_logging = setup_logging("04_canonical_merge")

# ============================================================
# Processing
# ============================================================
df = pd.read_csv(INPUT, low_memory=False)
input_total = len(df)

# group by name + coordinates → canonical clusters
grouped = df.groupby(["name", "lat", "lon"])
duplicate_groups = {g: group for g, group in grouped if len(group) > 1}

num_groups = len(duplicate_groups)

df["canonical_poi_id"] = df["id"]
mapping_rows = []

# build canonical mapping
for (name, lat, lon), group in duplicate_groups.items():
    canonical_id = group["id"].min()
    all_ids = list(group["id"])

    for poi_id in all_ids:
        mapping_rows.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "poi_id": poi_id,
            "canonical_poi_id": canonical_id
        })

    df.loc[df["id"].isin(all_ids), "canonical_poi_id"] = canonical_id

mapping_df = pd.DataFrame(mapping_rows)
mapped_total = len(mapping_df)

df.to_csv(OUTPUT_CANONICAL, index=False)
mapping_df.to_csv(OUTPUT_MAPPING, index=False)

# ============================================================
# STEP 04 log entry (compact, english)
# ============================================================
logging.info(
    f"[STEP 04] "
    f"input={input_total:,} "
    f"groups={num_groups:,} "
    f"mapped={mapped_total:,} "
    f"saved_clean={OUTPUT_CANONICAL.name} "
    f"saved_map={OUTPUT_MAPPING.name}"
)

finish_logging()

print("Script: 04_canonical_merge.py executed")