# src/preprocessing/poi_preprocessing/02_split_by_usercount.py

import sys
import pandas as pd
from pathlib import Path
import logging
from utils.logging_config import setup_logging
from utils.save_csv import save_csv

BASE_DIR = Path(__file__).resolve().parents[3]

POI_FILE   = BASE_DIR / "Dataset" / "processed" / "poi_processed" / "01_confirmed_bodensee.csv"
MEDIA_FILE = BASE_DIR / "Dataset" / "raw" / "media.csv"

OUTPUT_DIR   = BASE_DIR / "Dataset" / "processed" / "poi_processed"
OUTPUT_ALL   = OUTPUT_DIR / "02_confirmed_bodensee_with_users.csv"
OUTPUT_OVER0 = OUTPUT_DIR / "02_pois_usercount_over0.csv"
OUTPUT_ZERO  = OUTPUT_DIR / "02_pois_usercount_zero.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Central shared logfile
# ============================================================
LOG_DIR  = BASE_DIR / "Dataset" / "logs" / "preprocessing"
LOG_FILE = LOG_DIR / "poi_preprocessing.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    format="%(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

finish_logging = setup_logging("02_split_by_usercount")

try:
    df_poi = pd.read_csv(POI_FILE, low_memory=False)
    df_media = pd.read_csv(MEDIA_FILE, low_memory=False)

    input_poi = len(df_poi)
    input_media = len(df_media)

    if "location_id" not in df_media.columns or "owner_id" not in df_media.columns:
        raise ValueError("Missing location_id or owner_id in media.csv")

    df_media_small = df_media[["location_id", "owner_id"]].dropna().drop_duplicates()

    user_counts = (
        df_media_small
        .groupby("location_id")["owner_id"]
        .nunique()
        .reset_index()
        .rename(columns={"owner_id": "unique_user_count"})
    )

    user_stats = len(user_counts)

    df_merged = df_poi.merge(
        user_counts,
        how="left",
        left_on="id",
        right_on="location_id"
    )

    df_merged["unique_user_count"] = df_merged["unique_user_count"].fillna(0).astype(int)

    save_csv(df_merged, OUTPUT_ALL)

    df_over0 = df_merged[df_merged["unique_user_count"] > 0]
    df_zero  = df_merged[df_merged["unique_user_count"] == 0]

    over0 = len(df_over0)
    zero  = len(df_zero)

    save_csv(df_over0, OUTPUT_OVER0)
    save_csv(df_zero, OUTPUT_ZERO)

    logging.info(
        f"[STEP 02] "
        f"input={input_poi:,} "
        f"media={input_media:,} "
        f"userstats={user_stats:,} "
        f"over0={over0:,} "
        f"zero={zero:,} "
        f"saved_all={OUTPUT_ALL.name} "
        f"saved_over0={OUTPUT_OVER0.name} "
        f"saved_zero={OUTPUT_ZERO.name}"
    )

except Exception:
    logging.exception("ERROR in STEP 02")
    sys.exit(1)

finish_logging()

print("Script: 02_split_by_usercount executed")
