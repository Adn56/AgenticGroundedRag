# src/preprocessing/poi_preprocessing/01_filter_bodensee_confirmed.py

import sys, logging
import pandas as pd
from math import radians, sin, cos, sqrt, asin
from pathlib import Path
from utils.logging_config import setup_logging
from utils.save_csv import save_csv

# === Parameters ===
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_FILE  = BASE_DIR / "Dataset" / "processed" / "poi_processed" / "00_extended_locations_prepared.csv"
OUTPUT_DIR  = BASE_DIR / "Dataset" / "processed" / "poi_processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "01_confirmed_bodensee.csv"

LAT_BODENSEE, LON_BODENSEE, RADIUS_KM = 47.65, 9.40, 50.0

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# ============================================================
# Central shared logfile
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

finish_logging = setup_logging("01_filter_bodensee_confirmed")

try:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    total_input = len(df)

    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("Missing lat/lon columns")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    valid = df.dropna(subset=["lat", "lon"])
    valid = valid[(valid["lat"].between(-90, 90)) & (valid["lon"].between(-180, 180))]
    valid_coords = len(valid)

    valid["distance_km"] = valid.apply(
        lambda r: haversine(LAT_BODENSEE, LON_BODENSEE, r["lat"], r["lon"]),
        axis=1
    )

    filtered = valid[valid["distance_km"] <= RADIUS_KM]
    count_radius = len(filtered)
    pct = round((count_radius / total_input) * 100, 2)

    save_csv(filtered, OUTPUT_FILE)

    logging.info(
        f"[STEP 01] "
        f"input={total_input:,} "
        f"valid_coords={valid_coords:,} "
        f"radius{RADIUS_KM:.0f}km={count_radius:,} "
        f"pct={pct}% "
        f"saved={OUTPUT_FILE.name}"
    )

except Exception:
    logging.exception("ERROR in STEP 01")
    sys.exit(1)

finish_logging()

print("Script: 01_filter_bodensee_confirmed executed")
