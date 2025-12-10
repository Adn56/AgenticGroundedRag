# src/preprocessing/poi_preprocessing/00_clean_address.py

import csv
import sys
import pandas as pd
import json
from pathlib import Path
from utils.logging_config import setup_logging
import logging
import time

BASE_DIR = Path(__file__).resolve().parents[3]
LOG_DIR = BASE_DIR / "Dataset" / "logs" / "preprocessing"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "poi_preprocessing.log"

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("All preprocessing output files are stored in: Dataset/processed/poi_processed\n")
    f.write("--------------------------------------------------------------------------------\n")

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    format="%(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

finish_logging = setup_logging("00_clean_address")
start = time.time()

try:
    RAW_DIR = BASE_DIR / "Dataset" / "raw"
    OUTPUT_DIR = BASE_DIR / "Dataset" / "processed" / "poi_processed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_FILE = RAW_DIR / "extended_locations.csv"
    CITIES_FILE = RAW_DIR / "cities.csv"
    OUTPUT_FILE = OUTPUT_DIR / "00_extended_locations_prepared.csv"

    extended = pd.read_csv(INPUT_FILE, low_memory=False)
    cities = pd.read_csv(CITIES_FILE, low_memory=False)

    total_extended = len(extended)

    def safe_parse(val):
        if pd.isna(val) or not isinstance(val, str) or val.strip() == "":
            return {}
        try:
            return json.loads(val)
        except:
            return {}

    if "address_json" in extended.columns:
        address_df = extended["address_json"].apply(safe_parse).apply(pd.Series)
        extended = extended.drop(columns=["address_json"])
        extended = pd.concat([extended, address_df], axis=1)

    def clean_id(val):
        try:
            return str(int(float(val))).strip()
        except:
            return None

    extended["city_id"] = extended["city_id"].apply(clean_id)
    cities["id"] = cities["id"].apply(clean_id)

    cities_small = cities.rename(columns={
        "id": "city_id",
        "name": "city_name_from_cities"
    })[["city_id", "city_name_from_cities"]]

    merged = extended.merge(cities_small, on="city_id", how="left")

    def is_valid(s):
        return isinstance(s, str) and s.strip() not in ("", "nan", "none")

    mask_valid = merged["city_name"].apply(is_valid) | merged["city_name_from_cities"].apply(is_valid)

    valid_before = mask_valid.sum()
    invalid_before = len(merged) - valid_before

    merged = merged[mask_valid].copy()

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
        encoding="utf-8"
    )

    logging.info(
        f"[STEP 00] "
        f"loaded={total_extended:,} "
        f"valid_city={valid_before:,} "
        f"invalid_city={invalid_before:,} "
        f"final={len(merged):,} "
        f"saved={OUTPUT_FILE.name}"
    )

except Exception:
    logging.exception("ERROR in STEP 00")
    sys.exit(1)

finish_logging()

print("Script: 00_clean_address executed")
