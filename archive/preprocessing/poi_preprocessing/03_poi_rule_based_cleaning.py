# src/preprocessing/poi_preprocessing/03_poi_rule_based_cleaning.py

import re
import unicodedata
import pandas as pd
from pathlib import Path
from utils.logging_config import setup_logging
from utils.save_csv import save_csv
import logging

BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_FILE = BASE_DIR / "Dataset" / "processed" / "poi_processed" / "02_pois_usercount_over0.csv"

OUTPUT_DIR = BASE_DIR / "Dataset" / "processed" / "poi_processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE_CLEAN = OUTPUT_DIR / "03_poi_names_cleaned.csv"
OUTPUT_FILE_REMOVED = OUTPUT_DIR / "03_poi_names_removed.csv"

# ============================================================
# Shared central logfile (append to poi_preprocessing.log)
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

finish_logging = setup_logging("03_poi_rule_based_cleaning")

df = pd.read_csv(INPUT_FILE, low_memory=False)
input_total = len(df)

removed_all = []


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("ß", "ss")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["lc"] = df["name"].fillna("").apply(normalize_text)


def remove_and_log(df_local, mask, description):
    removed = df_local[mask].copy()
    removed.loc[:, "_reason"] = description
    remaining = df_local[~mask]
    removed_all.append(removed)
    return remaining


# Rule 1 — empty / unparseable names
mask_empty = df["lc"].isna() | (df["lc"].str.strip() == "")
df = remove_and_log(df, mask_empty, "empty_or_unparseable")

# Rule 2 — emojis / smileys
FULL_EMOJI_REGEX = (
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]"
    r"|(?:\:\)|\:\(|;\-\)|:\-\(|;D|:D|:-D|<3|;\)|:\]|:\[)"
)
mask_emoji = df["name"].str.contains(FULL_EMOJI_REGEX, regex=True, na=False)
df = remove_and_log(df, mask_emoji, "emoji_or_smiley")

# Rule 3 — name equals city name
mask_city1 = df["city_name"].fillna("").apply(normalize_text) == df["lc"]
df = remove_and_log(df, mask_city1, "name_equals_city_name")

mask_city2 = df["city_name_from_cities"].fillna("").apply(normalize_text) == df["lc"]
df = remove_and_log(df, mask_city2, "name_equals_city_from_cities")

# Rule 4 — forbidden words
FORBIDDEN_WHOLE = ["du", "ich", "stadt", "night", "city", "str", "yourself", "zu hause", "irgendwo", "welt"]
FORBIDDEN_SUBSTRING = [
    "treffen", "life", "love", "home", "strasse",
    "urlaub", "leben", "sonne", "himmel", "zuhause",
    "daheim", "haha"
]
pattern_whole = r"|".join([rf"\b{w}\b" for w in FORBIDDEN_WHOLE])
pattern_sub = r"|".join([rf"{w}" for w in FORBIDDEN_SUBSTRING])

mask_forbidden = df["lc"].str.contains(pattern_whole, regex=True, na=False) | \
                 df["lc"].str.contains(pattern_sub, regex=True, na=False)
df = remove_and_log(df, mask_forbidden, "forbidden_words")

# Rule 5 — temporary events
EVENT_WORDS = ["party", "openair", "event", "festival", "open air", "feier", "veranstaltung"]
pattern_events = r"|".join([rf"{w}" for w in EVENT_WORDS])
mask_events = df["lc"].str.contains(pattern_events, regex=True, na=False)
df = remove_and_log(df, mask_events, "temporary_event")

# Rule 6 — years in name
YEAR_PATTERN = r"\b(?:19\d{2}|20\d{2})\b"
mask_year = df["lc"].str.contains(YEAR_PATTERN, regex=True, na=False)
df = remove_and_log(df, mask_year, "year_in_name")

# Rule 7 — very short names
mask_short = df["lc"].str.fullmatch(r"[a-z]{1,2}")
df = remove_and_log(df, mask_short, "name_too_short")

# Rule 8 — region names
REGION_WORDS = [
    "germany", "deutschland",
    "baden wurttemberg", "bayern",
    "austria", "osterreich",
    "switzerland", "schweiz"
]
pattern_regions = r"|".join([rf"\b{w}\b" for w in REGION_WORDS])
mask_regions = df["lc"].str.contains(pattern_regions, regex=True, na=False)
df = remove_and_log(df, mask_regions, "region_name")

# Rule 9 — equals city.csv name
cities_path = BASE_DIR / "Dataset" / "raw" / "cities.csv"
cities = pd.read_csv(cities_path, low_memory=False)
city_names_normalized = set(
    cities["name"].fillna("").apply(normalize_text).unique()
)
mask_city_match = df["lc"].isin(city_names_normalized)
df = remove_and_log(df, mask_city_match, "name_equals_city_csv")

remaining_total = len(df)
removed_total = sum(len(x) for x in removed_all)

save_csv(df.drop(columns=["lc"]), OUTPUT_FILE_CLEAN)

if removed_all:
    removed_df = pd.concat(removed_all, ignore_index=True)
    save_csv(removed_df, OUTPUT_FILE_REMOVED)

logging.info(
    f"[STEP 03] "
    f"input={input_total:,} "
    f"removed={removed_total:,} "
    f"remaining={remaining_total:,} "
    f"saved_clean={OUTPUT_FILE_CLEAN.name} "
    f"saved_removed={OUTPUT_FILE_REMOVED.name}"
)

finish_logging()

print("Script: 03_poi_rule_based_cleaning.py executed")