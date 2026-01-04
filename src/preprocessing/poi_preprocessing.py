# -*- coding: utf-8 -*-
"""
Instagram CSV preprocessing & filtering pipeline with stepwise statistics.

Final outputs (as requested, all under DATA_DIR/output):
- business.csv
- city.csv
- checkin.csv
- user.csv
- media.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
)
from datetime import date, timedelta
import re
import unicodedata

# Try tqdm for nicer progress bars; fall back to plain loops if missing
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# =========================
# Config (change as needed)
# =========================
# src/preprocessing/poi_preprocessing.py
THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = THIS_FILE.parents[2]
# parents[0] = preprocessing
# parents[1] = src
# parents[2] = Repo-Root (AgenticGroundedRAG)

DATASET_DIR = PROJECT_ROOT / "Dataset"
RAW_DIR     = DATASET_DIR / "raw"
OUTPUT_DIR  = DATASET_DIR / "output_poi_preprocessing"

MEDIA_F  = RAW_DIR / "media.csv"
USERS_F  = RAW_DIR / "user.csv"
LOCS_F   = RAW_DIR / "extended_locations.csv"
CITIES_F = RAW_DIR / "cities.csv"

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TOP3_YEARS_FIXED = [2017, 2018, 2019]
MIN_NUM_INTERACTIONS = 10

# Bot detection thresholds
BOT_ACTIVITY_THRESHOLD = 50   # scan users with > 50 posts
BOT_SPEED_KMH = 900           # unrealistic if adjacent-post speed > 900 km/h

# Visit collapsing (same location within short gap) â€” kept but disabled below
VISIT_GAP_DAYS = 1
VISIT_KEEP = "first"

# Geo filter Bodensee bounding box
MIN_LAT = 47.20     # St. Gallen
MAX_LAT = 48.10     # Oberschwaben
MIN_LON = 8.40      # Schaffhausen
MAX_LON = 10.30     # Kleinwalsertal

# =========================
# Save / IO helpers
# =========================
def save_csv(df: pd.DataFrame, out_path: Path, fname: str):
    """Save CSV with directories created."""
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / fname, index=False)

def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Safely read CSV as strings; return empty DataFrame if missing."""
    if not path.exists():
        print(f"WARN: File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def _detect_first_existing(df: pd.DataFrame, candidates):
    """Return the first column name found in df from a list of candidates."""
    return next((c for c in candidates if c in df.columns), None)

# =========================
# Dtype / parsing helpers
# =========================
def _to_int(s):
    """Coerce to pandas nullable Int64 safely."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    """Safely parse timestamps to UTC datetimes."""
    if is_datetime64_any_dtype(series) or is_datetime64tz_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    if is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")

def _time_column(media: pd.DataFrame):
    # Filtering stage may use either; final outputs will ONLY use taken_at_timestamp
    if "taken_at_timestamp" in media.columns:
        return "taken_at_timestamp"
    if "created_time" in media.columns:
        return "created_time"
    raise KeyError("No time column found: expected 'taken_at_timestamp' or 'created_time'.")

def _add_is_photo(media: pd.DataFrame):
    """Detect photo rows based on the file extension in media_downloaded_name."""
    if "media_downloaded_name" in media.columns:
        exts = media["media_downloaded_name"].astype(str).str.lower().str.extract(r"(\.[a-z0-9]+)$")
        media["is_photo"] = exts[0].isin(PHOTO_EXTS)
    else:
        media["is_photo"] = False
    return media

def _detect_lat_lon_cols(df: pd.DataFrame):
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "lng", "x"]
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    return lat_col, lon_col

def _detect_loc_name_col(locs_df: pd.DataFrame):
    name_candidates = ["name", "location_name", "title", "display_name"]
    return next((c for c in name_candidates if c in locs_df.columns), None)

# =========================
# Stats (simple printouts)
# =========================
def analyze_statistics(baseline_df: pd.DataFrame,
                       prev_df: pd.DataFrame,
                       curr_df: pd.DataFrame,
                       label: str):
    n_curr = len(curr_df)
    n_prev = len(prev_df)
    n_base = len(baseline_df)

    users_curr = curr_df['owner_id'].nunique()
    users_prev = prev_df['owner_id'].nunique()
    users_base = baseline_df['owner_id'].nunique()

    locs_curr = curr_df['location_id'].nunique()
    locs_prev = prev_df['location_id'].nunique()
    locs_base = baseline_df['location_id'].nunique()

    pct_rows_vs_prev = (n_curr / n_prev * 100.0) if n_prev > 0 else 0.0
    pct_rows_vs_base = (n_curr / n_base * 100.0) if n_base > 0 else 0.0

    pct_users_vs_prev = (users_curr / users_prev * 100.0) if users_prev > 0 else 0.0
    pct_users_vs_base = (users_curr / users_base * 100.0) if users_base > 0 else 0.0

    pct_locs_vs_prev = (locs_curr / locs_prev * 100.0) if locs_prev > 0 else 0.0
    pct_locs_vs_base = (locs_curr / locs_base * 100.0) if locs_base > 0 else 0.0

    seq_len_prev = prev_df.groupby('owner_id').size() if len(prev_df) else pd.Series(dtype=int)
    seq_len_curr = curr_df.groupby('owner_id').size() if len(curr_df) else pd.Series(dtype=int)

    avg_seq_prev  = seq_len_prev.mean()  if len(seq_len_prev)  else 0.0
    avg_seq_curr  = seq_len_curr.mean()  if len(seq_len_curr)  else 0.0
    max_seq_curr  = seq_len_curr.max()   if len(seq_len_curr)  else 0
    min_seq_curr  = seq_len_curr.min()   if len(seq_len_curr)  else 0

    posts_per_loc_curr = curr_df.groupby('location_id').size() if len(curr_df) else pd.Series(dtype=int)
    avg_posts_per_loc  = posts_per_loc_curr.mean() if len(posts_per_loc_curr) else 0.0

    print(f"\n==== {label} ====")
    print(f"Media rows: {n_prev:,} --> {n_curr:,}  "
          f"(retained {pct_rows_vs_prev:.2f}% vs prev | {pct_rows_vs_base:.2f}% vs baseline)")
    print(f"Users:      {users_prev:,} --> {users_curr:,}  "
          f"(retained {pct_users_vs_prev:.2f}% vs prev | {pct_users_vs_base:.2f}% vs baseline)")
    print(f"Locations:  {locs_prev:,} --> {locs_curr:,}  "
          f"(retained {pct_locs_vs_prev:.2f}% vs prev | {pct_locs_vs_base:.2f}% vs baseline)")
    print(f"Avg user seq length: {avg_seq_prev:.2f} --> {avg_seq_curr:.2f}")
    print(f"Max/Min user seq length (curr): {max_seq_curr} / {min_seq_curr}")
    print(f"Avg posts per location (curr): {avg_posts_per_loc:.2f}")

# =========================
# Bot detection
# =========================
def user_has_unrealistic_speed(media_df: pd.DataFrame,
                               locs_df: pd.DataFrame,
                               user_id,
                               time_col: str,
                               bot_speed_kmh: float = 900) -> bool:
    m = media_df[media_df["owner_id"] == user_id].copy()
    m = m[m["location_id"].notna()]
    if m.empty:
        return False

    lat_col, lon_col = _detect_lat_lon_cols(locs_df)
    if lat_col is None or lon_col is None:
        return False

    mm = m.merge(
        locs_df[["id", lat_col, lon_col]].rename(columns={"id":"location_id"}),
        on="location_id", how="left"
    ).dropna(subset=[lat_col, lon_col])
    if mm.empty:
        return False

    mm[time_col] = _ensure_datetime_utc(mm[time_col])
    mm = mm.sort_values(by=[time_col])

    mm["prev_lat"] = mm[lat_col].shift()
    mm["prev_lon"] = mm[lon_col].shift()
    mm["time_diff_h"] = (mm[time_col] - mm[time_col].shift()).dt.total_seconds() / 3600.0

    def _dist_km(r):
        if pd.isna(r[lat_col]) or pd.isna(r[lon_col]) or pd.isna(r["prev_lat"]) or pd.isna(r["prev_lon"]):
            return np.nan
        return geodesic((r["prev_lat"], r["prev_lon"]), (r[lat_col], r[lon_col])).kilometers

    mm["dist_km"] = mm.apply(_dist_km, axis=1)
    mm = mm[(mm["time_diff_h"] > 0) & (mm["dist_km"].notna())]
    if mm.empty:
        return False

    mm["speed_kmh"] = mm["dist_km"] / mm["time_diff_h"]
    return (mm["speed_kmh"] > bot_speed_kmh).any()

# =========================
# Visit-level collapsing
# =========================
def collapse_consecutive_same_location(
        media_df: pd.DataFrame,
        time_col: str,
        min_gap_days: int = 7,
        keep: str = "first",
        verbose: bool = True,
        print_every: int = 1000,
) -> pd.DataFrame:
    """
    Collapse consecutive posts at the same location within a user's sequence.
    New visit starts if location changes OR time gap >= min_gap_days.
    """
    assert keep in ("first", "last")
    df = media_df.copy()

    if verbose:
        print(f"[Visit collapse] Starting... rows={len(df):,}, users={df['owner_id'].nunique():,}, "
              f"min_gap_days={min_gap_days}, keep='{keep}'")

    # Remove rows with NaN location_id
    keep_na_loc = False
    if not keep_na_loc:
        before_drop = len(df)
        df = df[df["location_id"].notna()].copy()
        if verbose and before_drop != len(df):
            print(f"[Visit collapse] Dropped rows with NaN location_id: {before_drop - len(df):,}")

    df["location_id"] = _to_int(df["location_id"])
    df[time_col] = _ensure_datetime_utc(df[time_col])

    # Sort for stable segmentation
    df = df.sort_values(["owner_id", time_col]).reset_index(drop=True)

    def _per_user(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).copy()
        gap_days = (g[time_col] - g[time_col].shift()).dt.total_seconds().div(86400.0)
        same_loc = g["location_id"].eq(g["location_id"].shift())
        new_visit = (~same_loc) | (gap_days.isna()) | (gap_days >= float(min_gap_days))
        g["__visit_id"] = new_visit.cumsum()
        if keep == "first":
            out = g.groupby("__visit_id", as_index=False, sort=False).head(1)
        else:
            out = g.groupby("__visit_id", as_index=False, sort=False).tail(1)
        return out

    groups = df.groupby("owner_id", group_keys=False)
    user_ids = list(groups.groups.keys())
    kept_list = []

    iterator = tqdm(user_ids, desc="Collapsing visits", unit="user") if _HAS_TQDM and verbose else user_ids

    for idx, uid in enumerate(iterator, 1):
        g = groups.get_group(uid)
        kept_list.append(_per_user(g))
        if (not _HAS_TQDM) and verbose and (idx % print_every == 0):
            print(f"[Visit collapse] Processed {idx:,} / {len(user_ids):,} users...")

    kept = pd.concat(kept_list, ignore_index=True)

    if "__visit_id" in kept.columns:
        kept = kept.drop(columns=["__visit_id"])

    if verbose:
        print(f"[Visit collapse] Done. rows: {len(df):,} -> {len(kept):,} "
              f"(reduction {-len(kept) + len(df):,})")

    return kept

# =========================
# Joint interactive filter
#   - user >= min_user_interactions
#   - POI  >= min_photos (photo posts)
#   - POI  >= min_users_per_poi (photo users)
# =========================
def iterative_joint_filter(
        media_df: pd.DataFrame,
        min_user_interactions: int = 10,
        min_photos: int = 5,
        min_users_per_poi: int = 3,
        max_iters: int = 50,
        verbose: bool = True,
) -> pd.DataFrame:

    required = {"owner_id", "location_id", "is_photo"}
    missing = required - set(media_df.columns)
    if missing:
        raise KeyError(f"media is missing required columns: {missing}")

    df = media_df.copy()
    prev_len = -1

    for it in range(1, max_iters + 1):
        if verbose:
            print(f"[Joint iter {it}]")

        # ----------------------------------
        # (1) USER constraint
        # ----------------------------------
        user_counts = df["owner_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["owner_id"].isin(keep_users)]

        # ----------------------------------
        # (2) POI photo count constraint
        # ----------------------------------
        photo_counts = (
            df[df["is_photo"] == True]
            .groupby("location_id")
            .size()
        )

        # ----------------------------------
        # (3) POI distinct user constraint (photo-based)
        # ----------------------------------
        user_counts_poi = (
            df[df["is_photo"] == True]
            .groupby("location_id")["owner_id"]
            .nunique()
        )

        poi_stats = pd.concat([photo_counts, user_counts_poi], axis=1)
        poi_stats.columns = ["photo_count", "user_count"]
        poi_stats = poi_stats.fillna(0)

        keep_pois = poi_stats[
            (poi_stats["photo_count"] >= min_photos) &
            (poi_stats["user_count"]  >= min_users_per_poi)
            ].index

        df = df[df["location_id"].isin(keep_pois)]

        # ----------------------------------
        # convergence check
        # ----------------------------------
        cur_len = len(df)

        if verbose:
            print(
                f"  rows={cur_len:,} | "
                f"users={df['owner_id'].nunique():,} | "
                f"pois={df['location_id'].nunique():,}"
            )

        if cur_len == prev_len:
            if verbose:
                print(
                    f"[Joint filter] Converged "
                    f"(users>={min_user_interactions}, "
                    f"photos>={min_photos}, "
                    f"poi-users>={min_users_per_poi})"
                )
            break

        prev_len = cur_len

    return df


# =========================
# Main
# =========================
def main():
    # Read CSVs (as strings), then coerce key ints
    media = pd.read_csv(MEDIA_F, dtype=str, low_memory=False)
    users = pd.read_csv(USERS_F, dtype=str, low_memory=False)
    locs  = pd.read_csv(LOCS_F, dtype=str, low_memory=False)
    cities= pd.read_csv(CITIES_F, dtype=str, low_memory=False)

    for col in ("id", "owner_id", "location_id"):
        if col in media.columns:
            media[col] = _to_int(media[col])
    if "id" in users.columns:
        users["id"] = _to_int(users["id"])
    if "id" in locs.columns:
        locs["id"] = _to_int(locs["id"])
    if "city_id" in locs.columns:
        locs["city_id"] = _to_int(locs["city_id"])
    if "id" in cities.columns:
        cities["id"] = _to_int(cities["id"])

    # Parse time for filtering (final outputs will only use taken_at_timestamp)
    time_col_for_filter = _time_column(media)
    media[time_col_for_filter] = _ensure_datetime_utc(media[time_col_for_filter])
    media = media[media[time_col_for_filter].notna()].copy()
    media["year"] = media[time_col_for_filter].dt.year
    media = _add_is_photo(media)

    baseline_media = media.copy()


    # Geo filter Bodensee bounding box
    prev = media
    lat_col, lon_col = _detect_lat_lon_cols(locs)
    if lat_col is None or lon_col is None:
        print("WARN geo filter skipped because latitude or longitude column not found in locations")
    else:
        locs_geo = locs[["id", lat_col, lon_col]].copy()
        locs_geo[lat_col] = pd.to_numeric(locs_geo[lat_col], errors="coerce")
        locs_geo[lon_col] = pd.to_numeric(locs_geo[lon_col], errors="coerce")

        media_geo = media.merge(
            locs_geo.rename(columns={"id": "location_id"}),
            on="location_id",
            how="left"
        )
        mask_box = (
                media_geo[lat_col].between(MIN_LAT, MAX_LAT)
                & media_geo[lon_col].between(MIN_LON, MAX_LON)
        )
        media = media_geo[mask_box].copy()
        media = media.drop(columns=[lat_col, lon_col])

        analyze_statistics(
            baseline_media,
            prev,
            media,
            label=(
                f"Geo filter Bodensee box "
                f"[lat {MIN_LAT} to {MAX_LAT}] "
                f"[lon {MIN_LON} to {MAX_LON}]"
            ),
        )

    # Keep fixed top years
    prev = media
    media = media[media["year"].isin(TOP3_YEARS_FIXED)].copy()
    analyze_statistics(baseline_media, prev, media, label=f"Keep years {TOP3_YEARS_FIXED}")


    # =========================================================
    # Normalize POI name for locations appearing in current media
    # (Unicode normalize -> ÃŸ->ss -> drop non-ASCII -> lowercase
    #  -> non-alnum to spaces -> squeeze spaces)
    # =========================================================
    print("\n[POI name] Normalizing location names for locations in filtered media...")

    def normalize_text(text: str) -> str:
        """Normalize POI name into a strict ASCII, lowercase, space-separated token string."""
        if not isinstance(text, str):
            return ""
        # 1. lowercase first
        text = text.lower()
        # 2. German umlaut normalization (before Unicode normalization!)
        text = (
            text.replace("Ã¤", "ae")
            .replace("Ã¶", "oe")
            .replace("Ã¼", "ue")
            .replace("ÃŸ", "ss")
        )
        # 3. Unicode normalization
        text = unicodedata.normalize("NFKD", text)
        # 4. ASCII fallback
        text = text.encode("ascii", "ignore").decode("ascii")
        # 5. cleanup
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    name_col = _detect_loc_name_col(locs)
    if name_col is None:
        name_col = "name"
        if "name" not in locs.columns:
            locs["name"] = ""

    # Only keep locations that actually appear in the current media
    kept_locs_now = set(media["location_id"].dropna().astype("Int64").unique())

    locs_name = locs[["id", name_col]].copy()
    locs_name["id"] = _to_int(locs_name["id"])
    locs_name = locs_name[locs_name["id"].isin(kept_locs_now)].copy()

    locs_name["location_name_norm"] = locs_name[name_col].fillna("").apply(normalize_text)

    prev = media
    media = media.merge(
        locs_name.rename(columns={"id": "location_id"})[["location_id", "location_name_norm"]],
        on="location_id",
        how="left"
    )

    # Keep empty string if name is missing
    media["location_name_norm"] = media["location_name_norm"].fillna("")

    # Quick stats: how many locations end up with empty normalized name
    empty_norm = int((media["location_name_norm"].str.strip() == "").sum())
    print(f"[POI name] Added column: location_name_norm")
    print(f"[POI name] Rows with empty normalized name: {empty_norm:,} / {len(media):,}")
    analyze_statistics(baseline_media, prev, media, label="Add normalized POI name (location_name_norm)")

    # =========================================================
    # Drop rows with empty normalized POI name
    # (e.g. original name empty or only symbols/emojis)
    # =========================================================
    print("\n[POI name] Dropping rows with empty normalized POI name...")

    prev = media

    # location_name_norm is empty or whitespace-only
    mask_empty_norm = media["location_name_norm"].isna() | (
            media["location_name_norm"].str.strip() == ""
    )

    dropped_rows = int(mask_empty_norm.sum())

    media = media[~mask_empty_norm].copy()

    print(
        f"[POI name] Dropped {dropped_rows:,} rows due to empty normalized POI name"
    )

    analyze_statistics(
        baseline_media,
        prev,
        media,
        label="Drop empty normalized POI names"
    )

    # =========================================================
    # Canonical POI dedup (STRICT) in FILTERED media
    # Key = (location_name_norm, lat, lon)
    # =========================================================
    print(
        "\n[Canonical POI strict] Dedup by (location_name_norm, lat, lon) in filtered media...")

    prev = media

    lat_col, lon_col = _detect_lat_lon_cols(locs)
    if lat_col is None or lon_col is None:
        print(
            "WARN canonical POI skipped because latitude or longitude column not found in locations")
    else:
        if "location_name_norm" not in media.columns:
            raise KeyError(
                "location_name_norm not found in media. Build it before canonical step.")

        # Attach lat/lon to current filtered media
        locs_geo = locs[["id", lat_col, lon_col]].copy()
        locs_geo["id"] = _to_int(locs_geo["id"])
        locs_geo[lat_col] = pd.to_numeric(locs_geo[lat_col], errors="coerce")
        locs_geo[lon_col] = pd.to_numeric(locs_geo[lon_col], errors="coerce")
        locs_geo = locs_geo.dropna(subset=["id", lat_col, lon_col]).copy()
        locs_geo = locs_geo.rename(columns={"id": "location_id"})

        media_ll = media.merge(locs_geo, on="location_id", how="left")

        # Drop media rows without coordinates (cannot canonicalize)
        before = len(media_ll)
        media_ll = media_ll.dropna(subset=[lat_col, lon_col]).copy()
        dropped = before - len(media_ll)
        if dropped > 0:
            print(
                f"[Canonical POI strict] Dropped {dropped:,} media rows due to missing lat/lon")

        # Build canonical id: min(original location_id) per (norm, lat, lon)
        media_ll["location_id"] = _to_int(media_ll["location_id"])
        key_cols = ["location_name_norm", lat_col, lon_col]

        canonical_map = (
            media_ll.dropna(subset=key_cols + ["location_id"])
            .groupby(key_cols, as_index=False)["location_id"]
            .min()
            .rename(columns={"location_id": "canonical_location_id"})
        )

        # Join canonical id back
        media_ll = media_ll.merge(canonical_map, on=key_cols, how="left")
        before = len(media_ll)
        media_ll = media_ll.dropna(subset=["canonical_location_id"]).copy()
        dropped = before - len(media_ll)
        if dropped > 0:
            print(
                f"[Canonical POI strict] Dropped {dropped:,} media rows due to missing canonical id")

        media_ll["canonical_location_id"] = _to_int(
            media_ll["canonical_location_id"])

        # Stats: how many groups actually merge >1 original location_id
        dup_groups = (
            media_ll.groupby("canonical_location_id")
            .filter(lambda g: g["location_id"].nunique() > 1)
            .groupby("canonical_location_id")
        )
        print(
            f"[Canonical POI strict] Canonical groups merging >=2 original POIs: {dup_groups.ngroups:,}")

        # Apply canonical replacement
        media = media_ll.copy()
        media["location_id"] = media["canonical_location_id"]
        media = media.drop(columns=["canonical_location_id", lat_col, lon_col],
                           errors="ignore")

        analyze_statistics(
            baseline_media,
            prev,
            media,
            label="Canonical POI strict merge by (location_name_norm, lat, lon)"
        )

    # =========================================================
    # Drop sentence-like / emotional POI names
    # (whole-token forbidden words, with business-name protection)
    # =========================================================
    print("\n[POI name] Dropping sentence-like / emotional POI names...")

    prev = media

    # -------------------------
    # Forbidden tokens (VERY conservative)
    # -------------------------
    FORBIDDEN_WHOLE_EN = {
        "my", "our",
        "home",
        "party",
        "love",
    }

    FORBIDDEN_WHOLE_DE = {
        "mein", "meine",
        "daheim",
        "zuhause",
    }

    FORBIDDEN_TOKENS = FORBIDDEN_WHOLE_EN | FORBIDDEN_WHOLE_DE

    # -------------------------
    # Business / real-POI protection tokens
    # If any of these appear, we KEEP the POI
    # -------------------------
    PROTECTED_TOKENS = {
        # English (strict, entity-indicative only)
        "hotel", "resort", "hostel",
        "restaurant", "cafe", "bar", "bistro", "pub",
        "museum", "library", "university", "campus",
        "shop", "store", "mall",
        "station", "airport",
        "clinic", "hospital",
        "theatre", "cinema",

        # German (strict, entity-indicative only)
        "hotel", "gasthaus", "pension",
        "restaurant", "cafe", "bar", "bistro",
        "museum", "bibliothek", "universitaet", "campus",
        "laden", "markt", "zentrum",
        "bahnhof", "flughafen",
        "klinik", "krankenhaus",
        "theater", "kino",
    }

    # -------------------------
    # Tokenize helper
    # -------------------------
    def tokenize(name: str):
        if not isinstance(name, str):
            return []
        return name.split(" ")

    # -------------------------
    # Build masks
    # -------------------------
    tokens = media["location_name_norm"].apply(tokenize)

    # forbidden token appears as a FULL token
    mask_forbidden = tokens.apply(
        lambda ts: any(t in FORBIDDEN_TOKENS for t in ts)
    )

    # protected token appears
    mask_protected = tokens.apply(
        lambda ts: any(t in PROTECTED_TOKENS for t in ts)
    )

    # final decision:
    # forbidden AND NOT protected
    mask_drop = mask_forbidden & (~mask_protected)

    dropped_rows = int(mask_drop.sum())

    # -------------------------
    # Show examples
    # -------------------------
    print(
        f"[POI name] Dropping {dropped_rows:,} rows due to sentence-like POI names"
    )

    if dropped_rows > 0:
        print("\n[POI name] Examples of dropped POIs:")

        examples = (
            media.loc[mask_drop, ["location_id", "location_name_norm"]]
            .drop_duplicates()
            .head(30)
        )

        for _, r in examples.iterrows():
            print(
                f"  - location_id={int(r['location_id'])}, "
                f"location_name_norm='{r['location_name_norm']}'"
            )

    # -------------------------
    # Apply drop
    # -------------------------
    media = media[~mask_drop].copy()

    analyze_statistics(
        baseline_media,
        prev,
        media,
        label="Drop sentence-like / emotional POI names"
    )

    # # =========================================================
    # # Canonical POI by identical lat/lon with sanity check
    # # Canonical chosen by max locs.edge_media_count (tie -> min id)
    # # Replace ONLY when dominance is strong (top1_share >= TOP1_DOMINANCE)
    # # and small nodes are truly tiny (loc_share <= SMALL_SHARE)
    # # IMPORTANT: restrict to POIs that appear in CURRENT filtered media only
    # # =========================================================
    # print(
    #     "\n[Canonical POI] Canonicalizing by identical lat/lon with sanity check...")
    #
    # TOP1_DOMINANCE = 0.80  # only canonicalize when the top location dominates the coordinate cluster
    # SMALL_SHARE = 0.02  # only replace locations whose share is very small
    # MIN_CLUSTER_TOTAL = 50  # skip tiny clusters to reduce noise
    #
    # lat_col, lon_col = _detect_lat_lon_cols(locs)
    #
    # if lat_col is None or lon_col is None:
    #     print(
    #         "WARN canonical POI skipped because latitude or longitude column not found in locations")
    # elif "edge_media_count" not in locs.columns:
    #     print(
    #         "WARN canonical POI skipped because edge_media_count not found in locations")
    # else:
    #     if "location_name_norm" not in media.columns:
    #         raise KeyError(
    #             "location_name_norm not found in media. Build it before canonical step.")
    #
    #     prev = media
    #
    #     # ---------------------------------------------------------
    #     # 1) Only keep locations that appear in CURRENT filtered media
    #     # ---------------------------------------------------------
    #     kept_locs_now = set(
    #         media["location_id"].dropna().astype("Int64").unique())
    #
    #     locs_key = locs[["id", lat_col, lon_col, "edge_media_count"]].copy()
    #     locs_key["id"] = _to_int(locs_key["id"])
    #     locs_key[lat_col] = pd.to_numeric(locs_key[lat_col], errors="coerce")
    #     locs_key[lon_col] = pd.to_numeric(locs_key[lon_col], errors="coerce")
    #     locs_key["edge_media_count"] = pd.to_numeric(
    #         locs_key["edge_media_count"], errors="coerce").fillna(0)
    #
    #     locs_key = locs_key.dropna(subset=["id", lat_col, lon_col]).copy()
    #     locs_key = locs_key[locs_key["id"].isin(kept_locs_now)].copy()
    #
    #     if locs_key.empty:
    #         print(
    #             "WARN canonical POI skipped because no locations remain after filtering to current media")
    #     else:
    #         # ---------------------------------------------------------
    #         # 2) Compute cluster totals and identify top1 per (lat, lon)
    #         # ---------------------------------------------------------
    #         cluster_totals = (
    #             locs_key.groupby([lat_col, lon_col], as_index=False)[
    #                 "edge_media_count"]
    #             .sum()
    #             .rename(
    #                 columns={"edge_media_count": "cluster_edge_media_count"})
    #         )
    #
    #         # sort for top1 selection: max edge_media_count, tie -> min id
    #         locs_sorted = locs_key.sort_values(
    #             by=[lat_col, lon_col, "edge_media_count", "id"],
    #             ascending=[True, True, False, True]
    #         ).copy()
    #
    #         top1 = (
    #             locs_sorted.groupby([lat_col, lon_col], as_index=False)
    #             .first()[[lat_col, lon_col, "id", "edge_media_count"]]
    #             .rename(columns={
    #                 "id": "top1_location_id",
    #                 "edge_media_count": "top1_edge_media_count"
    #             })
    #         )
    #
    #         top1 = top1.merge(cluster_totals, on=[lat_col, lon_col], how="left")
    #         top1["top1_share"] = np.where(
    #             top1["cluster_edge_media_count"] > 0,
    #             top1["top1_edge_media_count"] / top1[
    #                 "cluster_edge_media_count"],
    #             0.0
    #         )
    #
    #         # ---------------------------------------------------------
    #         # 3) Attach per-location shares and replacement decision
    #         # ---------------------------------------------------------
    #         locs_aug = locs_key.merge(top1, on=[lat_col, lon_col], how="left")
    #
    #         locs_aug = locs_aug.rename(columns={
    #             "id": "location_id",
    #             "edge_media_count": "location_edge_media_count"
    #         })
    #
    #         locs_aug["loc_share"] = np.where(
    #             locs_aug["cluster_edge_media_count"] > 0,
    #             locs_aug["location_edge_media_count"] / locs_aug[
    #                 "cluster_edge_media_count"],
    #             0.0
    #         )
    #
    #         locs_aug["eligible_cluster"] = locs_aug[
    #                                            "cluster_edge_media_count"] >= MIN_CLUSTER_TOTAL
    #
    #         locs_aug["replace_to_top1"] = (
    #                 locs_aug["eligible_cluster"]
    #                 & (locs_aug["top1_share"] >= TOP1_DOMINANCE)
    #                 & (locs_aug["loc_share"] <= SMALL_SHARE)
    #                 & (locs_aug["location_id"] != locs_aug["top1_location_id"])
    #         )
    #
    #         # ---------------------------------------------------------
    #         # 4) Merge these decisions back into current filtered media
    #         # ---------------------------------------------------------
    #         media_canon = media.merge(
    #             locs_aug[[
    #                 "location_id",
    #                 "top1_location_id",
    #                 "cluster_edge_media_count",
    #                 "top1_share",
    #                 "loc_share",
    #                 "replace_to_top1"
    #             ]],
    #             on="location_id",
    #             how="left"
    #         )
    #
    #         before_drop = len(media_canon)
    #         media_canon = media_canon.dropna(subset=["top1_location_id"]).copy()
    #         dropped = before_drop - len(media_canon)
    #         if dropped > 0:
    #             print(
    #                 f"[Canonical POI] Dropped {dropped:,} media rows due to missing canonical mapping")
    #
    #         media_canon["top1_location_id"] = _to_int(
    #             media_canon["top1_location_id"])
    #
    #         # keep originals for debug print
    #         media_canon["original_location_id"] = media_canon["location_id"]
    #         media_canon["original_location_name_norm"] = media_canon[
    #             "location_name_norm"]
    #
    #         # ---------------------------------------------------------
    #         # 5) Apply replacement (avoid np.where to keep Int64 safe)
    #         # ---------------------------------------------------------
    #         mask_replace = media_canon["replace_to_top1"] == True
    #         replaced_rows = int(mask_replace.sum())
    #
    #         media_canon.loc[mask_replace, "location_id"] = media_canon.loc[
    #             mask_replace, "top1_location_id"]
    #         media_canon["location_id"] = _to_int(media_canon["location_id"])
    #
    #         # Update location_name_norm only for replaced rows
    #         # We need the top1 normalized name inside media. We can derive it by mapping top1_location_id -> one existing name_norm.
    #         # Since strict step already canonicalized (name,lat,lon), top1_location_id should have consistent name_norm in media.
    #         top1_name_map = (
    #             media_canon.loc[:, ["location_id", "location_name_norm"]]
    #             .dropna()
    #             .drop_duplicates(subset=["location_id"])
    #             .set_index("location_id")["location_name_norm"]
    #             .to_dict()
    #         )
    #
    #         def _lookup_top1_norm(x):
    #             return top1_name_map.get(x, "")
    #
    #         if replaced_rows > 0:
    #             media_canon.loc[mask_replace, "location_name_norm"] = (
    #                 media_canon.loc[mask_replace, "location_id"]
    #                 .apply(_lookup_top1_norm)
    #                 .fillna("")
    #                 .astype(str)
    #             )
    #
    #         # final media
    #         media = media_canon.drop(columns=["top1_location_id"],
    #                                  errors="ignore")
    #
    #         print(
    #             f"[Canonical POI] Replaced rows: {replaced_rows:,} / {len(media_canon):,}")
    #
    #         analyze_statistics(
    #             baseline_media,
    #             prev,
    #             media,
    #             label="Canonical POI by lat/lon with sanity check (top1 dominance + small-share)"
    #         )
    #
    #         # ---------------------------------------------------------
    #         # 6) Show examples: which POIs were replaced
    #         # ---------------------------------------------------------
    #         print("\n[Canonical POI sanity] Examples of POI name replacements:")
    #
    #         replaced = media_canon[mask_replace].copy()
    #
    #         if replaced.empty:
    #             print("[Canonical POI sanity] No POI replacements applied.")
    #         else:
    #             # group by target canonical location_id (after replacement)
    #             groups = replaced.groupby("location_id", sort=False)
    #
    #             MAX_SHOW = 10
    #             shown = 0
    #
    #             for canon_id, g in groups:
    #                 if shown >= MAX_SHOW:
    #                     break
    #
    #                 top1_share = float(g["top1_share"].iloc[
    #                                        0]) if "top1_share" in g.columns and len(
    #                     g) else float("nan")
    #                 cluster_total = int(g["cluster_edge_media_count"].iloc[
    #                                         0]) if "cluster_edge_media_count" in g.columns else -1
    #
    #                 canon_name_norm = (
    #                     g["location_name_norm"].dropna().astype(str).iloc[0]
    #                     if "location_name_norm" in g.columns and g[
    #                         "location_name_norm"].notna().any()
    #                     else ""
    #                 )
    #
    #                 orig_pairs = (
    #                     g[["original_location_id",
    #                        "original_location_name_norm"]]
    #                     .drop_duplicates()
    #                     .sort_values("original_location_id")
    #                 )
    #
    #                 print("\n----------------------------------------")
    #                 print(f"Canonical location_id: {int(canon_id)}")
    #                 if canon_name_norm != "":
    #                     print(f"Canonical name_norm  : '{canon_name_norm}'")
    #                 if cluster_total >= 0:
    #                     print(f"Cluster total (count): {cluster_total}")
    #                 if not np.isnan(top1_share):
    #                     print(f"Top1 share           : {top1_share:.2f}")
    #                 print(f"Replaced POIs        : {len(orig_pairs)}")
    #
    #                 for _, r in orig_pairs.iterrows():
    #                     oid = r["original_location_id"]
    #                     oname = r["original_location_name_norm"]
    #                     print(
    #                         f"  - original_location_id={int(oid)}, name_norm='{oname}'")
    #
    #                 shown += 1
    #
    #             if shown == 0:
    #                 print("[Canonical POI sanity] No merged examples to show.")

    # Remove bot users
    prev = media
    counts_per_user = media["owner_id"].value_counts()
    suspicious_users = counts_per_user[counts_per_user > BOT_ACTIVITY_THRESHOLD].index.tolist()
    bot_users = []
    for uid in suspicious_users:
        if user_has_unrealistic_speed(media, locs, uid, time_col_for_filter, BOT_SPEED_KMH):
            bot_users.append(uid)
    bot_users = set(bot_users)
    media = media[~media["owner_id"].isin(bot_users)].copy()
    analyze_statistics(baseline_media, prev, media, label="Remove bot users")

    # Collapse consecutive same-location posts (visit-level dedup)
    prev = media
    media = collapse_consecutive_same_location(
        media_df=media,
        time_col=time_col_for_filter,
        min_gap_days=VISIT_GAP_DAYS,
        keep=VISIT_KEEP,
        verbose=True,
        print_every=2000
    )
    analyze_statistics(baseline_media, prev, media,
                       label=f"Collapse same-location visits (< {VISIT_GAP_DAYS} days, keep={VISIT_KEEP})")

    # =========================================================
    # Joint interactive filtering
    #   user >= 10 interactions
    #   POI  >= 5 photos
    #   POI  >= 3 distinct users (photo-based)
    # =========================================================
    prev = media

    media = iterative_joint_filter(
        media_df=media,
        min_user_interactions=MIN_NUM_INTERACTIONS,  # = 10
        min_photos=5,
        min_users_per_poi=3,
        max_iters=50,
        verbose=True
    )

    analyze_statistics(
        baseline_media,
        prev,
        media,
        label="Joint interactive filter (user>=10, POI>=5 photos & >=3 users)"
    )

    # Prepare filtered frames for final outputs
    kept_users = set(media["owner_id"].dropna().astype("Int64").unique())
    kept_locs  = set(media["location_id"].dropna().astype("Int64").unique())

    media_filtered = media.copy()

    # =========================================================
    # Photo coverage stats per business (after all filtering)
    # =========================================================
    print("\n[Photo coverage] Computing business photo coverage and photo count stats...")

    if "location_id" not in media_filtered.columns:
        print("WARN: media_filtered has no location_id, cannot compute business-level photo stats.")
    else:
        if "is_photo" not in media_filtered.columns:
            media_filtered = _add_is_photo(media_filtered)

        mf = media_filtered.copy()
        mf["business_id"] = _to_int(mf["location_id"])

        if "is_photo" in mf.columns:
            mf["is_photo"] = mf["is_photo"].fillna(False).astype(bool)
        else:
            mf["is_photo"] = False

        businesses_all = mf["business_id"].dropna().unique()
        n_business_all = len(businesses_all)

        photo_cnt_per_biz = (
            mf.groupby("business_id", dropna=True)["is_photo"]
            .sum()
            .astype(int)
        )

        n_business_with_photo = int((photo_cnt_per_biz > 0).sum())
        ratio = (n_business_with_photo / n_business_all) if n_business_all > 0 else 0.0

        if n_business_all > 0:
            photo_cnt_per_biz = photo_cnt_per_biz.reindex(businesses_all, fill_value=0)
            min_photos = int(photo_cnt_per_biz.min())
            max_photos = int(photo_cnt_per_biz.max())
            mean_photos = float(photo_cnt_per_biz.mean())
        else:
            min_photos = 0
            max_photos = 0
            mean_photos = 0.0

        print(f"[Photo coverage] Businesses total: {n_business_all:,}")
        print(f"[Photo coverage] Businesses with >=1 photo: {n_business_with_photo:,}")
        print(f"[Photo coverage] Coverage ratio: {ratio:.4f}  ({ratio*100:.2f}%)")
        print(f"[Photo coverage] Photos per business (all businesses): min={min_photos}, max={max_photos}, mean={mean_photos:.2f}")

        if n_business_all > 0:
            q = photo_cnt_per_biz.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
            print("[Photo coverage] Photo count quantiles:")
            for k, v in q.items():
                print(f"  - q{int(k*100):02d}: {float(v):.2f}")

        # Distinct users per business among PHOTO posts
        if "owner_id" not in mf.columns:
            print("WARN: media_filtered has no owner_id, cannot compute distinct photo-users per business.")
        else:
            photo_only = mf[mf["is_photo"] == True].copy()
            if photo_only.empty:
                print("[Photo coverage] No photo rows in filtered media, skip distinct photo-user stats.")
            else:
                photo_users_per_biz = (
                    photo_only.groupby("business_id")["owner_id"]
                    .nunique()
                    .astype(int)
                )
                photo_users_per_biz = photo_users_per_biz.reindex(businesses_all, fill_value=0)

                min_u = int(photo_users_per_biz.min()) if len(photo_users_per_biz) else 0
                max_u = int(photo_users_per_biz.max()) if len(photo_users_per_biz) else 0
                mean_u = float(photo_users_per_biz.mean()) if len(photo_users_per_biz) else 0.0

                print(f"[Photo coverage] Distinct photo-users per business (all businesses): min={min_u}, max={max_u}, mean={mean_u:.2f}")

                q_u = photo_users_per_biz.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
                print("[Photo coverage] Distinct photo-user count quantiles:")
                for k, v in q_u.items():
                    print(f"  - q{int(k*100):02d}: {float(v):.2f}")

    users_f = users[users["id"].isin(kept_users)].copy() if "id" in users.columns else users.iloc[0:0].copy()
    locs_f  = locs[locs["id"].isin(kept_locs)].copy()   if "id" in locs.columns  else locs.iloc[0:0].copy()
    if not locs_f.empty and "city_id" in locs_f.columns and "id" in cities.columns:
        kept_cities = set(locs_f["city_id"].dropna().astype("Int64").unique())
        cities_f = cities[cities["id"].isin(kept_cities)].copy()
    else:
        cities_f = cities.iloc[0:0].copy()

    # ---------- (1) business.csv ----------
    # post_count from media grouped by location_id
    post_cnt = (
        media_filtered.groupby("location_id")
        .size()
        .rename("post_count")
        .reset_index()
    )

    business = locs_f.copy()
    if "id" in business.columns:
        business["id"] = _to_int(business["id"])

    business = business.merge(post_cnt, left_on="id", right_on="location_id", how="left")
    business["post_count"] = business["post_count"].fillna(0).astype(int)

    # business_id primary key
    business["business_id"] = business["id"]

    # city column should contain city_id
    if "city_id" in business.columns:
        business["city"] = business["city_id"]
    else:
        business["city"] = pd.Series(dtype="Int64")

    # attach normalized latitude/longitude columns if not already there
    lat_col, lon_col = _detect_lat_lon_cols(business)
    if lat_col and "latitude" not in business.columns:
        business["latitude"] = business[lat_col]
    if lon_col and "longitude" not in business.columns:
        business["longitude"] = business[lon_col]

    # drop id/location_id/photo_count
    drop_cols = [c for c in ["id", "location_id", "photo_count"] if c in business.columns]
    business.drop(columns=drop_cols, inplace=True)

    # ensure name/address columns exist
    if "name" not in business.columns:
        name_col = _detect_loc_name_col(business)
        if name_col:
            business.rename(columns={name_col: "name"}, inplace=True)
        else:
            business["name"] = ""
    if "address" not in business.columns:
        business["address"] = ""

    core_cols = ["business_id", "name", "address", "city", "latitude", "longitude", "post_count"]
    remain_cols = [c for c in business.columns if c not in core_cols]
    business_out = business[core_cols + remain_cols]
    save_csv(business_out, OUTPUT_DIR, "business.csv")

    # ---------- (2) city.csv ----------
    save_csv(cities_f, OUTPUT_DIR, "city.csv")

    # ---------- (3) checkin.csv ----------
    chk = media_filtered.copy()
    if "taken_at_timestamp" not in chk.columns:
        missing_count = len(chk)
        print(f"\nWARN: 'taken_at_timestamp' column NOT FOUND. All {missing_count:,} rows considered missing for checkin 'date'.")
        date_series = pd.Series([""] * len(chk))
    else:
        chk["taken_at_timestamp"] = _ensure_datetime_utc(chk["taken_at_timestamp"])
        missing_count = chk["taken_at_timestamp"].isna().sum()
        if missing_count > 0:
            print(f"\nINFO: 'taken_at_timestamp' missing for {missing_count:,} rows in filtered media.")
        date_series = chk["taken_at_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z").fillna("")

    checkin = pd.DataFrame({
        "post_id":     _to_int(chk["id"]) if "id" in chk.columns else pd.Series(dtype="Int64"),
        "user_id":     _to_int(chk["owner_id"]) if "owner_id" in chk.columns else pd.Series(dtype="Int64"),
        "business_id": _to_int(chk["location_id"]) if "location_id" in chk.columns else pd.Series(dtype="Int64"),
        "date":        date_series
    })
    save_csv(checkin, OUTPUT_DIR, "checkin.csv")

    # ---------- (4) user.csv ----------
    save_csv(users_f, OUTPUT_DIR, "user.csv")

    # ---------- (5) media.csv ----------
    # keep multimedia-style columns, add user_id/business_id; DO NOT keep location_id
    base_cols = [c for c in ["id", "caption", "media_downloaded_name", "is_photo"] if c in media_filtered.columns]
    media_out = media_filtered[["id"] + base_cols[1:]].drop_duplicates().copy()

    media_out["user_id"]     = _to_int(media_filtered["owner_id"]) if "owner_id" in media_filtered.columns else pd.Series(dtype="Int64")
    media_out["business_id"] = _to_int(media_filtered["location_id"]) if "location_id" in media_filtered.columns else pd.Series(dtype="Int64")

    # merge aggregated comments if comment.csv exists (read from INPUT dir)
    cmt = _safe_read_csv(RAW_DIR / "comment.csv")
    if not cmt.empty:
        c_fk = _detect_first_existing(cmt, ["media_id", "post_id", "id"])
        c_txt = _detect_first_existing(cmt, ["text", "comment_text", "body", "content"])
        c_tim = _detect_first_existing(cmt, ["created_time", "created_at", "time", "timestamp"])
        if c_fk and c_txt:
            g = cmt.dropna(subset=[c_fk, c_txt]).copy()
            if c_tim:
                g[c_tim] = pd.to_datetime(g[c_tim], errors="coerce", utc=True)
                g = g.sort_values([c_fk, c_tim])
            comment_agg = (
                g.groupby(c_fk, as_index=False)[c_txt]
                .apply(lambda s: " || ".join([str(t) for t in s.dropna()]))
                .rename(columns={c_fk: "id", c_txt: "comments"})
            )
            media_out["id"]   = _to_int(media_out["id"])
            comment_agg["id"] = _to_int(comment_agg["id"])
            if media_out["id"].isna().all() or comment_agg["id"].isna().all():
                media_out["id"]   = media_out["id"].astype(str).str.strip()
                comment_agg["id"] = comment_agg["id"].astype(str).str.strip()
            media_out = media_out.merge(comment_agg, on="id", how="left")
        else:
            media_out["comments"] = ""
    else:
        media_out["comments"] = ""

    for col in ["caption", "media_downloaded_name", "comments"]:
        if col in media_out.columns:
            media_out[col] = media_out[col].fillna("")

    if "location_id" in media_out.columns:
        media_out.drop(columns=["location_id"], inplace=True)

    save_csv(media_out, OUTPUT_DIR, "media.csv")

    # -------------------------
    # Final confirmation prints
    # -------------------------
    print("\nSaved files (under output/):")
    print(f"  - {OUTPUT_DIR/'business.csv'}")
    print(f"  - {OUTPUT_DIR/'city.csv'}")
    print(f"  - {OUTPUT_DIR/'checkin.csv'}")
    print(f"  - {OUTPUT_DIR/'user.csv'}")
    print(f"  - {OUTPUT_DIR/'media.csv'}")

if __name__ == "__main__":
    main()