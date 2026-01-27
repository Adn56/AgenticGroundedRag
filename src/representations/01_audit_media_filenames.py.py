#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
Build local media index from media.csv (photos only)
============================================================

Output:
- media_id,business_id,local_image_path

Hard requirement:
- 100% of is_photo == True images must exist locally,
  otherwise the script fails.
"""

from pathlib import Path
import pandas as pd

# ============================================================
# PATHS (repo-relative)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEDIA_CSV = PROJECT_ROOT / "Dataset/output_poi_preprocessing/media.csv"

LOCAL_IMAGE_DIR = Path(
    "/Users/adrianhoffmann/Desktop/instagram_extracted_images"
)

OUT_CSV = PROJECT_ROOT / "Dataset/representations/media_local_index.csv"

# ============================================================
def main():
    df = pd.read_csv(MEDIA_CSV)

    required_cols = {"id", "business_id", "media_downloaded_name", "is_photo"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"media.csv missing columns: {missing}")

    # normalize is_photo
    if df["is_photo"].dtype != bool:
        df["is_photo"] = df["is_photo"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

    df = df[df["is_photo"]].copy()

    if df.empty:
        raise RuntimeError("No photo entries found in media.csv.")

    # build local paths
    df["filename"] = df["media_downloaded_name"].astype(str).apply(
        lambda x: Path(x).name
    )
    df["local_image_path"] = df["filename"].apply(
        lambda f: str(LOCAL_IMAGE_DIR / f)
    )

    # existence check
    exists_mask = df["local_image_path"].apply(lambda p: Path(p).exists())
    missing_count = (~exists_mask).sum()
    total = len(df)

    if missing_count > 0:
        found_pct = 100.0 * (total - missing_count) / total
        raise RuntimeError(
            f"Only {found_pct:.2f}% of image files referenced in media.csv "
            f"are present locally ({missing_count} missing)."
        )

    # final output
    out_df = df[["id", "business_id", "local_image_path"]].rename(
        columns={"id": "media_id"}
    )

    out_df.to_csv(OUT_CSV, index=False)

    # console output (exact wording)
    print("100% of image files referenced in media.csv are present locally.")


# ============================================================
if __name__ == "__main__":
    main()
