#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
02_generate_clip_embeddings_perf.py
============================================================

Fast CLIP image embeddings on macOS (MPS), minimal overhead.

Input (required):
- Dataset/representations/media_local_index.csv
  columns: media_id,business_id,local_image_path

Output:
- Dataset/representations/clip_embeddings_analyse/
  - embeddings_fp16.memmap
  - embeddings_index.csv
  - run_report.json

Notes:
- NO normalization (raw embeddings)
- First run default: 1000 images (adjust with N_IMAGES)
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", message="Using a slow image processor")



ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated JPEGs


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "openai/clip-vit-base-patch32"
EMB_DIM = 512

# First run requirement
N_IMAGES = None
SEED = 42

# Performance knobs (M4 Air sweet-spot)
BATCH_SIZE = 128     # you already validated this is faster; try 160/192 later if you want
NUM_WORKERS = 4      # 4-8 ok on macOS; 4 is stable
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True

USE_FP16_AUTOCAST_ON_MPS = True  # speed, no meaningful quality loss for embeddings storage


# ============================================================
# PATHS (repo)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IN_CSV = PROJECT_ROOT / "Dataset/representations/media_local_index.csv"

OUT_DIR = PROJECT_ROOT / "Dataset/representations/clip_embeddings_analyse"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_MEMMAP_PATH = OUT_DIR / "embeddings_fp16.memmap"
INDEX_CSV_PATH  = OUT_DIR / "embeddings_index.csv"
REPORT_JSON_PATH = OUT_DIR / "run_report.json"


# ============================================================
# Helpers
# ============================================================
def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def open_rgb(path: str):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


# ============================================================
# Dataset
# ============================================================
class ImgDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.media_id = df["media_id"].tolist()
        self.business_id = df["business_id"].tolist()
        self.path = df["local_image_path"].tolist()

    def __len__(self):
        return len(self.path)

    def __getitem__(self, i: int):
        img = open_rgb(self.path[i])
        return img, self.media_id[i], self.business_id[i], self.path[i]


def collate(batch):
    imgs, mids, bids, paths = [], [], [], []
    skipped = 0
    for img, mid, bid, p in batch:
        if img is None:
            skipped += 1
            continue
        imgs.append(img)
        mids.append(mid)
        bids.append(bid)
        paths.append(p)
    return imgs, mids, bids, paths, skipped


# ============================================================
def main():
    t0 = time.time()

    # CPU thread oversubscription on macOS can hurt
    try:
        cpu = os.cpu_count() or 8
        torch.set_num_threads(max(1, cpu // 2))
    except Exception:
        pass

    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    needed = {"media_id", "business_id", "local_image_path"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {sorted(needed)}")

    # First run: only 1000 images (deterministic)
    df = df.dropna(subset=["local_image_path"]).copy()

    if N_IMAGES is not None:
        if len(df) < N_IMAGES:
            raise ValueError(f"Not enough rows in input: {len(df)} < {N_IMAGES}")
        df = df.sample(n=N_IMAGES, random_state=SEED).reset_index(drop=True)
    else:
        # use ALL images
        df = df.reset_index(drop=True)


    device = pick_device()
    print("Device:", device)

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    use_amp = (device == "mps" and USE_FP16_AUTOCAST_ON_MPS)

    # memmap output
    n = len(df)
    emb_mm = np.memmap(EMB_MEMMAP_PATH, mode="w+", dtype=np.float16, shape=(n, EMB_DIM))

    ds = ImgDataset(df)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        pin_memory=False,
    )

    index_rows = []
    write_pos = 0
    skipped_total = 0

    # tqdm progress by embeddings written (not by batches)
    pbar = tqdm(total=n, desc="Embedding", unit="img")

    with torch.inference_mode():
        for imgs, mids, bids, paths, skipped in dl:
            skipped_total += skipped

            if not imgs:
                continue

            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if use_amp:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    feats = model.get_image_features(**inputs)
            else:
                feats = model.get_image_features(**inputs)

            # RAW embeddings only (no normalization)
            feats_np = feats.detach().to("cpu").to(torch.float16).numpy()
            b = feats_np.shape[0]

            emb_mm[write_pos:write_pos + b, :] = feats_np
            # flush rarely; too frequent flush slows down
            if write_pos % (BATCH_SIZE * 16) == 0:
                emb_mm.flush()

            # index mapping
            for i in range(b):
                index_rows.append({
                    "emb_index": write_pos + i,
                    "media_id": mids[i],
                    "business_id": bids[i],
                    "local_image_path": paths[i],
                })

            write_pos += b
            pbar.update(b)

    pbar.close()
    emb_mm.flush()

    # Save index CSV
    pd.DataFrame(index_rows).to_csv(INDEX_CSV_PATH, index=False)

    runtime = time.time() - t0

    report = {
        "device": device,
        "model": MODEL_NAME,
        "n_requested":len(df) if N_IMAGES is None else int(N_IMAGES),
        "n_written": int(write_pos),
        "skipped_images": int(skipped_total),
        "batch_size": int(BATCH_SIZE),
        "num_workers": int(NUM_WORKERS),
        "dtype": "float16",
        "normalized": False,
        "output_dir": str(OUT_DIR),
        "runtime_seconds": round(runtime, 3),
        "throughput_img_per_s": round(write_pos / runtime, 3) if runtime > 0 else None,
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n✔ Done.")
    print("✔ Saved embeddings (memmap):", EMB_MEMMAP_PATH)
    print("✔ Saved index CSV          :", INDEX_CSV_PATH)
    print("✔ Saved report JSON        :", REPORT_JSON_PATH)
    print(f"✔ Written embeddings       : {write_pos} / {N_IMAGES}")
    print(f"✔ Skipped unreadable images: {skipped_total}")
    print(f"✔ Runtime (seconds)        : {report['runtime_seconds']}")
    print(f"✔ Throughput (img/s)       : {report['throughput_img_per_s']}")


if __name__ == "__main__":
    main()
