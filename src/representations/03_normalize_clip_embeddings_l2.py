#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
03_normalize_clip_embeddings_l2.py
============================================================

L2-normalization of CLIP image embeddings (memmap → memmap).

- Input : raw CLIP embeddings (fp16 memmap)
- Output: L2-normalized CLIP embeddings (fp16 memmap)
- Purpose: cosine similarity / facility location / retrieval

This script is:
- fast (O(n·d))
- memory-safe (no full load into RAM)
- deterministic
"""

import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
EMB_DIM = 512
EPS = 1e-12

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IN_MEMMAP = PROJECT_ROOT / "Dataset/representations/clip_embeddings_analyse/embeddings_fp16.memmap"
OUT_MEMMAP = PROJECT_ROOT / "Dataset/representations/clip_embeddings_analyse/embeddings_fp16_l2.memmap"

# Chunking (controls memory & speed)
CHUNK_SIZE = 8192  # rows per chunk; safe & fast


# ============================================================
def main():
    t0 = time.time()

    if not IN_MEMMAP.exists():
        raise FileNotFoundError(f"Input memmap not found: {IN_MEMMAP}")

    # Infer number of vectors from file size
    n_vectors = IN_MEMMAP.stat().st_size // (2 * EMB_DIM)  # fp16 = 2 bytes

    print(f"Normalizing {n_vectors} embeddings (dim={EMB_DIM})")

    X = np.memmap(
        IN_MEMMAP,
        dtype=np.float16,
        mode="r",
        shape=(n_vectors, EMB_DIM),
    )

    Y = np.memmap(
        OUT_MEMMAP,
        dtype=np.float16,
        mode="w+",
        shape=(n_vectors, EMB_DIM),
    )

    skipped = 0

    for start in tqdm(range(0, n_vectors, CHUNK_SIZE), desc="L2-normalizing"):
        end = min(start + CHUNK_SIZE, n_vectors)

        chunk = X[start:end].astype(np.float32)  # compute in fp32 for stability
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)

        # handle zero or invalid vectors defensively
        mask = norms > EPS
        if not np.all(mask):
            skipped += int((~mask).sum())
            norms[~mask] = 1.0  # avoid division by zero

        chunk = chunk / norms
        Y[start:end] = chunk.astype(np.float16)

    Y.flush()

    runtime = time.time() - t0

    print("\n✔ L2-normalization finished")
    print(f"✔ Output memmap : {OUT_MEMMAP}")
    print(f"✔ Skipped vectors (degenerate): {skipped}")
    print(f"✔ Runtime (seconds): {runtime:.3f}")
    print(f"✔ Throughput (vec/s): {int(n_vectors / runtime) if runtime > 0 else 'n/a'}")


if __name__ == "__main__":
    main()
