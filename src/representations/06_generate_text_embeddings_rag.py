#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 06 – TEXT EMBEDDINGS FOR RAG (BATCHED, ROBUST, PAPER-READY)

This script performs the offline text-embedding step for a Retrieval-Augmented
Generation (RAG) pipeline.

Design goals:
- Deterministic, single-pass preprocessing
- Explicit error handling and auditability
- Resume capability (safe interruption at any time)
- High throughput via batch embedding
- Clear separation between representation and retrieval layers

INPUT:
Dataset/representations/poi_descriptions_llm.jsonl

OUTPUT:
Dataset/representations/poi_text_embeddings.jsonl   (per-POI embedding + status)
Dataset/representations/poi_text_index.faiss        (FAISS cosine similarity index)
Dataset/representations/poi_text_meta.jsonl         (FAISS row → business_id mapping)
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_EMBEDDING = "text-embedding-3-large"

BATCH_SIZE = 64               # main speed lever (32–128 safe)
MAX_RETRIES = 5
BACKOFF_BASE_SEC = 1.5        # exponential backoff

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REP_DIR = PROJECT_ROOT / "Dataset" / "representations"

INPUT_JSONL = REP_DIR / "poi_descriptions_llm.jsonl"

OUT_EMB_JSONL = REP_DIR / "poi_text_embeddings.jsonl"
OUT_INDEX = REP_DIR / "poi_text_index.faiss"
OUT_META = REP_DIR / "poi_text_meta.jsonl"


# ============================================================
# INITIALIZATION
# ============================================================

client = OpenAI()
REP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_done_ids(path: Path) -> set:
    """
    Load business_ids already processed in a previous run.
    Enables deterministic resume behaviour.
    """
    done = set()
    if not path.exists():
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("status") in ("ok", "error"):
                    done.add(obj.get("business_id"))
            except Exception:
                continue
    return done


def embed_batch_with_retries(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts with retry logic and exponential backoff.
    Returns a list of embedding vectors.
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(
                model=MODEL_EMBEDDING,
                input=texts
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF_BASE_SEC ** attempt)
    raise last_err


# ============================================================
# MAIN – EMBEDDING PHASE
# ============================================================

if not INPUT_JSONL.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

done_ids = load_done_ids(OUT_EMB_JSONL)

ok_count = 0
err_count = 0
skipped_count = 0

pending: List[Dict] = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(OUT_EMB_JSONL, "a", encoding="utf-8") as out_f:
    for line in tqdm(lines, desc="Embedding POI summaries (batched)"):
        obj = json.loads(line)

        business_id = obj.get("business_id")
        representation_id = obj.get("representation_id")
        summary = (obj.get("summary") or "").strip()

        if business_id in done_ids:
            skipped_count += 1
            continue

        if not summary:
            err_count += 1
            out_f.write(json.dumps({
                "business_id": business_id,
                "representation_id": representation_id,
                "status": "error",
                "error": "empty_summary"
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            continue

        pending.append({
            "business_id": business_id,
            "representation_id": representation_id,
            "summary": summary
        })

        if len(pending) >= BATCH_SIZE:
            try:
                embeddings = embed_batch_with_retries(
                    [p["summary"] for p in pending]
                )

                for p, emb in zip(pending, embeddings):
                    ok_count += 1
                    out_f.write(json.dumps({
                        "business_id": p["business_id"],
                        "representation_id": p["representation_id"],
                        "status": "ok",
                        "embedding": emb
                    }, ensure_ascii=False) + "\n")

                out_f.flush()
                pending.clear()

            except Exception as e:
                err_count += len(pending)
                for p in pending:
                    out_f.write(json.dumps({
                        "business_id": p["business_id"],
                        "representation_id": p["representation_id"],
                        "status": "error",
                        "error": str(e)[:500]
                    }, ensure_ascii=False) + "\n")
                out_f.flush()
                pending.clear()

# Process remaining items
if pending:
    embeddings = embed_batch_with_retries([p["summary"] for p in pending])
    for p, emb in zip(pending, embeddings):
        ok_count += 1
        with open(OUT_EMB_JSONL, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps({
                "business_id": p["business_id"],
                "representation_id": p["representation_id"],
                "status": "ok",
                "embedding": emb
            }, ensure_ascii=False) + "\n")

print("\n=== EMBEDDING RUN SUMMARY ===")
print(f"ok:      {ok_count}")
print(f"errors:  {err_count}")
print(f"skipped: {skipped_count}")
print(f"output:  {OUT_EMB_JSONL}")


# ============================================================
# FAISS INDEX CONSTRUCTION
# ============================================================

successful = []
with open(OUT_EMB_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row.get("status") == "ok" and row.get("embedding"):
            successful.append(row)

if not successful:
    raise RuntimeError("No successful embeddings found. FAISS index not created.")

X = np.array([r["embedding"] for r in successful], dtype="float32")
faiss.normalize_L2(X)

index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
faiss.write_index(index, str(OUT_INDEX))

with open(OUT_META, "w", encoding="utf-8") as f:
    for i, r in enumerate(successful):
        f.write(json.dumps({
            "faiss_row": i,
            "business_id": r["business_id"],
            "representation_id": r.get("representation_id")
        }, ensure_ascii=False) + "\n")

print("\n✅ FAISS index created")
print(f"vectors: {X.shape}")
print(f"index:   {OUT_INDEX}")
print(f"meta:    {OUT_META}")
