#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENERATE EMBEDDINGS FOR ALL HYPOTHESES 

- Reads: hypotheses_seed2026.jsonl
- For EACH row:
    * extracts hypothesis["summary"]
    * embeds summary
- Writes:
    hypotheses_seed2026_embeddings.csv

Guarantees:
- exactly one embedding per hypothesis
- no partial state
"""

import csv
import json
from pathlib import Path
from openai import OpenAI

# ============================================================
# PATHS
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "results"

JSONL_INPUT = (
        DATASET_DIR / "hypothesis_test_eval_1000.jsonl"
)

CSV_OUTPUT = (
        DATASET_DIR / "hypothesis_test_eval_1000_embeddings.csv"
)

# ============================================================
# CONFIG
# ============================================================

EMBED_MODEL = "text-embedding-3-large"

# ============================================================
# INIT
# ============================================================

client = OpenAI()

# ============================================================
# LOAD JSONL INPUT
# ============================================================

rows = []
with open(JSONL_INPUT, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

print(f"Loaded {len(rows)} hypotheses")

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

out_rows = []

for i, r in enumerate(rows, start=1):

    user_id = r["user_id"]

    hypothesis = r["hypothesis"]
    text = hypothesis.get("summary")

    if not text:
        raise ValueError(f"No summary found for user_id={user_id}")

    emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )

    vec = emb.data[0].embedding

    out_rows.append({
        "user_id": user_id,
        "embedding": json.dumps(vec)
    })

    if i % 10 == 0 or i == len(rows):
        print(f"Embedded {i}/{len(rows)} hypotheses")

# ============================================================
# WRITE OUTPUT 
# ============================================================

with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["user_id", "embedding"])
    writer.writeheader()
    writer.writerows(out_rows)

print("\nDone.")
print(f"Written {len(out_rows)} embeddings to:")
print(CSV_OUTPUT)
