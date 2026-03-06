#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster-level attribute frequency profiles

For each cluster:
- aggregates structured attributes across all POIs
- selects top-k most frequent attribute phrases per category
- outputs structured semantic profiles per cluster

"""

import json
from pathlib import Path
from collections import defaultdict, Counter

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / "Dataset" / "core"

IN_PATH = DATASET / "poi_semantic_states.jsonl"

OUT_PATH = DATASET / "cluster_attribute_profiles.jsonl"

# =========================
# CONFIG
# =========================
TOP_K = 5   # oder 10, wie du willst

ATTRIBUTE_FIELDS = [
    "spatial_configuration",
    "style",
    "atmosphere",
    "target_group",
    "distinctive_features"
]

# =========================
# AGGREGATION
# =========================
cluster_counters = defaultdict(lambda: {
    field: Counter() for field in ATTRIBUTE_FIELDS
})

with open(IN_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        cid = obj["cluster_id"]
        attrs = obj.get("attributes", {})

        for field in ATTRIBUTE_FIELDS:
            for phrase in attrs.get(field, []):
                cluster_counters[cid][field][phrase] += 1

# =========================
# WRITE OUTPUT
# =========================
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for cid in sorted(cluster_counters.keys()):
        out_obj = {
            "cluster_id": cid,
            "attributes": {}
        }

        for field in ATTRIBUTE_FIELDS:
            most_common = cluster_counters[cid][field].most_common(TOP_K)
            out_obj["attributes"][field] = [term for term, _ in most_common]

        f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

print(f"Cluster attribute profiles written to: {OUT_PATH}")
