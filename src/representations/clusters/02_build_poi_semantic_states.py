#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build canonical POI semantic state representation (Reviewer-clean)

- loads LLM-generated POI descriptions
- loads fixed POI → cluster assignments (k=16)
- merges both 1:1 by business_id
- writes canonical POI semantic state dataset

No clustering. No filtering. No aggregation.
This dataset serves as the primary semantic state space for LLM/RAG.
"""

import json
from pathlib import Path

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / "Dataset"

DESC_PATH = DATASET / "representations" / "poi_descriptions_llm.jsonl"
CLUSTER_PATH = DATASET / "representations" / "Cluster" / "poi_cluster_assignment.csv"

OUT_DIR = DATASET / "core"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "poi_semantic_states.jsonl"

# =========================
# LOAD CLUSTERS
# =========================
def load_cluster_mapping(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            poi, cluster = line.strip().split(",")
            mapping[int(poi)] = int(cluster)
    return mapping

# =========================
# MAIN
# =========================
def main():
    poi_to_cluster = load_cluster_mapping(CLUSTER_PATH)

    written = 0
    missing_cluster = 0

    with open(DESC_PATH, "r", encoding="utf-8") as fin, \
            open(OUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            obj = json.loads(line)
            poi_id = int(obj["business_id"])

            cluster_id = poi_to_cluster.get(poi_id)
            if cluster_id is None:
                missing_cluster += 1
                continue  # bewusst: nur POIs mit definierter Cluster-Zugehörigkeit

            obj_out = dict(obj)
            obj_out["cluster_id"] = cluster_id

            fout.write(json.dumps(obj_out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written POI semantic states: {written}")
    print(f"POIs without cluster assignment skipped: {missing_cluster}")
    print(f"Output written to: {OUT_PATH}")

if __name__ == "__main__":
    main()
