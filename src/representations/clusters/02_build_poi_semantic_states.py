#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build canonical POI semantic state representation

- loads LLM-generated POI descriptions
- loads fixed POI → cluster assignments (k=16)
- loads business metadata (name, lat, lon, city_id)
- loads city metadata (city_name)
- merges all 1:1 by business_id
- writes canonical POI semantic state dataset

No clustering. No filtering. No aggregation.
Primary semantic state space for LLM/RAG.
"""

import json
import csv
from pathlib import Path

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET = PROJECT_ROOT / "Dataset"

DESC_PATH = DATASET / "representations" / "poi_descriptions_llm.jsonl"
CLUSTER_PATH = DATASET / "representations" / "Cluster" / "poi_cluster_assignment.csv"

BUSINESS_PATH = DATASET / "output_poi_preprocessing" / "business.csv"
CITY_PATH = DATASET / "output_poi_preprocessing" / "city.csv"

OUT_DIR = DATASET / "core"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "poi_semantic_states.jsonl"


# =========================
# LOAD CLUSTERS
# =========================
def load_cluster_mapping(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            poi, cluster = line.strip().split(",")
            mapping[int(poi)] = int(cluster)
    return mapping


# =========================
# LOAD BUSINESS METADATA
# =========================
def load_business_metadata(path):
    mapping = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poi_id = int(row["business_id"])

            mapping[poi_id] = {
                "name": row.get("name"),
                "latitude": float(row["latitude"]) if row.get("latitude") else None,
                "longitude": float(row["longitude"]) if row.get("longitude") else None,
                "city_id": int(row["city_id"]) if row.get("city_id") else None
            }

    return mapping


# =========================
# LOAD CITY METADATA
# =========================
def load_city_mapping(path):
    mapping = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            city_id = int(row["id"])
            mapping[city_id] = row.get("name")

    return mapping


# =========================
# MAIN
# =========================
def main():
    poi_to_cluster = load_cluster_mapping(CLUSTER_PATH)
    business_meta = load_business_metadata(BUSINESS_PATH)
    city_mapping = load_city_mapping(CITY_PATH)

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
                continue  # nur POIs mit definierter Cluster-Zugehörigkeit

            obj_out = dict(obj)
            obj_out["cluster_id"] = cluster_id

            # -------------------------
            # Business Metadata
            # -------------------------
            meta = business_meta.get(poi_id)
            if meta:
                obj_out["name"] = meta["name"]
                obj_out["latitude"] = meta["latitude"]
                obj_out["longitude"] = meta["longitude"]

                city_id = meta["city_id"]
                if city_id:
                    obj_out["city_name"] = city_mapping.get(city_id)

            fout.write(json.dumps(obj_out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written POI semantic states: {written}")
    print(f"POIs without cluster assignment skipped: {missing_cluster}")
    print(f"Output written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
