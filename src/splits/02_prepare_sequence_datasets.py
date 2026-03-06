#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FULL DATASET PREPARATION PIPELINE

TRAIN:
- raw trajectories
- target = last POI
- sequences shorter than WINDOW+1 are DROPPED

VALIDATION:
- target already provided
- STRICT (no dropping)

WINDOW = 9
"""

import json
import csv
from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# CONFIG
# ============================================================

WINDOW = 9

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "Dataset" / "splits"

CLUSTER_PATH = PROJECT_ROOT / "Dataset/representations/Cluster/poi_cluster_assignment.csv"

SPLITS = {
    "test": SPLIT_DIR / "test.jsonl",
    "validation": SPLIT_DIR / "validation.jsonl",
}

# ============================================================
# IO
# ============================================================

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ============================================================
# CLUSTER MAP
# ============================================================

def load_cluster_map(path: Path):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            m[str(r["poi_id"])] = int(r["cluster_id"])
    return m

# ============================================================
# TEMPORAL
# ============================================================

def parse_iso(ts: str):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)

def enrich_event(ev: dict):
    ts = ev.get("date")
    if not ts:
        return ev
    dt = parse_iso(ts)
    ev["date_day"]  = dt.strftime("%A")
    ev["date_ymd"]  = dt.strftime("%Y-%m-%d")
    ev["date_time"] = dt.strftime("%H:%M:%S")
    return ev

# ============================================================
# PIPELINE
# ============================================================

def process_split(name: str, path: Path, cluster_map: dict):
    rows = load_jsonl(path)

    llm_rows = []
    cluster_rows = []
    dropped = 0

    for i, row in enumerate(rows):
        user_id = row.get("user_id")
        traj = row.get("trajectory", [])

        # ---------------------------
        # TARGET HANDLING
        # ---------------------------
        if name == "train":
            if len(traj) < WINDOW + 1:
                dropped += 1
                continue
            target = traj[-1]
            traj = traj[:-1]
        else:
            target = row.get("target")
            if not target or "business_id" not in target:
                raise KeyError(f"[{name}][Row {i}] Missing target")

        # ---------------------------
        # LEAKAGE FLAG
        # ---------------------------
        target_id = str(target["business_id"])
        last_ids = {str(ev["business_id"]) for ev in traj[-WINDOW:]}
        seen_flag = target_id in last_ids

        # ---------------------------
        # TEMPORAL
        # ---------------------------
        for ev in traj:
            enrich_event(ev)
        enrich_event(target)

        # ---------------------------
        # LAST-9
        # ---------------------------
        if len(traj) < WINDOW:
            if name == "train":
                dropped += 1
                continue
            else:
                raise ValueError(f"[{name}][Row {i}] Sequence < {WINDOW}")

        last9 = traj[-WINDOW:]

        llm_rows.append({
            "user_id": user_id,
            "input": last9,
            "target": target,
            "target_seen_in_last_9": seen_flag
        })

        # ---------------------------
        # CLUSTER-ONLY
        # ---------------------------
        input_clusters = []
        for ev in last9:
            pid = str(ev["business_id"])
            if pid not in cluster_map:
                raise KeyError(f"[{name}][Row {i}] POI {pid} not in cluster map")
            input_clusters.append(cluster_map[pid])

        if target_id not in cluster_map:
            raise KeyError(f"[{name}][Row {i}] Target {target_id} not in cluster map")

        cluster_rows.append({
            "user_id": user_id,
            "input": input_clusters,
            "target": cluster_map[target_id],
            "target_seen_in_last_9": seen_flag
        })

    llm_out = path.with_name(f"{name}_last9_input.jsonl")
    cluster_out = path.with_name(f"{name}_cluster_only.jsonl")

    write_jsonl(llm_out, llm_rows)
    write_jsonl(cluster_out, cluster_rows)

    return llm_out, cluster_out, len(llm_rows), dropped

# ============================================================
# RUN
# ============================================================

def main():
    print("\nFULL DATASET PREPARATION PIPELINE\n")

    cluster_map = load_cluster_map(CLUSTER_PATH)
    print(f"[OK] Loaded {len(cluster_map)} cluster assignments\n")

    for name, path in SPLITS.items():
        if not path.exists():
            print(f"[SKIP] {name} not found")
            continue

        llm_out, cl_out, kept, dropped = process_split(name, path, cluster_map)

        print(f"[{name.upper()}]")
        print(f"  Input        : {path.name}")
        print(f"  LLM dataset  : {llm_out.name}")
        print(f"  Cluster only : {cl_out.name}")
        print(f"  Kept rows    : {kept}")
        print(f"  Dropped rows : {dropped}\n")

if __name__ == "__main__":
    main()
