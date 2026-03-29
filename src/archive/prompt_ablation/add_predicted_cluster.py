#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLMC CLUSTER PREDICTION ATTACHMENT (MULTI-SEED)

- Reads multiple validation_dev_100_seed*.jsonl files
- Predicts NEXT CLUSTER via VLMC (longest-prefix backoff)
- Attaches ONLY:
    - predicted_cluster
    - used_k
- Does NOT evaluate correctness
- Does NOT touch target or hypothesis logic
- Writes one output file per seed
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

K_MAX = 9
MIN_K_ALLOWED = 2

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "Dataset"

POI_STATES = DATASET / "core" / "poi_semantic_states.jsonl"
FINAL_PATTERNS = DATASET / "core" / "sequential_patterns_FINAL.csv"

SPLIT_DIR = DATASET / "splits"
OUT_DIR = PROJECT_ROOT / "prompt_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD business_id → cluster_id
# ============================================================

def load_cluster_map(path: Path):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            bid = obj.get("business_id")
            cid = obj.get("cluster_id")
            if bid is not None and cid is not None:
                mapping[int(bid)] = int(cid)
    return mapping

# ============================================================
# LOAD FINAL VLMC PATTERNS
# ============================================================

def load_final_patterns(path: Path):
    patterns_by_k = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            k = int(r["k"])
            prefix = tuple(map(int, r["prefix"].split("-")))
            pred = int(r["pred_next"])
            patterns_by_k[k][prefix] = pred
    return patterns_by_k

# ============================================================
# VLMC PREDICTION
# ============================================================

def predict_cluster(cluster_seq, patterns_by_k):
    for k in range(min(K_MAX, len(cluster_seq)), MIN_K_ALLOWED - 1, -1):
        prefix = tuple(cluster_seq[-k:])
        if prefix in patterns_by_k.get(k, {}):
            return patterns_by_k[k][prefix], k
    return None, 0

# ============================================================
# MAIN
# ============================================================

def main():
    cluster_map = load_cluster_map(POI_STATES)
    patterns = load_final_patterns(FINAL_PATTERNS)

    print("[INFO] Loaded cluster map and VLMC patterns\n")

    in_path = SPLIT_DIR / f"test_eval_1000.jsonl"
    out_path = OUT_DIR / f"test_eval_1000_with_predicted_cluster.jsonl"

    print(f"[PROCESS] {in_path.name}")
    with open(in_path, encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            # --------------------------------------------
            # Map input POIs → cluster sequence
            # --------------------------------------------
            cluster_seq = []
            valid = True

            for e in obj.get("input", []):
                bid = e.get("business_id") if isinstance(e, dict) else None
                if bid is None:
                    valid = False
                    break
                bid = int(bid)
                if bid not in cluster_map:
                    valid = False
                    break
                cluster_seq.append(cluster_map[bid])

            # --------------------------------------------
            # Predict next cluster
            # --------------------------------------------
            if not valid or len(cluster_seq) < MIN_K_ALLOWED:
                obj["predicted_cluster"] = None
                obj["used_k"] = 0
            else:
                pred, used_k = predict_cluster(cluster_seq, patterns)
                obj["predicted_cluster"] = pred
                obj["used_k"] = used_k

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[DONE]  → {out_path.name}\n")

if __name__ == "__main__":
    main()