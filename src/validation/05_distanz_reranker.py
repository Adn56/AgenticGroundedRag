#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FULL PIPELINE (MULTI-SEED) + BASELINE + DISTANCE RE-RANKING

- Baseline: FAISS order
- Distance Re-Ranking: based on last POI → candidate distance
- Sweep over RAG_K
- Evaluate TopK thresholds
- Median rank tracking
"""

import json
import sys
import numpy as np
import pandas as pd
import faiss
import math

from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE = Path("/Users/adrianhoffmann/Documents/Master/Projektarbeit/AgenticGroundedRagClone")
DATA = BASE / "Dataset"

SEEDS = [
    #{
     #   "name": "seed2026",
      #  "hyp": DATA / "validation" / "hypotheses_seed2026_without_cluster_modell52.jsonl",
       # "emb": DATA / "validation" / "hypotheses_seed2026_without_cluster_modell52_embeddings.csv",
    #},
    {
        "name": "seed2027",
        "hyp": DATA / "validation" / "hypotheses_seed2027_without_cluster_modell52.jsonl",
        "emb": DATA / "validation" / "hypotheses_seed2027_without_cluster_modell52_embeddings.csv",
    },
    #{
     #   "name": "seed2028",
      #  "hyp": DATA / "validation" / "hypotheses_seed2028_without_cluster.jsonl",
       # "emb": DATA / "validation" / "hypotheses_seed2028_without_cluster_embeddings.csv",
    #}
]

POI_STATES  = DATA / "core" / "poi_semantic_states.jsonl"
FAISS_INDEX = DATA / "representations" / "poi_text_index.faiss"
FAISS_META  = DATA / "representations" / "poi_text_meta.jsonl"

LOG_DIR = BASE / "analysis_outputs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "distance_reranking_sweep.log"

# ============================================================
# CONFIG
# ============================================================

RAG_K_VALUES = [50, 100, 200, 300,400,500,600]
TOP_K_EVAL   = [5, 10, 15, 20, 25, 30]

TOTAL_CASES = 100

SEED = 2027

# ============================================================
# TEE
# ============================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

# ============================================================
# HAVERSINE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# ============================================================
# HELPERS
# ============================================================

def load_poi_data(path):
    coords = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("status") == "ok":
                coords[int(obj["business_id"])] = (
                    obj["latitude"],
                    obj["longitude"]
                )
    return coords

def load_meta(path):
    bids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            bids.append(int(json.loads(line)["business_id"]))
    return bids

def load_jsonl_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_emb_map(path):
    df_emb = pd.read_csv(path)
    return {
        str(r["user_id"]): np.array(json.loads(r["embedding"]), dtype=np.float32)
        for _, r in df_emb.iterrows()
    }

def safe_median(x):
    return float(np.median(x)) if len(x) else np.nan

# ============================================================
# MAIN
# ============================================================

with open(LOG_PATH, "w", encoding="utf-8") as log_fh:
    sys.stdout = Tee(sys.__stdout__, log_fh)

    print(f"LOG FILE: {LOG_PATH}\n")

    print("Loading data...")
    poi_coords = load_poi_data(POI_STATES)
    meta_bids  = load_meta(FAISS_META)
    index      = faiss.read_index(str(FAISS_INDEX))

    all_results = []
    all_medians = []
    topk_records = []

    for seed_cfg in SEEDS:

        seed_name = seed_cfg["name"]
        rows = load_jsonl_rows(seed_cfg["hyp"])
        df = pd.DataFrame(rows)
        emb_map = load_emb_map(seed_cfg["emb"])

        for K in RAG_K_VALUES:

            print(f"\n--- {seed_name}: RAG_K={K} ---")

            counts_base = {k: 0 for k in TOP_K_EVAL}
            counts_dist = {k: 0 for k in TOP_K_EVAL}

            eligible = 0
            base_ranks = []

            for _, r in df.iterrows():

                uid = str(r["user_id"])
                target = int(r["target"])

                if uid not in emb_map:
                    continue

                sequence = r.get("sequence")

                # fallback: manchmal heißt es "input"
                if sequence is None:
                    sequence = r.get("input")

                if not sequence:
                    continue

                last_poi = sequence[-1]
                if last_poi not in poi_coords:
                    continue

                q_emb = emb_map[uid]
                _, I = index.search(q_emb.reshape(1, -1), K)
                candidates = [meta_bids[i] for i in I[0]]

                if target not in candidates:
                    continue

                # filter candidates with coords
                valid_candidates = []
                for bid in candidates:
                    if bid in poi_coords:
                        valid_candidates.append(bid)

                if target not in valid_candidates:
                    continue

                eligible += 1

                # ---------------- BASELINE ----------------
                base_rank = valid_candidates.index(target) + 1
                base_ranks.append(base_rank)

                # ---------------- DISTANCE RERANK ----------------
                lat1, lon1 = poi_coords[last_poi]

                dist_list = []
                for bid in valid_candidates:
                    lat2, lon2 = poi_coords[bid]
                    d = haversine(lat1, lon1, lat2, lon2)
                    dist_list.append((bid, d))

                ranked = sorted(dist_list, key=lambda x: x[1])

                dist_rank = next(i+1 for i,(bid,_) in enumerate(ranked) if bid == target)

                # ---------------- STORE TOP-K ----------------
                topk_records.append({
                    "seed": seed_name,
                    "RAG_K": K,
                    "user_id": uid,
                    "target": target,
                    "last_poi": last_poi,
                    "base_top20": valid_candidates[:20],
                    "dist_top20": [bid for bid,_ in ranked[:20]],
                    "base_rank": base_rank,
                    "dist_rank": dist_rank
                })

                # hits
                for tk in TOP_K_EVAL:
                    if base_rank <= tk:
                        counts_base[tk] += 1
                    if dist_rank <= tk:
                        counts_dist[tk] += 1

            all_medians.append({
                "seed": seed_name,
                "RAG_K": K,
                "eligible_cases": eligible,
                "median_rank_base": safe_median(base_ranks),
            })

            for tk in TOP_K_EVAL:
                all_results.append({
                    "seed": seed_name,
                    "RAG_K": K,
                    "TopK": tk,
                    "base_hits": counts_base[tk],
                    "dist_hits": counts_dist[tk],
                    "eligible_cases": eligible
                })

    # ============================================================
    # AGGREGATION
    # ============================================================

    res_df = pd.DataFrame(all_results)

    print("\n=== RESULTS ===\n")
    print(res_df.to_string(index=False))

    agg_hits = (
        res_df
        .groupby(["RAG_K","TopK"])[["base_hits","dist_hits"]]
        .mean()
        .reset_index()
        .sort_values(["TopK","RAG_K"])
    )

    print("\n=== MEAN RESULTS ===\n")
    print(agg_hits.to_string(index=False))



    print("\n=== BEST K (DISTANCE) ===\n")

    # 🔹 Coverage vorberechnen (mean über seeds)
    cov_df = (
        res_df
        .groupby(["seed", "RAG_K"])["eligible_cases"]
        .mean()
        .reset_index()
        .groupby("RAG_K")["eligible_cases"]
        .mean()
        .reset_index()
    )

    cov_df["coverage_percent"] = 100 * cov_df["eligible_cases"] / TOTAL_CASES

    for topk in TOP_K_EVAL:
        sub = agg_hits[agg_hits["TopK"] == topk]
        best = sub.loc[sub["dist_hits"].idxmax()]

        best_k = int(best["RAG_K"])

        # passende coverage holen
        cov_row = cov_df[cov_df["RAG_K"] == best_k].iloc[0]
        coverage = cov_row["coverage_percent"]

        print(f"TopK={topk}")
        print(f"  Best RAG_K={best_k}")
        print(f"  Hits={best['dist_hits']:.3f}")
        print(f"  Coverage={coverage:.2f}%\n")

    # ============================================================
    # SAVE TOP-K RESULTS
    # ============================================================
    
    topk_df = pd.DataFrame(topk_records)
    topk_df.to_csv(LOG_DIR / "rag_{SEED_TAG}_without_cluster_distance_top20.csv", index=False)
    
    print("\nTop-20 predictions saved.")

    print("\nDONE.")
    sys.stdout = sys.__stdout__
