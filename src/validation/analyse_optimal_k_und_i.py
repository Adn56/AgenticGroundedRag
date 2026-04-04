#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FULL PIPELINE (MULTI-SEED) + BASELINE (NO RE-RANKING) + MEDIAN RANK + CONSOLE LOG FILE

1) For each seed:
   - RAG sweep (FAISS retrieval)
   - Baseline (no reranking): rank of target in FAISS candidate order (after same doc-filtering)
   - Distance re-ranking
   - hit computation for multiple TopK thresholds
   - median baseline rank per RAG_K
2) Aggregate across seeds:
   - mean hits per (RAG_K, TopK) for baseline / Distance / TF-IDF
   - mean coverage per RAG_K
   - median-of-medians for baseline rank per RAG_K (also mean median printed)
3) Global maximum selection (on mean Distance hits)

Additionally:
- Saves the ENTIRE console output to a log file (while still printing to console).
"""

import json
import sys
import numpy as np
import pandas as pd
import faiss

from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PATHS
# ============================================================

BASE = Path("/Users/adrianhoffmann/Documents/Master/Projektarbeit/AgenticGroundedRagClone")
DATA = BASE / "Dataset"

# ---- configure 2 or 3 seeds here ----
SEEDS = [
    {
        "name": "seed2026",
        "hyp": DATA / "validation" / "hypotheses_seed2026_without_cluster.jsonl",
        "emb": DATA / "validation" / "hypotheses_seed2026_without_cluster_embeddings.csv",
    },
    {
        "name": "seed2027",
        "hyp": DATA / "validation" / "hypotheses_seed2027_without_cluster.jsonl",
        "emb": DATA / "validation" / "hypotheses_seed2027_clusterOnly_embeddings.csv",
    }
]

POI_STATES  = DATA / "core" / "poi_semantic_states.jsonl"
FAISS_INDEX = DATA / "representations" / "poi_text_index.faiss"
FAISS_META  = DATA / "representations" / "poi_text_meta.jsonl"

# Log file for full console output
LOG_DIR = BASE / "analysis_outputs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "reranking_sweep_full_console_clusterOnly.log"

# ============================================================
# CONFIG
# ============================================================

RAG_K_VALUES = [50, 100, 200, 300, 400, 500, 600, 700]

TOP_K_EVAL   = [5, 10, 15, 20, 25, 30]

TOTAL_CASES = 100  # used for coverage_percent (= eligible_cases / TOTAL_CASES)

# ============================================================
# TEE (console + file)
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
# HELPERS
# ============================================================

def load_poi_states(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("status") == "ok":
                out[int(obj["business_id"])] = obj["attributes"]
    return out

def load_meta(path):
    bids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            bids.append(int(json.loads(line)["business_id"]))
    return bids

def build_doc(attr):
    tokens = []
    for k in [
        "spatial_configuration",
        "style",
        "atmosphere",
        "target_group",
        "distinctive_features",
    ]:
        tokens.extend(attr.get(k, []))
    return [t.lower() for t in tokens if isinstance(t, str)]

def tokens_to_text(tokens):
    return " ".join(tokens)

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
# MAIN (redirect console to file + console)
# ============================================================

with open(LOG_PATH, "w", encoding="utf-8") as log_fh:
    sys.stdout = Tee(sys.__stdout__, log_fh)

    print(f"LOG FILE: {LOG_PATH}\n")

    # ============================================================
    # LOAD STATIC ARTIFACTS ONCE
    # ============================================================

    print("Loading static data (POI states, FAISS index, meta)...")
    poi_states = load_poi_states(POI_STATES)
    meta_bids  = load_meta(FAISS_META)
    index      = faiss.read_index(str(FAISS_INDEX))

    # ============================================================
    # RUN PER SEED
    # ============================================================

    all_results = []
    # will store per-seed, per-RAG_K median rank for baseline
    all_medians = []  # rows: seed, RAG_K, median_rank_base

    for seed_cfg in SEEDS:

        seed_name = seed_cfg["name"]
        hyp_path  = seed_cfg["hyp"]
        emb_path  = seed_cfg["emb"]

        print("\n\n==============================")
        print(f"RUNNING: {seed_name}")
        print(f"HYP: {hyp_path}")
        print(f"EMB: {emb_path}")
        print("==============================\n")

        rows = load_jsonl_rows(hyp_path)
        df = pd.DataFrame(rows)
        emb_map = load_emb_map(emb_path)

        for K in RAG_K_VALUES:

            print(f"--- {seed_name}: RAG cutoff K={K} ---")

            counts_base  = {k: 0 for k in TOP_K_EVAL}
            counts_bm25  = {k: 0 for k in TOP_K_EVAL}
            counts_tfidf = {k: 0 for k in TOP_K_EVAL}

            eligible = 0
            base_ranks = []  # collect baseline ranks to compute median

            for _, r in df.iterrows():

                uid = str(r["user_id"])
                target = int(r["target"])

                if uid not in emb_map:
                    continue

                q_emb = emb_map[uid]
                _, I = index.search(q_emb.reshape(1, -1), K)
                candidates = [meta_bids[i] for i in I[0]]

                # eligible only if target is inside FAISS Top-K
                if target not in candidates:
                    continue

                # Build corpus with the SAME filtering logic for baseline + rerankers
                # (so baseline ranks are comparable to reranking ranks)
                corpus_tokens, bids = [], []
                for bid in candidates:
                    attr = poi_states.get(bid)
                    if not attr:
                        continue
                    toks = build_doc(attr)
                    if toks:
                        corpus_tokens.append(toks)
                        bids.append(bid)

                if target not in bids:
                    continue

                hyp_tokens = build_doc(r["hypothesis"])
                if not hyp_tokens:
                    continue

                eligible += 1

                # ---------------- BASELINE (no reranking) ----------------
                # rank in FAISS order AFTER same doc-filtering (bids keeps candidate order)
                base_rank = bids.index(target) + 1
                base_ranks.append(base_rank)

                # ---------------- BM25 ----------------
                bm25 = BM25Okapi(corpus_tokens)
                bm25_scores = bm25.get_scores(hyp_tokens)

                bm25_ranked = sorted(
                    zip(bids, bm25_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                bm25_rank = next(i + 1 for i, (bid, _) in enumerate(bm25_ranked) if bid == target)

                # ---------------- TF-IDF ----------------
                corpus_text = [tokens_to_text(t) for t in corpus_tokens]
                hyp_text = tokens_to_text(hyp_tokens)

                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(corpus_text + [hyp_text])
                sims = cosine_similarity(X[-1], X[:-1])[0]

                tfidf_ranked = sorted(
                    zip(bids, sims),
                    key=lambda x: x[1],
                    reverse=True
                )
                tfidf_rank = next(i + 1 for i, (bid, _) in enumerate(tfidf_ranked) if bid == target)

                # hits for thresholds
                for tk in TOP_K_EVAL:
                    if base_rank <= tk:
                        counts_base[tk] += 1
                    if bm25_rank <= tk:
                        counts_bm25[tk] += 1
                    if tfidf_rank <= tk:
                        counts_tfidf[tk] += 1

            # store median baseline rank per seed, per K
            all_medians.append({
                "seed": seed_name,
                "RAG_K": K,
                "eligible_cases": eligible,
                "median_rank_base": safe_median(base_ranks),
            })

            # store per TopK rows (absolute counts)
            for tk in TOP_K_EVAL:
                all_results.append({
                    "seed": seed_name,
                    "RAG_K": K,
                    "TopK": tk,

                    "base_hits": counts_base[tk],
                    "bm25_hits": counts_bm25[tk],
                    "tfidf_hits": counts_tfidf[tk],

                    "eligible_cases": eligible
                })

    # ============================================================
    # AGGREGATION ACROSS SEEDS
    # ============================================================

    res_df = pd.DataFrame(all_results)
    med_df = pd.DataFrame(all_medians)

    print("\n\n========================================")
    print("=== ABSOLUTE HIT COUNTS (PER SEED) ===")
    print("========================================\n")
    print(res_df.to_string(index=False))

    # ---- Mean hits across seeds
    agg_hits = (
        res_df
        .groupby(["RAG_K", "TopK"])[["base_hits", "bm25_hits", "tfidf_hits"]]
        .mean()
        .reset_index()
        .sort_values(["TopK", "RAG_K"])
    )

    print("\n\n========================================")
    print("=== MEAN HITS (AGGREGATED OVER SEEDS) ===")
    print("========================================\n")
    print(agg_hits.to_string(index=False))

    # ---- Coverage: mean eligible_cases over seeds (per RAG_K)
    agg_cov = (
        res_df
        .groupby(["seed", "RAG_K"])["eligible_cases"]
        .mean()
        .reset_index()
        .groupby("RAG_K")["eligible_cases"]
        .mean()
        .reset_index()
        .sort_values("RAG_K")
    )
    agg_cov["coverage_percent"] = 100 * agg_cov["eligible_cases"] / TOTAL_CASES

    print("\n\n========================================")
    print("=== RETRIEVAL COVERAGE (MEAN OVER SEEDS) ===")
    print("========================================\n")
    print(agg_cov.to_string(index=False))

    # ---- Baseline median rank reporting (per seed and aggregated)
    print("\n\n========================================")
    print("=== BASELINE (NO RERANKING): MEDIAN RANK (PER SEED) ===")
    print("========================================\n")
    print(med_df.sort_values(["seed", "RAG_K"]).to_string(index=False))

    agg_med = (
        med_df
        .groupby("RAG_K")["median_rank_base"]
        .agg(mean_median="mean", median_of_medians="median")
        .reset_index()
        .sort_values("RAG_K")
    )

    print("\n\n========================================")
    print("=== BASELINE (NO RERANKING): MEDIAN RANK (AGGREGATED) ===")
    print("========================================\n")
    print(agg_med.to_string(index=False))

    # ============================================================
    # GLOBAL MAXIMUM SELECTION (ON MEAN BM25 HITS)
    # ============================================================

    print("\n\n========================================")
    print("=== GLOBAL MAXIMUM SELECTION (Distance, MEAN OVER SEEDS) ===")
    print("========================================\n")

    for topk in TOP_K_EVAL:
        sub = agg_hits[agg_hits["TopK"] == topk]
        max_mean = sub["bm25_hits"].max()
        best_rows = sub[sub["bm25_hits"] == max_mean]
        best_k = best_rows["RAG_K"].min()

        cov_row = agg_cov[agg_cov["RAG_K"] == best_k].iloc[0]

        print(f"TopK={topk}")
        print(f"  Global Max Mean Hits (BM25) = {max_mean:.3f}")
        print(f"  Smallest RAG_K achieving this = {best_k}")
        print(f"  Coverage at this K (mean over seeds) = {cov_row['coverage_percent']:.2f}%")
        print()

    print("\nDONE.")
    print(f"Full console output saved to: {LOG_PATH}")

# restore stdout (optional, script end anyway)
sys.stdout = sys.__stdout__
