#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
from collections import Counter
import faiss
from pathlib import Path

# ============================================================
# PATHS
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "Dataset"
BASELINE_PATH = PROJECT_ROOT / "baseline"
RESULTS_PATH = PROJECT_ROOT / "results"

SEQ_PATH = DATASET_DIR / "splits" / f"test_eval_1000.jsonl"



TRAIN_PATH =  DATASET_DIR / "splits"/"train.jsonl"
TEST_PATH  = DATASET_DIR / "splits"/"test_eval_1000.jsonl"

LLM_PATH = BASELINE_PATH / "baseline_llm_zero_shot_predictions.jsonl"
KNN_PATH = BASELINE_PATH / "knn_predictions.jsonl"
DISTANCE_PATH = BASELINE_PATH / "distance_baseline_top20.csv"
OWN_PATH = RESULTS_PATH / "final_llm_ranking_test_eval_1000.csv"


FAISS_INDEX_PATH = DATASET_DIR / "representations" / "poi_text_index.faiss"
META_PATH = DATASET_DIR / "representations" / "poi_text_meta.jsonl"

K = 5
CONTEXT_LENGTH = 9


# ============================================================
# EMBEDDINGS
# ============================================================

def load_embeddings():
    index = faiss.read_index(str(FAISS_INDEX_PATH))  # <-- FIX
    poi_to_row = {}

    with open(META_PATH, "r") as f:
        for line in f:
            row = json.loads(line)
            poi_to_row[str(row["business_id"])] = int(row["faiss_row"])

    return index, poi_to_row


def get_embedding(index, poi_to_row, poi_id):
    row = poi_to_row.get(str(poi_id))
    if row is None:
        return None
    return index.reconstruct(row)


def cosine(a, b):
    if a is None or b is None:
        return None

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


# ============================================================
# LOAD TRAIN
# ============================================================

def load_train(train_path):
    popularity = Counter()

    with open(train_path, "r") as f:
        for line in f:
            row = json.loads(line)
            for visit in row["trajectory"]:
                poi = str(visit["business_id"])
                popularity[poi] += 1

    total = sum(popularity.values())
    if total == 0:
        return {}

    for poi in popularity:
        popularity[poi] /= total

    return dict(popularity)


# ============================================================
# LOAD TEST
# ============================================================

def load_test(test_path):
    targets = {}
    contexts = {}

    with open(test_path, "r") as f:
        for line in f:
            row = json.loads(line)

            user_id = str(row["user_id"])
            input_seq = [str(v["business_id"]) for v in row["input"]]

            contexts[user_id] = input_seq[-CONTEXT_LENGTH:]
            targets[user_id] = str(row["target"]["business_id"])

    return targets, contexts


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_csv_predictions(path):
    df = pd.read_csv(path)
    result = {}

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("rank")
        result[str(user_id)] = group["business_id"].astype(str).tolist()

    return result


def load_jsonl_predictions(path):
    preds = {}

    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            preds[str(row["user_id"])] = [str(p) for p in row["prediction"]]

    return preds


# ============================================================
# HELPERS
# ============================================================

def topk_recs(r):
    return r[:K]


def is_valid_embedding_poi(p, all_pois):
    return p in all_pois


def is_valid_training_poi(p, popularity):
    return p in popularity


def is_valid_joint_poi(p, all_pois, popularity):
    return (p in all_pois) and (p in popularity)


def safe_mean(vals):
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


# ============================================================
# BASIC DIAGNOSTICS
# ============================================================

def avg_list_length(preds):
    return safe_mean([min(len(r), K) for r in preds.values()])


def validity_rate_embedding_space(preds, all_pois):
    total = 0
    valid = 0

    for r in preds.values():
        for p in topk_recs(r):
            total += 1
            if is_valid_embedding_poi(p, all_pois):
                valid += 1

    return (valid / total if total > 0 else 0.0), valid, total


def validity_rate_training_space(preds, popularity):
    total = 0
    valid = 0

    for r in preds.values():
        for p in topk_recs(r):
            total += 1
            if is_valid_training_poi(p, popularity):
                valid += 1

    return (valid / total if total > 0 else 0.0), valid, total


def unique_ratio(preds):
    vals = []

    for r in preds.values():
        recs = topk_recs(r)
        if len(recs) > 0:
            vals.append(len(set(recs)) / len(recs))

    return safe_mean(vals)


# ============================================================
# CLASSIC METRICS
# ============================================================

def coverage_at_k(preds, all_pois):
    rec = set()

    for r in preds.values():
        for p in topk_recs(r):
            if is_valid_embedding_poi(p, all_pois):
                rec.add(p)

    return len(rec) / len(all_pois) if len(all_pois) > 0 else 0.0, len(rec), len(all_pois)


def novelty_at_k(preds, popularity, all_pois):
    """
    Novelty only over POIs that are valid both
    in the embedding space and in the training popularity space.
    This avoids artificially inflated novelty from OOD predictions.
    """
    vals = []
    valid_count = 0
    total_count = 0

    for r in preds.values():
        for p in topk_recs(r):
            total_count += 1

            if not is_valid_joint_poi(p, all_pois, popularity):
                continue

            pop = popularity[p]
            if pop <= 0:
                continue

            vals.append(-np.log(pop))
            valid_count += 1

    return safe_mean(vals), valid_count, total_count


def avg_popularity_at_k(preds, popularity, all_pois):
    """
    Average popularity only over valid POIs in both spaces.
    """
    vals = []
    valid_count = 0
    total_count = 0

    for r in preds.values():
        for p in topk_recs(r):
            total_count += 1

            if not is_valid_joint_poi(p, all_pois, popularity):
                continue

            vals.append(popularity[p])
            valid_count += 1

    return safe_mean(vals), valid_count, total_count


# ============================================================
# EMBEDDING METRICS
# ============================================================

def semantic_similarity(preds, targets, index, poi_to_row, all_pois):
    """
    Mean cosine similarity between predicted POIs and the true target POI,
    computed only for prediction-target pairs where both embeddings exist.
    """
    vals = []
    valid_pairs = 0
    total_pairs = 0

    for u, r in preds.items():
        t = targets.get(u)
        if t is None or t not in all_pois:
            continue

        e_t = get_embedding(index, poi_to_row, t)
        if e_t is None:
            continue

        for p in topk_recs(r):
            total_pairs += 1

            if p not in all_pois:
                continue

            e_p = get_embedding(index, poi_to_row, p)
            if e_p is None:
                continue

            sim = cosine(e_p, e_t)
            if sim is None:
                continue

            vals.append(sim)
            valid_pairs += 1

    return safe_mean(vals), valid_pairs, total_pairs


def diversity_embedding(preds, index, poi_to_row, all_pois):
    """
    Intra-list embedding diversity:
    1 - mean pairwise cosine similarity
    computed only on valid embedded POIs within each list.
    """
    vals = []
    valid_lists = 0
    total_lists = 0

    for r in preds.values():
        total_lists += 1

        recs = [p for p in topk_recs(r) if p in all_pois]
        embs = []

        for p in recs:
            e = get_embedding(index, poi_to_row, p)
            if e is not None:
                embs.append(e)

        if len(embs) < 2:
            continue

        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sim = cosine(embs[i], embs[j])
                if sim is not None:
                    sims.append(sim)

        if len(sims) == 0:
            continue

        vals.append(1.0 - np.mean(sims))
        valid_lists += 1

    return safe_mean(vals), valid_lists, total_lists


# ============================================================
# GINI
# ============================================================

def gini_index_at_k(preds, all_pois):
    counts = Counter()

    for r in preds.values():
        for p in topk_recs(r):
            if is_valid_embedding_poi(p, all_pois):
                counts[p] += 1

    x = np.array([counts.get(p, 0) for p in all_pois], dtype=np.float64)

    if x.sum() == 0:
        return 0.0

    x_sorted = np.sort(x)
    n = len(x_sorted)
    i = np.arange(1, n + 1)

    gini = (2.0 * np.sum(i * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1.0) / n
    return float(max(0.0, min(1.0, gini)))


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, preds, targets, popularity, all_pois, index, poi_to_row):
    print("\n==============================")
    print(name)
    print("==============================")

    avg_len = avg_list_length(preds)
    uniq = unique_ratio(preds)

    vr_embed, valid_embed, total_embed = validity_rate_embedding_space(preds, all_pois)
    vr_train, valid_train, total_train = validity_rate_training_space(preds, popularity)

    cov, cov_num, cov_den = coverage_at_k(preds, all_pois)
    gini = gini_index_at_k(preds, all_pois)

    novelty, novelty_valid, novelty_total = novelty_at_k(preds, popularity, all_pois)
    avg_pop, avg_pop_valid, avg_pop_total = avg_popularity_at_k(preds, popularity, all_pois)

    sem_sim, sem_valid, sem_total = semantic_similarity(preds, targets, index, poi_to_row, all_pois)
    div_emb, div_valid, div_total = diversity_embedding(preds, index, poi_to_row, all_pois)

    print(f"AvgListLength@{K}:                 {avg_len:.4f}")
    print(f"UniqueRatio@{K}:                  {uniq:.4f}")

    print(f"ValidityRate@{K} (embedding):     {vr_embed:.4f}   [{valid_embed}/{total_embed}]")
    print(f"ValidityRate@{K} (training):      {vr_train:.4f}   [{valid_train}/{total_train}]")

    print(f"Coverage@{K}:                     {cov:.4f}   [{cov_num}/{cov_den}]")
    print(f"GiniIndex@{K}:                    {gini:.4f}")

    print(f"Novelty@{K}:                      {novelty:.4f}   [{novelty_valid}/{novelty_total}]")
    print(f"AvgPopularity@{K}:                {avg_pop:.6f}   [{avg_pop_valid}/{avg_pop_total}]")

    print(f"SemanticSimilarity@{K}:           {sem_sim:.4f}   [{sem_valid}/{sem_total}]")
    print(f"Diversity@{K} (embedding):        {div_emb:.4f}   [{div_valid}/{div_total} lists]")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading TRAIN...")
    popularity = load_train(TRAIN_PATH)

    print("Loading TEST...")
    targets, _ = load_test(TEST_PATH)

    print("Loading predictions...")
    preds_llm = load_jsonl_predictions(LLM_PATH)
    preds_knn = load_jsonl_predictions(KNN_PATH)
    preds_own = load_csv_predictions(OWN_PATH)
    preds_distance = load_csv_predictions(DISTANCE_PATH)

    print("Loading embeddings...")
    index, poi_to_row = load_embeddings()
    all_pois = set(poi_to_row.keys())

    evaluate("LLM BASELINE", preds_llm, targets, popularity, all_pois, index, poi_to_row)
    evaluate("KNN BASELINE", preds_knn, targets, popularity, all_pois, index, poi_to_row)
    evaluate("OWN MODEL", preds_own, targets, popularity, all_pois, index, poi_to_row)
    evaluate("DISTANCE BASELINE", preds_distance, targets, popularity, all_pois, index, poi_to_row)


if __name__ == "__main__":
    main()