import pandas as pd
import numpy as np
import faiss
import json
from pathlib import Path
from itertools import combinations
from scipy.stats import ttest_ind

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

DATASET_DIR = Path("/Users/adrianhoffmann/Documents/Master/Projektarbeit/AgenticGroundedRagClone/Dataset")

CSV_PATH = Path("/Users/adrianhoffmann/Documents/Master/Projektarbeit/AgenticGroundedRagClone/results/final_llm_ranking_test_eval_1000.csv")

BASELINE_PATH = Path("/Users/adrianhoffmann/Documents/Master/Projektarbeit/AgenticGroundedRagClone/baseline/baseline_llm_zero_shot_predictions.jsonl")

KNN_PATH = Path("/Users/adrianhoffmann/Documents/Master/Projektarbeit/AgenticGroundedRagClone/baseline/knn_predictions.jsonl")

DISTANCE_PATH = Path("/Users/adrianhoffmann/Documents/Master/Projektarbeit/AgenticGroundedRagClone/baseline/distance_baseline_top20.csv")

FAISS_INDEX = DATASET_DIR / "representations" / "poi_text_index.faiss"
FAISS_META  = DATASET_DIR / "representations" / "poi_text_meta.jsonl"


# ---------------------------------------------------
# LOAD EMBEDDINGS
# ---------------------------------------------------

index = faiss.read_index(str(FAISS_INDEX))
embeddings = index.reconstruct_n(0, index.ntotal)

# Cosine-Normalisierung
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

business_to_vec = {}

with open(FAISS_META) as f:
    for i, line in enumerate(f):
        meta = json.loads(line)
        bid = int(meta["business_id"])
        business_to_vec[bid] = embeddings[i]

print("Loaded embeddings:", len(business_to_vec))


# ---------------------------------------------------
# DISTANCE FUNCTIONS
# ---------------------------------------------------

def cosine_distance(v1, v2):
    return 1 - float(np.dot(v1, v2))


def compute_metrics(business_list):

    vecs = []

    for b in business_list:
        if b in business_to_vec:
            vecs.append(business_to_vec[b])

    if len(vecs) < 2:
        return None

    # Pairwise distances
    distances = []

    for v1, v2 in combinations(vecs, 2):
        distances.append(cosine_distance(v1, v2))

    distances = np.array(distances, dtype=float)

    # Centroid
    centroid = np.mean(vecs, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    centroid_dist = []

    for v in vecs:
        centroid_dist.append(cosine_distance(v, centroid))

    centroid_dist = np.array(centroid_dist, dtype=float)

    return {
        "mean": float(np.mean(distances)),
        "max": float(np.max(distances)),
        "min": float(np.min(distances)),
        "std": float(np.std(distances)),
        "centroid": float(np.mean(centroid_dist))
    }


def aggregate_results(results):

    if len(results) == 0:
        return None

    return {
        "mean": float(np.mean([r["mean"] for r in results])),
        "max": float(np.mean([r["max"] for r in results])),
        "min": float(np.mean([r["min"] for r in results])),
        "std": float(np.mean([r["std"] for r in results])),
        "centroid": float(np.mean([r["centroid"] for r in results]))
    }


# ---------------------------------------------------
# 1) EIGENE METHODE
# ---------------------------------------------------

df = pd.read_csv(CSV_PATH)
df = df[df["rank"] <= 5]

method_results = []

for user_id, group in df.groupby("user_id"):
    businesses = group.sort_values("rank")["business_id"].tolist()
    res = compute_metrics(businesses)
    if res is not None:
        method_results.append(res)

print("\n===== Eigene Methode =====")
print("Users used:", len(method_results))
print(aggregate_results(method_results))


# ---------------------------------------------------
# 2) LLM BASELINE
# ---------------------------------------------------

baseline_results = []

with open(BASELINE_PATH) as f:
    for line in f:
        row = json.loads(line)
        preds = row["prediction"][:5]

        valid = []
        for p in preds:
            try:
                pid = int(p)
                if pid in business_to_vec:
                    valid.append(pid)
            except:
                continue

        res = compute_metrics(valid)
        if res is not None:
            baseline_results.append(res)

print("\n===== LLM Baseline =====")
print("Users used:", len(baseline_results))
print(aggregate_results(baseline_results))


# ---------------------------------------------------
# 3) KNN BASELINE
# ---------------------------------------------------

knn_results = []

with open(KNN_PATH) as f:
    for line in f:
        row = json.loads(line)
        preds = row["prediction"][:5]

        valid = []
        for p in preds:
            try:
                pid = int(p)
                if pid in business_to_vec:
                    valid.append(pid)
            except:
                continue

        res = compute_metrics(valid)
        if res is not None:
            knn_results.append(res)

print("\n===== kNN Baseline =====")
print("Users used:", len(knn_results))
print(aggregate_results(knn_results))


# ---------------------------------------------------
# 4) DISTANCE BASELINE (NEU)
# ---------------------------------------------------

distance_results = []

df_dist = pd.read_csv(DISTANCE_PATH)
df_dist = df_dist[df_dist["rank"] <= 5]

for user_id, group in df_dist.groupby("user_id"):
    businesses = group.sort_values("rank")["business_id"].tolist()
    res = compute_metrics(businesses)
    if res is not None:
        distance_results.append(res)

print("\n===== Distance Baseline =====")
print("Users used:", len(distance_results))
print(aggregate_results(distance_results))


# ---------------------------------------------------
# WELCH T-TESTS
# ---------------------------------------------------

method_means = [r["mean"] for r in method_results]
baseline_means = [r["mean"] for r in baseline_results]
knn_means = [r["mean"] for r in knn_results]
distance_means = [r["mean"] for r in distance_results]

print("\n===== Welch t-tests =====")

def test(a, b, name):
    t, p = ttest_ind(a, b, equal_var=False)
    print(f"\n{name}")
    print("t-statistic:", t)
    print("p-value:", p)

test(method_means, baseline_means, "Ours vs LLM")
test(method_means, knn_means, "Ours vs kNN")
test(method_means, distance_means, "Ours vs Distance")
test(knn_means, distance_means, "kNN vs Distance")
