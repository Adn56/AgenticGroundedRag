#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM BEHAVIOURAL SCORING PIPELINE
Fully integrated prompt generator + deterministic inference.

Parallel version (6 concurrent requests)

Features:
- Temperature = 0
- Output saved as JSONL
- Crash-safe (flush after every write)
- Resume capability
"""

import json
import re
import math
import time
from pathlib import Path
from statistics import mean
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gpt-4.1-mini"
TEMPERATURE = 0.0
MAX_WORKERS = 6

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset"

SEQ_PATH = DATASET_DIR / "splits" / f"test_eval_1000.jsonl"
TOP20_PATH = PROJECT_ROOT / "results" / "distance_reranked_top20.csv"
POI_PATH = DATASET_DIR / "core" / "poi_semantic_states.jsonl"

PROMPT_LOG_PATH = PROJECT_ROOT / "results" / f"prompts_scoring_test_eval_1000.jsonl"
OUTPUT_LOG_PATH = PROJECT_ROOT / "results" / f"scoring_test_eval_1000.jsonl"

client = OpenAI()

# Thread-safe file writing
file_lock = threading.Lock()

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a behavioural plausibility estimator.

Task:
For each candidate point of interest, assign three independent continuous scores in the closed interval [0,1].

Definitions:

s_dist:
Spatial plausibility of the candidate displacement relative to the empirically observed movement pattern in the sequence.
Evaluation is grounded in prior inter-POI displacements.
The AVERAGE_MOVEMENT_RADIUS serves as a descriptive reference and does not constitute a decision threshold.

s_time:
Temporal plausibility given the provided weekday and time-of-day.
Evaluation is grounded in the observed temporal structure of the sequence and interpreted in relation to the candidate’s stated functional characteristics.

s_sem:
Semantic continuity between the candidate and the preceding sequence.
Evaluation is grounded in structured attribute alignment, recurrence patterns, and coherent contextual interpretation.

Scale interpretation:
Scores represent graded compatibility with the relevant behavioural pattern.
0.0 denotes complete incompatibility.
1.0 denotes maximal compatibility.
Intermediate values reflect proportional alignment.

Constraints:
Each candidate must be evaluated independently.
Scores must not be comparatively normalised across candidates.
No ranking or aggregation is permitted.
No explanatory text may be produced.
Output must be strictly valid JSON.

Additionally, estimate a retrieval_sufficiency score in the closed interval [0,1].

Definition:
retrieval_sufficiency reflects how confident you are that the true next location
is represented among the listed candidates.

Low values indicate likely candidate set insufficiency.
High values indicate strong confidence that the correct target is present.
"""

# ============================================================
# HELPERS
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def format_attributes(attr_dict):
    lines = []
    for key, values in attr_dict.items():
        if key == "summary":
            continue
        if values and isinstance(values, list):
            filtered = [v for v in values if v]
            if filtered:
                lines.append(f"{key}: {', '.join(filtered)}")
    return "\n".join(lines)


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


# ============================================================
# RESUME LOGIC
# ============================================================

processed_users = set()

if OUTPUT_LOG_PATH.exists():
    with open(OUTPUT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            processed_users.add(obj["user_id"])

print(f"Resume mode active. Already processed: {len(processed_users)} users.")


# ============================================================
# LOAD DATA
# ============================================================

splits = load_jsonl(SEQ_PATH)
df_top20 = pd.read_csv(TOP20_PATH)

poi_meta = {}
for obj in load_jsonl(POI_PATH):
    if obj.get("status") == "ok":
        poi_meta[int(obj["business_id"])] = obj


# ============================================================
# WORKER FUNCTION
# ============================================================

def process_split(split):

    user_id = split["user_id"]

    if user_id in processed_users:
        return

    sequence = split["input"]
    target = split["target"]

    sequence_block = ""
    jump_distances = []

    for i, visit in enumerate(sequence):

        bid = visit["business_id"]
        meta = poi_meta.get(bid)
        if not meta:
            continue

        attr_text = format_attributes(meta["attributes"])

        dist_prev_text = ""
        if i > 0:
            prev_bid = sequence[i-1]["business_id"]
            prev_meta = poi_meta.get(prev_bid)

            if prev_meta:
                d = haversine(
                    prev_meta["latitude"], prev_meta["longitude"],
                    meta["latitude"], meta["longitude"]
                )

                jump_distances.append(d)
                dist_prev_text = f"\nDistance_from_previous: {round(d,1)} km"

        sequence_block += f"""POI {i+1}
Name: {meta.get("name")}
Date: {visit["date_ymd"]} ({visit["date_day"]})
Time: {visit["date_time"]}{dist_prev_text}
{attr_text}

"""

    if not jump_distances:
        return

    avg_radius = round(mean(jump_distances), 2)

    last_bid = sequence[-1]["business_id"]
    last_meta = poi_meta.get(last_bid)

    if not last_meta:
        return

    group = df_top20[df_top20["user_id"] == user_id]

    candidates_block = ""

    for idx, row in enumerate(group.itertuples(), start=1):

        bid = int(row.business_id)
        meta = poi_meta.get(bid)

        if not meta:
            continue

        dist = haversine(
            last_meta["latitude"], last_meta["longitude"],
            meta["latitude"], meta["longitude"]
        )

        attr_text = format_attributes(meta["attributes"])

        city_name = meta.get("city_name")

        if city_name and str(city_name).strip():
            city_line = f"City: {city_name}"
        else:
            city_line = "City: Not specified"

        candidates_block += f"""Candidate {idx}
Business_ID: {bid}
Name: {meta.get("name")}
{city_line}
Distance_from_last: {round(dist,2)} km
{attr_text}

"""

    user_prompt = f"""AVERAGE_MOVEMENT_RADIUS: {avg_radius} km

NEXT_VISIT
Date: {target["date_ymd"]} ({target["date_day"]})
Time: {target["date_time"]}

SEQUENCE
{sequence_block}
CANDIDATES
{candidates_block}
Return:

{{
  "candidates": [
    {{
      "business_id": 0,
      "s_dist": 0.00,
      "s_time": 0.00,
      "s_sem": 0.00
    }}
  ],
  "retrieval_sufficiency": 0.00
}}
"""

    timestamp = time.time()

    with file_lock:
        with open(PROMPT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "user_id": user_id,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "timestamp": timestamp
            }) + "\n")
            f.flush()

    response = client.responses.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw_output = response.output_text.strip()

    try:
        cleaned_output = extract_json(raw_output)
        parsed_output = json.loads(cleaned_output)
        retrieval_sufficiency = parsed_output.get("retrieval_sufficiency")

    except Exception:
        parsed_output = {
            "parse_error": True,
            "raw_output": raw_output
        }
        retrieval_sufficiency = None

    with file_lock:
        with open(OUTPUT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "user_id": user_id,
                "model": MODEL_NAME,
                "scores": parsed_output.get("candidates"),
                "retrieval_sufficiency": retrieval_sufficiency,
                "raw_output": raw_output,
                "timestamp": time.time()
            }) + "\n")
            f.flush()

    print(f"Processed user_id {user_id}")


# ============================================================
# PARALLEL EXECUTION
# ============================================================

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = []

    for split in splits:
        futures.append(executor.submit(process_split, split))

    for future in as_completed(futures):
        future.result()

print("Inference complete.")
