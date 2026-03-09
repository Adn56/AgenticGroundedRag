#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BENUDECE BASELINE REPRODUCTION (STRICT REPO VERSION)

Faithful reproduction of the prompt_type == '1' baseline from:
ssai-trento/LLM-zero-shot-NL

Key properties
--------------
- NO system prompt
- prompt identical to repository
- single string prompt input
- parallel API requests
- resumable
- immediate disk writes

Outputs
-------
1. predictions JSONL
2. prompts CSV
3. token usage CSV
"""

import os
import json
import csv
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0
MAX_WORKERS = 6

HIST_LEN = 6
CONTEXT_LEN = 3

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "Dataset"

SEQ_PATH = DATASET_DIR / "splits" / f"test_eval_1000.jsonl"

OUT_DIR = PROJECT_ROOT/ "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRED = OUT_DIR / f"baseline_llm_zero_shot_predictions.jsonl"
OUT_PROMPTS = OUT_DIR / f"baseline_llm_zero_shot_prompts.csv"
OUT_USAGE = OUT_DIR / f"baseline_llm_zero_shot_usage.csv"

client = OpenAI()

# ============================================================
# EXACT REPO PROMPT
# ============================================================

def build_prompt(historical, context, target):

    prompt = f"""
    		Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

                Predict <next_place_id> by considering:
                1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
                2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

                Present your answer in a JSON object with:
                "prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
                
		The data:
                    <historical_stays>: {historical}
                    <context_stays>: {context}
                    <target_stay>: {target}
                   """
    return prompt


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def processed_users(path):

    if not path.exists():
        return set()

    done = set()

    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add(row["user_id"])
            except:
                pass

    return done


def visit_to_triplet(v):

    return [
        v["date_time"],
        v["date_day"],
        str(v["business_id"])
    ]


def split_visits(visits):

    hist = visits[:HIST_LEN]
    ctx = visits[HIST_LEN:HIST_LEN + CONTEXT_LEN]

    return hist, ctx


def build_target(target):

    return [
        target["date_time"],
        target["date_day"],
        "<next_place_id>"
    ]


def extract_json(text):

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    try:
        return json.loads(text[start:end + 1])
    except:
        return None


# ============================================================
# WORKER
# ============================================================

def run_sequence(seq):

    user_id = seq["user_id"]

    visits = seq["input"]
    target = seq["target"]

    hist, ctx = split_visits(visits)

    hist = [visit_to_triplet(v) for v in hist]
    ctx = [visit_to_triplet(v) for v in ctx]

    target_stay = build_target(target)

    prompt = build_prompt(hist, ctx, target_stay)

    t0 = time.time()

    resp = client.responses.create(
        model=LLM_MODEL,
        input=prompt,
        temperature=TEMPERATURE
    )

    latency = round(time.time() - t0, 3)

    text = resp.output_text

    parsed = extract_json(text)

    prediction = None
    reason = None

    if parsed:
        prediction = parsed.get("prediction")
        reason = parsed.get("reason")

    usage = resp.usage

    return {

        "user_id": user_id,
        "prompt": prompt,
        "raw_response": text,
        "parsed": parsed,
        "prediction": prediction,
        "reason": reason,
        "target": target["business_id"],
        "sequence": [v["business_id"] for v in visits],
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "latency": latency
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing")

    data = load_jsonl(SEQ_PATH)

    done = processed_users(OUT_PRED)

    pending = [x for x in data if x["user_id"] not in done]

    print("total:", len(data))
    print("pending:", len(pending))

    prompt_exists = OUT_PROMPTS.exists()
    usage_exists = OUT_USAGE.exists()

    with open(OUT_PRED, "a") as pred_f, \
            open(OUT_PROMPTS, "a", newline="") as prompt_f, \
            open(OUT_USAGE, "a", newline="") as usage_f:

        prompt_writer = csv.writer(prompt_f)

        if not prompt_exists:
            prompt_writer.writerow(["user_id", "prompt"])

        usage_writer = csv.writer(usage_f)

        if not usage_exists:
            usage_writer.writerow(["user_id", "input_tokens", "output_tokens", "total_tokens"])

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:

            futures = [pool.submit(run_sequence, s) for s in pending]

            for f in as_completed(futures):

                try:
                    r = f.result()
                except Exception as e:
                    print("ERROR", e)
                    continue

                pred_f.write(json.dumps({
                    "user_id": r["user_id"],
                    "prediction": r["prediction"],
                    "reason": r["reason"],
                    "target": r["target"],
                    "sequence": r["sequence"]
                }) + "\n")
                pred_f.flush()

                prompt_writer.writerow([r["user_id"], r["prompt"]])
                prompt_f.flush()

                usage_writer.writerow([
                    r["user_id"],
                    r["input_tokens"],
                    r["output_tokens"],
                    r["total_tokens"]
                ])
                usage_f.flush()

                print("done user", r["user_id"])


if __name__ == "__main__":
    main()