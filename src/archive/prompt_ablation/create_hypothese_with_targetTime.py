#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-ONLY SEMANTIC HYPOTHESIS GENERATION (PARALLEL WORKERS)

- 6 parallel workers
- Resumable (skip processed user_ids)
- Writes AFTER EACH RESPONSE
- Stores:
    (1) hypotheses JSONL
    (2) exact prompts CSV
    (3) token usage CSV
"""

import os
import json
import csv
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================


LLM_MODEL  = "gpt-4.1-mini"
TEMPERATURE = 0
NUM_WORKERS = 6

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

SEQ_PATH = DATASET_DIR / "splits" / f"test_eval_1000.jsonl"
SEM_PATH = DATASET_DIR / "core" / "poi_semantic_states.jsonl"

OUT_DIR = PROJECT_ROOT / "prompt_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_HYP_JSONL = OUT_DIR / f"hypothesis_test_eval_1000_withTime.jsonl"
OUT_PROMPT_CSV = OUT_DIR / f"prompts_test_eval_1000_withTime.csv"
OUT_USAGE_CSV = OUT_DIR / f"usage_test_eval_1000_withTime.csv"

# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
Generate a semantically grounded description of a single point of interest
that plausibly continues a given sequence.

For each output field, identify attributes that recur across multiple POIs
and form a clear semantic pattern, and carry those attributes over directly
into the hypothesis.
Ignore isolated, weakly supported, or singular observations.
Do not mix unrelated characteristics across different places.

Describe the place directly as an observed entity.
Do not include reasoning, explanations, or transition descriptions.
"""

USER_PROMPT_TEMPLATE = """
You are given the LAST 9 points of interest visited by a user.
Each POI is described using visually grounded attributes.

POI SEQUENCE (ordered from oldest to most recent):

<<<POI_BLOCKS>>>


The next visit will occur at:

<<<TARGET_TIME>>>

Temporal constraint:
Use this upcoming visit time as an additional behavioral signal.
If recurring weekday or time-of-day patterns are present in the sequence,
the hypothesis must remain temporally plausible.

Decision policy (must be followed):

1) Sequence-first carryover:
   For each output field, if attributes recur across multiple POIs
   and form a clear semantic pattern, carry those attributes over
   directly into the hypothesis.
   Preserve concreteness and do not generalize away dominant details.

2) No mixing:
   Do not combine unrelated features from different POIs.
   The output must describe one coherent, single place.

3) No target leakage:
   Do not assume knowledge about the true next POI or its identity.
   Base all attributes strictly on the provided sequence.

Task:
Construct a semantically consistent description of a single point of interest
that plausibly continues the sequence.

Return the result strictly in the following JSON format:

{
  "spatial_configuration": [],
  "style": [],
  "atmosphere": [],
  "target_group": [],
  "distinctive_features": [],
  "summary": ""
}

Summary constraints:
- Must be a concise synthesis of the fields above.
- Must not introduce new characteristics.
- Must not explicitly mention the target group.
"""

# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_existing_user_ids(path):
    if not path.exists():
        return set()

    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(int(json.loads(line)["user_id"]))
            except:
                pass
    return done

def format_datetime(date_ymd, date_day, date_time):
    return f"{date_ymd}, on {date_day} at {date_time}"

def poi_block(visit, poi):

    visit_string = format_datetime(
        visit["date_ymd"],
        visit["date_day"],
        visit["date_time"]
    )

    return json.dumps({
        "visit_time": visit_string,
        "summary": poi["summary"],
        **poi["attributes"]
    }, ensure_ascii=False, indent=2)

def extract_json(text):

    start = text.find("{")
    end   = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object in LLM output")

    return json.loads(text[start:end+1])

# ============================================================
# GLOBALS
# ============================================================

lock = threading.Lock()

# ============================================================
# WORKER
# ============================================================

def process_sequence(seq):

    user_id = int(seq["user_id"])

    context = seq["input"]
    context_ids = [int(v["business_id"]) for v in context]

    if any(b not in semantic for b in context_ids):
        return

    target = seq["target"]

    target_time_string = format_datetime(
        target["date_ymd"],
        target["date_day"],
        target["date_time"]
    )

    poi_blocks = ""

    for i, v in enumerate(context,1):
        poi_blocks += (
            f"POI {i}:\n"
            f"{poi_block(v, semantic[int(v['business_id'])])}\n\n"
        )

    user_prompt = USER_PROMPT_TEMPLATE \
        .replace("<<<POI_BLOCKS>>>", poi_blocks) \
        .replace("<<<TARGET_TIME>>>", target_time_string)

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_prompt},
        ],
        temperature=TEMPERATURE
    )

    usage = resp.usage

    hypothesis = extract_json(resp.output_text)

    with lock:

        prompt_w.writerow({
            "user_id":user_id,
            "system_prompt":SYSTEM_PROMPT.strip(),
            "user_prompt":user_prompt.strip()
        })
        prompt_f.flush()

        hyp_f.write(json.dumps({
            "user_id":user_id,
            "sequence":context_ids,
            "target":int(seq["target"]["business_id"]),
            "target_seen_in_last_9":seq["target_seen_in_last_9"],
            "prediction_correct":seq.get("prediction_correct"),
            "hypothesis":hypothesis
        },ensure_ascii=False)+"\n")
        hyp_f.flush()

        usage_w.writerow({
            "user_id":user_id,
            "input_tokens":usage.input_tokens,
            "output_tokens":usage.output_tokens,
            "total_tokens":usage.total_tokens
        })
        usage_f.flush()

        print(f"[OK] user_id={user_id}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI()

    sequences = load_jsonl(SEQ_PATH)

    semantic = {
        int(o["business_id"]):o
        for o in load_jsonl(SEM_PATH)
        if o.get("status")=="ok"
    }

    done_users = load_existing_user_ids(OUT_HYP_JSONL)

    sequences = [s for s in sequences if int(s["user_id"]) not in done_users]

    print("Remaining sequences:",len(sequences))

    prompt_exists = OUT_PROMPT_CSV.exists()
    usage_exists  = OUT_USAGE_CSV.exists()

    prompt_f = OUT_PROMPT_CSV.open("a",encoding="utf-8",newline="")
    usage_f  = OUT_USAGE_CSV.open("a",encoding="utf-8",newline="")
    hyp_f    = OUT_HYP_JSONL.open("a",encoding="utf-8")

    prompt_w = csv.DictWriter(prompt_f,
                              fieldnames=["user_id","system_prompt","user_prompt"])

    usage_w = csv.DictWriter(usage_f,
                             fieldnames=["user_id","input_tokens","output_tokens","total_tokens"])

    if not prompt_exists:
        prompt_w.writeheader()

    if not usage_exists:
        usage_w.writeheader()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(process_sequence, sequences)

    prompt_f.close()
    usage_f.close()
    hyp_f.close()

    print("\n[DONE]")