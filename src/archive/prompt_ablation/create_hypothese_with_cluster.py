#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-ONLY SEMANTIC HYPOTHESIS GENERATION with Cluster

- Resumable (skip processed user_ids)
- Writes AFTER EACH RESPONSE
- Stores:
    (1) hypotheses JSONL
    (2) exact prompts CSV (verbatim LLM input)
- NO embeddings
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import threading

# ============================================================
# CONFIG
# ============================================================

LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0
MAX_WORKERS = 6

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

SEQ_PATH = PROJECT_ROOT / "prompt_ablation" / f"test_eval_1000_with_predicted_cluster.jsonl"
SEM_PATH = DATASET_DIR / "core" / "poi_semantic_states.jsonl"
CLUSTER_PROFILE_PATH = DATASET_DIR / "core" / "cluster_attribute_profiles.jsonl"

OUT_DIR = PROJECT_ROOT / "prompt_ablation"

OUT_HYP_JSONL = OUT_DIR / f"hypotheses_with_predicted_cluster.jsonl"
OUT_PROMPT_CSV = OUT_DIR / f"prompts_with_predicted_cluster.csv"
OUT_USAGE_CSV = OUT_DIR / f"usage_with_predicted_cluster.csv"

SYSTEM_PROMPT = """
Generate a semantically grounded description of a single point of interest
that plausibly continues a given sequence.

For each output field, identify attributes that recur across multiple POIs
and form a clear semantic pattern, and carry those attributes over directly
into the hypothesis.
Ignore isolated, weakly supported, or singular observations.
Do not mix unrelated characteristics across different places.

If a conflict arises between sequence-derived attributes and cluster-based hints,
the sequence-derived attributes must always take precedence.

Describe the place directly as an observed entity.
Do not include reasoning, explanations, or transition descriptions.
"""

USER_PROMPT_TEMPLATE = """
You are given the LAST 9 points of interest visited by a user.
Each POI is described using visually grounded attributes.

POI SEQUENCE (ordered from oldest to most recent):

<<<POI_BLOCKS>>>

Additional probabilistic hint:

A Markov-based sequence model predicts that the next point of interest
is most likely associated with the following semantic cluster:

<<<PREDICTED_CLUSTER_BLOCK>>>

Decision policy (must be followed):

1) Sequence-first carryover:
   For each output field, if attributes recur across multiple POIs
   and form a clear semantic pattern, carry those attributes over
   directly into the hypothesis.
   Preserve concreteness and do not generalize away dominant details.

2) Cluster-as-prior only when needed:
   If a field does NOT exhibit a clear or dominant pattern in the sequence,
   use the predicted cluster only to fill that field with attributes
   that are typical for and semantically compatible with the cluster.
   The cluster must not override clearly recurrent sequence attributes.

3) No mixing:
   Do not combine unrelated features from different POIs.
   The output must describe one coherent, single place.

4) No target leakage:
   Do not assume knowledge about the true next POI or its identity.
   Base all attributes strictly on the provided sequence and the probabilistic hint.

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

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_existing_user_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(int(json.loads(line)["user_id"]))
            except Exception:
                continue
    return done

def format_datetime(ts: str) -> str:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")

def poi_block(visit_date, poi):
    return json.dumps({
        "visit_time": format_datetime(visit_date),
        "summary": poi["summary"],
        **poi["attributes"]
    }, ensure_ascii=False, indent=2)

def format_cluster_block(cid, profiles):
    return json.dumps({
        "cluster_id": cid,
        "attributes": profiles.get(cid, {})
    }, ensure_ascii=False, indent=2)

def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in LLM output")
    return json.loads(text[start:end + 1])

# ============================================================
# MAIN
# ============================================================

def main():

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI()

    sequences = load_jsonl(SEQ_PATH)

    semantic = {
        int(o["business_id"]): o
        for o in load_jsonl(SEM_PATH)
        if o.get("status") == "ok"
    }

    cluster_profiles = {
        int(o["cluster_id"]): o["attributes"]
        for o in load_jsonl(CLUSTER_PROFILE_PATH)
    }

    done_users = load_existing_user_ids(OUT_HYP_JSONL)

    lock = threading.Lock()

    prompt_exists = OUT_PROMPT_CSV.exists()
    prompt_f = OUT_PROMPT_CSV.open("a", encoding="utf-8", newline="")
    prompt_w = csv.DictWriter(prompt_f, fieldnames=["user_id","system_prompt","user_prompt"])

    if not prompt_exists:
        prompt_w.writeheader()

    usage_exists = OUT_USAGE_CSV.exists()
    usage_f = OUT_USAGE_CSV.open("a", encoding="utf-8", newline="")
    usage_w = csv.DictWriter(
        usage_f,
        fieldnames=["user_id","input_tokens","output_tokens","total_tokens"]
    )

    if not usage_exists:
        usage_w.writeheader()

    hyp_f = OUT_HYP_JSONL.open("a", encoding="utf-8")

    def process(seq):

        user_id = int(seq["user_id"])

        if user_id in done_users:
            return

        context = seq["input"]
        context_ids = [int(v["business_id"]) for v in context]

        if any(b not in semantic for b in context_ids):
            return

        poi_blocks = ""

        for i, v in enumerate(context, 1):
            poi_blocks += f"POI {i}:\n{poi_block(v['date'], semantic[int(v['business_id'])])}\n\n"

        user_prompt = USER_PROMPT_TEMPLATE \
            .replace("<<<POI_BLOCKS>>>", poi_blocks) \
            .replace(
            "<<<PREDICTED_CLUSTER_BLOCK>>>",
            format_cluster_block(int(seq["predicted_cluster"]), cluster_profiles)
        )

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
                "predicted_cluster":seq["predicted_cluster"],
                "hypothesis":hypothesis
            }, ensure_ascii=False) + "\n")
            hyp_f.flush()

            usage_w.writerow({
                "user_id":user_id,
                "input_tokens":usage.input_tokens,
                "output_tokens":usage.output_tokens,
                "total_tokens":usage.total_tokens
            })
            usage_f.flush()

            done_users.add(user_id)

        print(f"[OK] user_id={user_id}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process, sequences)

    prompt_f.close()
    usage_f.close()
    hyp_f.close()

    print("\n[DONE]")

if __name__ == "__main__":
    main()