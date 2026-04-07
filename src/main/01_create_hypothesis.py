#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-ONLY SEMANTIC HYPOTHESIS GENERATION

- Resumable (skip processed user_ids)
- Writes AFTER EACH RESPONSE
- Stores:
    (1) hypothesis JSONL
    (2) exact prompts CSV (verbatim LLM input)
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

SEQ_PATH = DATASET_DIR / "splits" / f"test_eval_1000.jsonl"
SEM_PATH = DATASET_DIR / "core" / "poi_semantic_states.jsonl"

OUT_DIR = PROJECT_ROOT/ "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_HYP_JSONL = OUT_DIR / f"hypothesis_test_eval_1000.jsonl"
OUT_PROMPT_CSV = OUT_DIR / f"prompts_test_eval_1000.csv"
OUT_USAGE_CSV = OUT_DIR / f"usage_test_eval_1000.csv"



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

    done_users = load_existing_user_ids(OUT_HYP_JSONL)

    prompt_exists = OUT_PROMPT_CSV.exists()
    prompt_f = OUT_PROMPT_CSV.open("a", encoding="utf-8", newline="")
    prompt_w = csv.DictWriter(
        prompt_f,
        fieldnames=["user_id", "seed", "system_prompt", "user_prompt"]
    )
    if not prompt_exists:
        prompt_w.writeheader()
        prompt_f.flush()

    # ============================================================
    # USAGE CSV (token logging)
    # ============================================================
    usage_exists = OUT_USAGE_CSV.exists()
    usage_f = OUT_USAGE_CSV.open("a", encoding="utf-8", newline="")
    usage_w = csv.DictWriter(
        usage_f,
        fieldnames=["user_id", "seed", "input_tokens", "output_tokens", "total_tokens"]
    )
    if not usage_exists:
        usage_w.writeheader()
        usage_f.flush()

    with OUT_HYP_JSONL.open("a", encoding="utf-8") as hyp_f:
        for seq in sequences:
            user_id = int(seq["user_id"])
            if user_id in done_users:
                continue

            context = seq["input"]
            context_ids = [int(v["business_id"]) for v in context]

            if any(b not in semantic for b in context_ids):
                continue

            poi_blocks = ""
            for i, v in enumerate(context, 1):
                poi_blocks += f"POI {i}:\n{poi_block(v['date'], semantic[int(v['business_id'])])}\n\n"

            user_prompt = USER_PROMPT_TEMPLATE \
                .replace("<<<POI_BLOCKS>>>", poi_blocks)

            resp = client.responses.create(
                model=LLM_MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
            )
            usage = resp.usage

            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            total_tokens = usage.total_tokens


            hypothesis = extract_json(resp.output_text)

            # --- write prompt (EXACT) ---
            prompt_w.writerow({
                "user_id": user_id,
                "system_prompt": SYSTEM_PROMPT.strip(),
                "user_prompt": user_prompt.strip()
            })
            prompt_f.flush()

            # --- write hypothesis ---
            hyp_f.write(json.dumps({
                "user_id": user_id,
                "sequence": context_ids,
                "target": int(seq["target"]["business_id"]),
                "target_seen_in_last_9": seq["target_seen_in_last_9"],
                "prediction_correct": seq.get("prediction_correct"),
                "hypothesis": hypothesis
            }, ensure_ascii=False) + "\n")
            hyp_f.flush()

            usage_w.writerow({
                "user_id": user_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            })
            usage_f.flush()


            done_users.add(user_id)
            print(f"[OK] user_id={user_id} | hypothesis + prompt + usage written")

    prompt_f.close()
    usage_f.close()
    print("\n[DONE]")

if __name__ == "__main__":
    main()
