#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-ONLY SEMANTIC HYPOTHESIS GENERATION (RESUMABLE, SEED-SPECIFIC)

- One output file per seed
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
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

SEED_TAG   = "seed2026"      # change to seed2027 / seed2028
LLM_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"

SEQ_PATH = DATASET_DIR / "validation" / f"validation_dev_100_{SEED_TAG}_with_predicted_cluster.jsonl"
CLUSTER_PROFILE_PATH = DATASET_DIR / "core" / "cluster_attribute_profiles.jsonl"

OUT_DIR = DATASET_DIR / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_HYP_JSONL = OUT_DIR / f"hypotheses_{SEED_TAG}_clusterOnly.jsonl"
OUT_PROMPT_CSV = OUT_DIR / f"prompts_{SEED_TAG}_clusterOnly.csv"
OUT_USAGE_CSV = OUT_DIR / f"usage_{SEED_TAG}_clusterOnly.csv"



SYSTEM_PROMPT = """
Generate a semantically grounded description of a single point of interest
that belongs to a given semantic cluster.

Describe the place directly as an observed entity.
Do not include reasoning or explanations.
Do not reference any specific real-world location.
"""

USER_PROMPT_TEMPLATE = """
You are given a semantic cluster that represents the space
to which the next point of interest belongs.

CLUSTER DESCRIPTION:

<<<PREDICTED_CLUSTER_BLOCK>>>

Task:

Construct a concrete and coherent description of a single
point of interest that belongs to this cluster.

The output must describe one specific place that is
semantically consistent with the cluster attributes.

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

    cluster_profiles = {
        int(o["cluster_id"]): o["attributes"]
        for o in load_jsonl(CLUSTER_PROFILE_PATH)
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

            predicted_cluster = int(seq["predicted_cluster"])

            user_prompt = USER_PROMPT_TEMPLATE.replace(
                "<<<PREDICTED_CLUSTER_BLOCK>>>",
                format_cluster_block(predicted_cluster, cluster_profiles)
            )

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
                "seed": SEED_TAG,
                "system_prompt": SYSTEM_PROMPT.strip(),
                "user_prompt": user_prompt.strip()
            })
            prompt_f.flush()

            # --- write hypothesis ---
            hyp_f.write(json.dumps({
                "user_id": user_id,
                "target": int(seq["target"]["business_id"]),
                "target_seen_in_last_9": seq["target_seen_in_last_9"],
                "prediction_correct": seq.get("prediction_correct"),
                "predicted_cluster": seq["predicted_cluster"],
                "hypothesis": hypothesis
            }, ensure_ascii=False) + "\n")
            hyp_f.flush()

            usage_w.writerow({
                "user_id": user_id,
                "seed": SEED_TAG,
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