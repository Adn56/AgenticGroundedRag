#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM POI DESCRIPTION PIPELINE (FAST, RESUME, ABORT-SAFE, ASYNC)

- groups images by business_id
- loads ALL images per business_id (k=5 representatives)
- sends images directly to the OpenAI Responses API
- writes EACH result immediately to disk (JSONL)
- safe against Ctrl+C / IDE stop
- resume-capable
- ASYNC with max 6 parallel requests
"""

import asyncio
import pandas as pd
import os
import base64
import json
from dotenv import load_dotenv
from pathlib import Path
from openai import AsyncOpenAI


# ============================================================
# ENV
# ============================================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set.")

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset" / "representations"

# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

INPUT_CSV = DATASET_DIR / "poi_representative_images_k5.csv"

OUTPUT_JSONL = DATASET_DIR / "poi_descriptions_llm.jsonl"

MODEL_NAME = "gpt-5.2"
TEMPERATURE = 0.2
MAX_TOKENS = 1000

MAX_CONCURRENT_REQUESTS = 6

# ============================================================
# CLIENT
# ============================================================
client = AsyncOpenAI(api_key=API_KEY)

# ============================================================
# PROMPTS
# ============================================================
SYSTEM_PROMPT = """
You are an expert analyst for scientific description of points of interest (POIs).

Describe the place strictly based on visible elements in the provided images.
Produce a concise, neutral, and factual POI description representing a visually grounded abstraction.
Focus exclusively on the place itself, not on the images, the analysis process, or perception.

All statements must rely exclusively on directly observable visual evidence.
Do not speculate beyond what is visually supported by clear functional or spatial indicators.

ATTRIBUTE DEFINITIONS:

- Spatial Configuration:
Functional and spatial characteristics of the place derived from visible layout,
object arrangement, activity zones, or recurring contextual settings.

- Style:
Surface-level architectural or aesthetic characteristics observable in materials,
forms, and construction.

- Atmosphere:
Visually conveyed mood or character inferred from lighting, openness, and spatial density.

- Target Group:
Types of people visually implied to use the place.

- Distinctive Features:
Visually salient elements that clearly differentiate the place.

- Summary:
A compact factual description of the place synthesizing spatial configuration, style,
atmosphere, and distinctive features.
Exclude target group information and introduce no new elements.
Prefer generalizable, place-defining characteristics over incidental or non-representative details.


INTERPRETATION RULES:
- Aggregate attributes as the union of all visually supported characteristics across the image set.
  Do not restrict attribute lists to properties shared by all images.
- Images may depict heterogeneous sub-areas of the same POI; do not assume visual consistency.
- Attribute values must be directly supported by visual evidence and may occur only in subsets of images.
- If no clearly identifiable spatial structure is visible, allow minimal contextual abstraction
  derived from visible elements only.
- If all attribute lists would otherwise be empty, produce a minimal baseline description
  characterizing the place as lacking a clearly identifiable spatial structure.

OUTPUT RULES:
- Attribute values must be single words or short noun phrases.
- Do NOT use full sentences in attribute lists.
- The Summary must NOT refer to images, analysis, perception, or impressions.
- Avoid phrases such as "the images show", "appears", "overall impression", or similar.
- Use object- and space-centered language.
- Output only the JSON object and nothing else.
"""

USER_PROMPT = """
Analyze the provided images and return the POI description strictly in the following JSON format:

{
  "spatial_configuration": [],
  "style": [],
  "atmosphere": [],
  "target_group": [],
  "distinctive_features": [],
  "summary": ""
}
"""

# ============================================================
# HELPERS
# ============================================================
def load_images_as_base64(paths):
    blocks = []
    for p in paths:
        if isinstance(p, str) and Path(p).exists():
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                blocks.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}"
                })
    return blocks


async def call_llm_async(image_blocks):
    response = await client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": USER_PROMPT}] + image_blocks
            }
        ],
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS
    )
    return response.output_text


def normalize_llm_output(text):
    data = json.loads(text)

    for key in [
        "spatial_configuration",
        "style",
        "atmosphere",
        "target_group",
        "distinctive_features"
    ]:
        if key not in data or data[key] is None:
            data[key] = []
        elif isinstance(data[key], str):
            data[key] = [data[key]]

    if "summary" not in data or not isinstance(data["summary"], str):
        data["summary"] = ""

    return data

# ============================================================
# LOAD DATA + RESUME
# ============================================================
df = pd.read_csv(INPUT_CSV)

processed_business_ids = set()
if OUTPUT_JSONL.exists():
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            processed_business_ids.add(json.loads(line)["business_id"])
    print(f"[RESUME] {len(processed_business_ids)} POIs already processed")

# ============================================================
# ASYNC WORKER
# ============================================================

async def process_business(business_id, g, semaphore):
    async with semaphore:

        image_blocks = load_images_as_base64(g["local_image_path"].tolist())

        print(f"[LLM] business_id={business_id} | images={len(image_blocks)}")


        expected = len(g)
        loaded = len(image_blocks)
        if loaded != expected:
            print(
                f"[WARN] business_id={business_id}: "
                f"{loaded}/{expected} images loaded"
            )

        try:
            if not image_blocks:
                raise RuntimeError("No valid images")

            raw_text = await call_llm_async(image_blocks)
            parsed_output = normalize_llm_output(raw_text)
            status = "ok"

        except Exception as e:
            parsed_output = {
                "spatial_configuration": [],
                "style": [],
                "atmosphere": [],
                "target_group": [],
                "distinctive_features": [],
                "summary": ""
            }
            status = "failed"
            print(f"[ERROR] business_id={business_id}: {e}")

        summary_text = parsed_output.pop("summary", "")

        row = {
            "business_id": int(business_id),
            "representation_id": str(business_id),
            "num_images_used": int(len(image_blocks)),
            "summary": str(summary_text),
            "attributes": parsed_output,
            "status": status
        }

        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ============================================================
# MAIN ASYNC LOOP
# ============================================================
async def main():
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []

    for business_id, g in df.groupby("business_id"):
        if int(business_id) in processed_business_ids:
            print(f"[SKIP] business_id={business_id}")
            continue

        tasks.append(process_business(business_id, g, semaphore))

    await asyncio.gather(*tasks)

# ============================================================
# RUN
# ============================================================
asyncio.run(main())
print("\n[DONE] All results safely written to disk.")
