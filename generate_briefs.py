"""Generate the standard bilingual brief set for every processed crossing.

Run after 01_clean.ipynb:
    python generate_briefs.py

Unchanged model-generated rows are reused. Pass --force to regenerate all rows.
"""

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from vlm_config import API_KEY, BASE_URL, MODEL, REQUEST_TIMEOUT


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "crossings.csv"
OUTPUT_PATH = DATA_DIR / "briefs.csv"
DEFAULT_WORKERS = int(os.getenv("BRIEF_WORKERS", "12"))
GAUGE_UNAVAILABLE_SENTINEL = -999999

BRIEF_FIELDS = ("warning_en", "warning_es", "maintenance_suggestion")
BRIEF_INPUT_COLUMNS = [
    "CartID", "Name", "creek", "inlets_active", "inlets_poor_failed", "inlets_uninspected",
    "dist_nearest_inlet_ft", "imp_projects_nearby", "imp_restoration_nearby", "imp_planned_cost",
    "dist_nearest_imp_ft", "imp_next_fiscal_year", "sso_same_creek", "sso_rain_related",
    "risk_score", "risk_rank", "nearest_gauge", "gauge_dist_mi", "gauge_height_ft",
    "gauge_reading_time",
]

SYSTEM_PROMPT = """You are a San Antonio municipal infrastructure and flood-safety analyst.
Given the provided crossing facts, return a valid JSON object with exactly these three keys:
- "warning_en": Exactly two concise sentences in English for drivers and residents.
- "warning_es": An accurate Spanish translation of warning_en, also exactly two sentences.
- "maintenance_suggestion": One concrete, actionable Public Works inspection or maintenance recommendation tied directly to the supplied facts.

DATA RULES:
- Use only the supplied facts. Do not invent a creek, landmark, closure, event, cause, or infrastructure condition.
- "risk_score" is a relative infrastructure risk index (z-score), not a flood probability. Never express it as a percentage.
- If "creek" is null, the zero SSO values mean no named creek is available; they do not establish that there have been zero sewer overflows.
- "imp_projects_nearby" and "imp_next_fiscal_year" describe upcoming planned projects, not completed work.
- The USGS gauge is a nearby regional reference, not a measurement at the crossing. Its height alone does not establish flooding or crossing safety, and no flood threshold is supplied.
- A null gauge value means the height is unavailable. Do not infer a warning, safety status, or trend from it.
- Never tell drivers that a crossing is safe based on a gauge reading."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate every brief, including unchanged rows.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent model requests.")
    return parser.parse_args()


def clean_record(record):
    """Return documented facts with JSON-safe missing values."""
    payload = {}
    for column in BRIEF_INPUT_COLUMNS:
        value = record.get(column)
        if pd.isna(value):
            value = None
        elif column == "gauge_height_ft" and float(value) <= GAUGE_UNAVAILABLE_SENTINEL:
            value = None
        elif hasattr(value, "item"):
            value = value.item()
        payload[column] = value
    return payload


def input_signature(record):
    encoded = json.dumps(clean_record(record), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_json_response(content):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    if any(not isinstance(parsed.get(key), str) or not parsed[key].strip() for key in BRIEF_FIELDS):
        raise ValueError("Model response is missing a non-empty brief field")
    return {key: parsed[key].strip() for key in BRIEF_FIELDS}


def fallback_brief(record):
    name = str(record["Name"])
    rank = int(record["risk_rank"])
    score = float(record["risk_score"])
    poor = int(record["inlets_poor_failed"])
    uninspected = int(record["inlets_uninspected"])
    return {
        "warning_en": (
            f"Use caution near {name}; its relative infrastructure risk rank is #{rank} "
            f"with an index of {score:.2f}. During heavy rainfall, turn around, don't drown."
        ),
        "warning_es": (
            f"Tenga precaución cerca de {name}; su rango relativo de riesgo de infraestructura es "
            f"#{rank}, con un índice de {score:.2f}. Durante lluvias intensas, dé la vuelta y no cruce aguas inundadas."
        ),
        "maintenance_suggestion": (
            f"Inspect nearby storm inlets, prioritizing {poor} poor or failed and "
            f"{uninspected} uninspected inlets, then document drainage conditions."
        ),
    }


def generate_brief(client, record):
    last_error = ""
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(clean_record(record), ensure_ascii=False)},
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return {
                **record,
                **parse_json_response(response.choices[0].message.content or ""),
                "input_signature": input_signature(record),
                "brief_source": "model",
                "brief_error": "",
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:300]

    return {
        **record,
        **fallback_brief(record),
        "input_signature": input_signature(record),
        "brief_source": "fallback",
        "brief_error": last_error,
    }


def reusable_briefs():
    if not OUTPUT_PATH.exists():
        return {}
    existing = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
    required = {"CartID", "input_signature", "brief_source", *BRIEF_FIELDS}
    if not required.issubset(existing.columns):
        return {}
    existing = existing[existing["brief_source"] == "model"]
    return {row.CartID: row._asdict() for row in existing.itertuples(index=False)}


def write_output(rows):
    output = pd.DataFrame(rows).sort_values("risk_rank", ascending=True)
    temporary_path = OUTPUT_PATH.with_suffix(".csv.tmp")
    output.to_csv(temporary_path, index=False, encoding="utf-8")
    temporary_path.replace(OUTPUT_PATH)
    return output


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Crossing data not found: {INPUT_PATH}")

    crossings = pd.read_csv(INPUT_PATH)
    required = {"CartID", "Name", "risk_score", "risk_rank", "inlets_poor_failed", "inlets_uninspected"}
    missing = sorted(required - set(crossings.columns))
    if missing:
        raise ValueError(f"crossings.csv is missing required columns: {', '.join(missing)}")
    if crossings["CartID"].duplicated().any():
        raise ValueError("crossings.csv contains duplicate CartID values")

    records = crossings.to_dict(orient="records")
    reusable = {} if args.force else reusable_briefs()
    completed = []
    pending = []

    for record in records:
        prior = reusable.get(record["CartID"])
        if prior and prior["input_signature"] == input_signature(record):
            completed.append({
                **record,
                **{key: prior[key] for key in BRIEF_FIELDS},
                "input_signature": prior["input_signature"],
                "brief_source": "model",
                "brief_error": "",
            })
        else:
            pending.append(record)

    if pending:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT, max_retries=0)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            generated = list(tqdm(
                pool.map(lambda row: generate_brief(client, row), pending),
                total=len(pending),
                desc="Generating briefs",
            ))
        completed.extend(generated)

    output = write_output(completed)
    model_count = int((output["brief_source"] == "model").sum())
    fallback_count = int((output["brief_source"] == "fallback").sum())
    print(f"Saved {len(output)} briefs to {OUTPUT_PATH}")
    print(f"Reused {len(records) - len(pending)} unchanged model briefs; generated {len(pending)} rows")
    print(f"Model briefs: {model_count}; fallbacks: {fallback_count}")
    if fallback_count:
        print("Run the command again to retry fallback rows, or inspect brief_error in briefs.csv.")


if __name__ == "__main__":
    main()
