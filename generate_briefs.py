import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "crossings.csv"
OUTPUT_PATH = DATA_DIR / "briefs.csv"
BASE_URL = os.getenv("VLLM_BASE_URL", "http://10.184.47.255:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MAX_WORKERS = int(os.getenv("BRIEF_WORKERS", "12"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

BRIEF_INPUT_COLUMNS = [
    "CartID",
    "Name",
    "creek",
    "inlets_active",
    "inlets_poor_failed",
    "inlets_uninspected",
    "dist_nearest_inlet_ft",
    "imp_projects_nearby",
    "imp_restoration_nearby",
    "imp_planned_cost",
    "dist_nearest_imp_ft",
    "imp_next_fiscal_year",
    "sso_same_creek",
    "sso_rain_related",
    "risk_score",
    "risk_rank",
    "nearest_gauge",
    "gauge_dist_mi",
    "gauge_height_ft",
    "gauge_reading_time",
]
GAUGE_UNAVAILABLE_SENTINEL = -999999

SYSTEM_PROMPT = """You are a San Antonio municipal infrastructure and flood-safety analyst.
Given the provided crossing facts, return a valid JSON object with exactly these three keys:
- "warning_en": Exactly two concise sentences in English for drivers/residents.
- "warning_es": An accurate Spanish translation of warning_en, also exactly two sentences.
- "maintenance_suggestion": One concrete, actionable Public Works inspection or maintenance recommendation tied directly to the supplied facts.

DATA RULES:
- Use only the supplied facts. Do not invent a creek, landmark, closure, event, cause, or infrastructure condition.
- "risk_score" is a relative infrastructure risk index (z-score), not a flood probability. Never express it as a percentage.
- If "creek" is null, the zero SSO values mean no named creek is available; they do NOT establish that there have been zero sewer overflows. Do not claim zero overflows or invent a creek name.
- "imp_projects_nearby" and "imp_next_fiscal_year" describe upcoming planned projects, not completed work.
- The USGS gauge is a nearby regional reference, not a measurement at the crossing. Its reported height alone does not establish flooding or crossing safety, and no flood threshold is supplied. Do not infer a flood warning, safety status, or trend from it.
- A null gauge value or "gauge_height_ft" equal to -999999 means the height is unavailable; do not repeat that sentinel as a real measurement.
- Keep the warning actionable but do not tell drivers that the crossing is safe based on a gauge reading."""


def clean_record(record):
    """Keep only documented fields and turn missing/sentinel values into JSON nulls."""
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


def parse_json_safe(content):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    required = ("warning_en", "warning_es", "maintenance_suggestion")
    if any(not isinstance(parsed.get(key), str) or not parsed[key].strip() for key in required):
        raise ValueError("Model response is missing a non-empty brief field")
    return {key: parsed[key].strip() for key in required}


def fallback_brief(record):
    name = str(record["Name"])
    rank = int(record["risk_rank"])
    score = float(record["risk_score"])
    poor = int(record["inlets_poor_failed"])
    uninspected = int(record["inlets_uninspected"])
    has_creek = pd.notna(record.get("creek")) and str(record.get("creek", "")).strip()
    if has_creek:
        sso_context = (
            f" The dataset records {int(record['sso_rain_related'])} rain-related and "
            f"{int(record['sso_same_creek'])} total same-creek sewer overflow reports"
        )
        sso_context_es = (
            f" El conjunto de datos registra {int(record['sso_rain_related'])} reportes de "
            f"desbordamiento relacionados con lluvia y {int(record['sso_same_creek'])} en total "
            "en el mismo arroyo"
        )
    else:
        sso_context = " Same-creek sewer overflow counts are unknown because no named creek is available"
        sso_context_es = " Se desconocen los desbordamientos en el mismo arroyo porque no hay uno identificado"
    return {
        "warning_en": (
            f"Use caution near {name}; its relative infrastructure risk rank is #{rank} "
            f"(index {score:.2f}). During heavy rainfall, turn around, don't drown.{sso_context}."
        ),
        "warning_es": (
            f"Tenga precaución cerca de {name}; su rango relativo de riesgo de infraestructura es "
            f"#{rank} (índice {score:.2f}). Durante lluvias intensas, dé la vuelta y no cruce aguas inundadas.{sso_context_es}."
        ),
        "maintenance_suggestion": (
            f"Inspect the nearby storm inlets, prioritizing {poor} poor or failed and "
            f"{uninspected} uninspected inlets, then document drainage conditions."
        ),
    }


def generate_brief(record):
    payload = clean_record(record)
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            brief = parse_json_safe(response.choices[0].message.content or "")
            return {**record, **brief}
        except Exception:
            continue
    return {**record, **fallback_brief(record)}


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Crossing data not found: {INPUT_PATH}")

    crossings = pd.read_csv(INPUT_PATH)
    required_columns = {"CartID", "Name", "risk_score", "risk_rank"}
    missing = sorted(required_columns - set(crossings.columns))
    if missing:
        raise ValueError(f"crossings.csv is missing required columns: {', '.join(missing)}")

    records = crossings.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(tqdm(pool.map(generate_brief, records), total=len(records)))

    briefs = pd.DataFrame(results).sort_values("risk_rank", ascending=True)
    briefs.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(briefs)} briefs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
