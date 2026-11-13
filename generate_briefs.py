import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(base_url="[http://10.184.47.255:8000/v1](http://10.184.47.255:8000/v1)", api_key="EMPTY")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"

DATA_DIR = Path("data/processed")
df = pd.read_csv(DATA_DIR / "crossings.csv")

SYSTEM_PROMPT = """You are a San Antonio municipal flood-safety analyst.
Given ONLY the factual infrastructure metrics for a low water crossing, return a valid JSON object with exactly three keys:
- "warning_en": Exactly two sentences warning drivers/residents about flood risk at this crossing based ONLY on the provided numbers.
- "warning_es": Exact Spanish translation of warning_en (two sentences).
- "maintenance_suggestion": One concrete, actionable Public Works inspection or maintenance recommendation tied directly to the provided numbers.

STRICT DATA RULES:
- Use ONLY the provided fields (inlets_active, inlets_poor_failed, inlets_uninspected, dist_nearest_inlet_ft, imp_projects_nearby, imp_restoration_nearby, imp_planned_cost, dist_nearest_imp_ft, imp_next_fiscal_year, sso_same_creek, sso_rain_related, risk_score, risk_rank).
- "risk_score" is a relative infrastructure warning index (z-score), NOT a flood probability percentage. Never express it as a percentage.
- If "creek" is null, "sso_same_creek" and "sso_rain_related" are 0 because the crossing has no named creek in the dataset—do NOT claim there have been zero sewer overflows, and do NOT invent a creek name.
- "imp_projects_nearby" and "imp_next_fiscal_year" (FY 2026-2030) are upcoming planned projects, not completed past work.
- Do NOT invent landmarks, historical storms, or facts not present in the input."""

def clean_record(row):
    return {
        k: (None if pd.isna(v) else v)
        for k, v in row.items()
        if k not in {"x", "y"}
    }

def parse_json_safe(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(cleaned)

def generate_brief(row_dict):
    payload = clean_record(row_dict)
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload)}
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            parsed = parse_json_safe(resp.choices[0].message.content)
            return {
                **row_dict,
                "warning_en": parsed.get("warning_en", ""),
                "warning_es": parsed.get("warning_es", ""),
                "maintenance_suggestion": parsed.get("maintenance_suggestion", "")
            }
        except Exception:
            continue
    return {
        **row_dict,
        "warning_en": f"Exercise caution at {row_dict['Name']} (Risk Rank #{int(row_dict['risk_rank'])}, score {row_dict['risk_score']}). Turn around, don't drown during heavy rainfall.",
        "warning_es": f"Tenga precaución en {row_dict['Name']} (Rango de riesgo #{int(row_dict['risk_rank'])}, puntuación {row_dict['risk_score']}). Dé la vuelta, no se ahogue durante lluvias intensas.",
        "maintenance_suggestion": f"Inspect nearby storm inlets ({int(row_dict['inlets_poor_failed'])} poor/failed, {int(row_dict['inlets_uninspected'])} uninspected) and evaluate drainage capacity."
    }

records = df.to_dict(orient="records")
with ThreadPoolExecutor(max_workers=12) as pool:
    results = list(tqdm(pool.map(generate_brief, records), total=len(records)))

briefs_df = pd.DataFrame(results)
briefs_df.to_csv(DATA_DIR / "briefs.csv", index=False)
print(f"Saved {len(briefs_df)} briefs to {DATA_DIR / 'briefs.csv'}")