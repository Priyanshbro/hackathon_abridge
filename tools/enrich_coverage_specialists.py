"""Add specialist-office-visit benefit lines to mock_coverage_271.json for
every patient whose active conditions map to a specialist (detect.py's
SPECIALIST_MAP), not just Elias/Julius.

detect.coverage_awareness() only fires when the 271 actually carries a
matching specialty benefit line -- deliberately, so it never claims coverage
it can't ground. Elias and Julius got a cardiology line by hand early in the
build; every other patient's 271 only had the base "Health Benefit Plan
Coverage" item, so the generator silently had nothing to find for them.
Deterministic (seeded by patient id), so copay/auth values are stable across
reruns.

Output: overwrites data/synthetic-ambient-fhir-25/mock_coverage_271.json and
its .jsonl mirror.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "synthetic-ambient-fhir-25" / "synthetic-ambient-fhir-25.jsonl"
COVERAGE_JSON = ROOT / "data" / "synthetic-ambient-fhir-25" / "mock_coverage_271.json"
COVERAGE_JSONL = ROOT / "data" / "synthetic-ambient-fhir-25" / "mock_coverage_271.jsonl"

SPECIALIST_MAP = {
    "hypertension": ("cardiologist", "cardiology"),
    "hyperlipidemia": ("cardiologist", "cardiology"),
    "diabetes": ("endocrinologist", "endocrinology"),
    "prediabetes": ("endocrinologist", "endocrinology"),
    "migraine": ("neurologist", "neurology"),
    "osteoarthritis": ("orthopedist", "orthopedics"),
    "depression": ("psychiatrist", "psychiatry"),
    "anxiety": ("psychiatrist", "psychiatry"),
    "gingivitis": ("periodontist", "periodontics"),
    "cancer": ("oncologist", "oncology"),
    "hepatitis": ("hepatologist", "hepatology"),
    "liver": ("hepatologist", "hepatology"),
}

DESCRIPTIONS = (
    "PCP referral required for specialist visits under this plan.",
    "No referral required; must use an in-network specialist.",
)


def specialty_names(rec: dict) -> list[str]:
    labels = rec["patient_context"]["longitudinal_summary"]["condition_labels"]
    seen = []
    for label in labels:
        for keyword, (_, specialty) in SPECIALIST_MAP.items():
            if keyword in label.lower() and specialty not in seen:
                seen.append(specialty)
    return seen


def make_item(pid: str, specialty: str) -> dict:
    rng = random.Random(f"{pid}:{specialty}")
    copay = rng.choice([25, 30, 35, 40, 45, 50, 60])
    auth_required = rng.random() < 0.5
    return {
        "category": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/ex-benefitcategory",
                    "code": "88",
                    "display": f"Specialist office visit ({specialty})",
                }
            ]
        },
        "excluded": False,
        "authorizationRequired": auth_required,
        "description": DESCRIPTIONS[0] if auth_required else DESCRIPTIONS[1],
        "benefit": [
            {"type": {"coding": [{"code": "copay"}]}, "allowedMoney": {"value": copay, "currency": "USD"}}
        ],
    }


def main() -> None:
    cohort = {}
    with open(COHORT) as f:
        for line in f:
            rec = json.loads(line)
            cohort[rec["metadata"]["patient_id"]] = rec

    coverage = json.load(open(COVERAGE_JSON))
    added = 0
    for entry in coverage:
        pid = entry["patient_id"]
        rec = cohort.get(pid)
        if rec is None:
            continue
        insurance = entry["eligibility_response"]["insurance"][0]
        if not insurance.get("inforce"):
            continue  # uninsured -- no benefit lines to add

        existing_specialties = {
            i["category"]["coding"][0]["display"].split("(")[-1].rstrip(")")
            for i in insurance.get("item", [])
            if "Specialist" in i["category"]["coding"][0]["display"]
        }
        for specialty in specialty_names(rec):
            if specialty in existing_specialties:
                continue
            insurance.setdefault("item", []).append(make_item(pid, specialty))
            added += 1

    json.dump(coverage, open(COVERAGE_JSON, "w"), indent=2)
    with open(COVERAGE_JSONL, "w") as f:
        for entry in coverage:
            f.write(json.dumps(entry) + "\n")

    print(f"Added {added} specialist benefit lines across {len(coverage)} patients.")


if __name__ == "__main__":
    main()
