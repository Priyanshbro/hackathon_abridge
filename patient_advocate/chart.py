from __future__ import annotations
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic-ambient-fhir-25"

_cohort = None
_coverage = None

# Static wearable fixture -- only Elias Wisozk has one (the hero demo case).
# Numbers from PLAN.md: sleeps on sister's sofa bed, sleep/RHR/awakenings
# shift in the weeks before this visit.
WEARABLE = {
    "4b4735a2-ee12-ec86-041f-3ba4d5c81ec9": {  # Elias Wisozk
        "sleep_hours": {"before": 6.8, "after": 5.4},
        "awakenings": {"before": 1.1, "after": 3.1},
        "resting_hr": {"before": 72, "after": 76},
    },
}


def load_cohort() -> list[dict]:
    global _cohort
    if _cohort is None:
        with open(DATA_DIR / "synthetic-ambient-fhir-25.jsonl") as f:
            _cohort = [json.loads(line) for line in f]
    return _cohort


def load_coverage() -> dict[str, dict]:
    global _coverage
    if _coverage is None:
        records = json.load(open(DATA_DIR / "mock_coverage_271.json"))
        _coverage = {r["patient_id"]: r for r in records}
    return _coverage


def record_for(patient_id: str) -> dict:
    for r in load_cohort():
        if r["metadata"]["patient_id"] == patient_id:
            return r
    raise KeyError(patient_id)


def coverage_for(patient_id: str) -> dict:
    return load_coverage()[patient_id]


def wearable_for(patient_id: str) -> dict | None:
    return WEARABLE.get(patient_id)


def condition_labels(rec: dict) -> list[str]:
    return rec["patient_context"]["longitudinal_summary"]["condition_labels"]


def medication_labels(rec: dict) -> list[str]:
    return rec["patient_context"]["longitudinal_summary"]["medication_labels"]


def observations(rec: dict) -> list[dict]:
    return rec["encounter_fhir"]["related_resources"].get("Observation", [])


def medication_requests(rec: dict) -> list[dict]:
    return rec["encounter_fhir"]["related_resources"].get("MedicationRequest", [])


def encounter_class(rec: dict) -> str:
    return rec["encounter_fhir"]["encounter"].get("class", {}).get("code", "")


def service_provider(rec: dict) -> str:
    return rec["encounter_fhir"]["encounter"].get("serviceProvider", {}).get("display", "")


def visit_title(rec: dict) -> str:
    return rec["metadata"].get("visit_title", "")
