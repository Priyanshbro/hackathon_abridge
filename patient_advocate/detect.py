from __future__ import annotations
from dataclasses import dataclass

import chart


@dataclass
class Candidate:
    id: str
    topic: str  # for resolution matching -- "|"-delimited keywords, e.g. "cost|afford|dollar"
    kind: str  # gap | score | drug_disease | clue | wearable | referral | preventive
    # referral/preventive matter specifically to deliver.py's goals-of-care
    # guard, which drops them outright for hospice/SNF encounters
    trigger: str  # human-readable why
    evidence: list[dict]  # [{source, field, value, date}] -- must be verifiable
    priority: float
    resolved_by: str | None = None  # utterance idx + span, set during the visit
    verifiable_from_record: bool = True
    # False when the claim depends on history this encounter does not contain
    # -- prior screening, whether a lab was ever repeated. longitudinal_summary
    # carries procedure COUNTS with no labels, so "never addressed" is
    # unknowable. Ground-truth review: 17 of 52 high-severity gaps were marked
    # "is there anything in history that can explain". These deliver as a
    # QUESTION ("Am I up to date on my screening?"), never an assertion --
    # honest about the uncertainty, and still the right ask even with a full
    # chart, since real records are always incomplete.
    plan_verbs: tuple[str, ...] | None = None  # override resolve.py's default
    # clinical plan-verb list when "addressed" means something more specific
    # than start/order/refer -- e.g. cost_adherence_risk should pass
    # contingency verbs ("call back", "assistance program", "sliding scale",
    # "90-day supply", "price check") so a plain reassurance line like
    # "these are all cheap generics" never counts as resolving it.


# keyword found in condition_labels -> (specialist title, benefit-line specialty name)
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


def coverage_awareness(rec: dict) -> Candidate | None:
    """Patient has an active condition mapping to a specialist type, their
    coverage is in force, and that specialist was never discussed. Only
    fires when the 271 actually carries a benefit line for that specialty
    -- a condition/specialist match with no matching coverage evidence is
    skipped rather than claimed, so groundedness never has to trust a
    mismatched benefit line."""
    coverage = chart.coverage_for(rec["metadata"]["patient_id"])
    insurance = coverage["eligibility_response"]["insurance"][0]
    if not insurance["inforce"]:
        return None

    transcript_lower = rec["transcript"].lower()
    for label in chart.condition_labels(rec):
        for keyword, (specialist, specialty) in SPECIALIST_MAP.items():
            if keyword not in label.lower() or specialist in transcript_lower:
                continue
            specialist_item = next(
                (
                    i
                    for i in insurance["item"]
                    if specialty in i["category"]["coding"][0]["display"].lower()
                ),
                None,
            )
            if specialist_item is None:
                continue  # no coverage evidence for this specialty -- don't claim it
            evidence = [
                {"source": "longitudinal_summary", "field": "condition_labels", "value": label},
                {"source": "coverage_271", "field": "insurance.inforce", "value": True},
                {
                    "source": "coverage_271",
                    "field": "coverage.payor",
                    "value": coverage["coverage"]["payor"][0]["display"],
                },
                {
                    "source": "coverage_271",
                    "field": "insurance.item.benefit.copay",
                    "value": specialist_item["benefit"][0]["allowedMoney"]["value"],
                },
                # deterministic, already computed above -- surfaced as its own
                # evidence item so Gate B doesn't have to trust the trigger text
                {"source": "transcript", "field": f"mentions_{specialist}", "value": False},
            ]
            return Candidate(
                id=f"coverage-{specialist}",
                topic=specialist,
                kind="referral",
                trigger=f"active condition ({label}) maps to {specialist}, never discussed",
                evidence=evidence,
                priority=0.5,
            )
    return None


# --- fill these in ---

def bp_control(rec: dict) -> Candidate | None:
    """BP >=140/90 -> stage 2."""
    raise NotImplementedError


def a1c_threshold(rec: dict) -> Candidate | None:
    """5.7-6.4 prediabetes, >=6.5 diabetes."""
    raise NotImplementedError


def wearable_delta(rec: dict) -> Candidate | None:
    """Sleep / RHR / steps shift beyond threshold in window. Uses
    chart.wearable_for(patient_id) -- only Elias has fixture data."""
    raise NotImplementedError


def transcript_clue(rec: dict) -> Candidate | None:
    """Lifestyle/exposure cue in patient's own words (sofa bed, shift
    work, new med) that the doctor never picks up on."""
    raise NotImplementedError


def cost_adherence_risk(rec: dict) -> Candidate | None:
    """Prior cost/coverage disruption + new prescriptions today + no
    written cost contingency. See PLAN.md for the 4-step build."""
    raise NotImplementedError


GENERATORS = [
    bp_control,
    a1c_threshold,
    wearable_delta,
    transcript_clue,
    cost_adherence_risk,
    coverage_awareness,
]


def run_all(rec: dict) -> list[Candidate]:
    return [c for gen in GENERATORS for c in [gen(rec)] if c is not None]
