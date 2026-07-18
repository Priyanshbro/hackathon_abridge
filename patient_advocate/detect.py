from __future__ import annotations
import json
from dataclasses import dataclass

import chart
import clues

DISCOVER_MODEL = "claude-opus-4-8"


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


# transcript_clue and wearable_delta are deliberately ABSENT. They became
# prompt inputs (clues.py), not generators -- Python finds the cue, the model
# does the abductive reasoning. See clues.py for the measurement.
GENERATORS = [
    bp_control,
    a1c_threshold,
    cost_adherence_risk,
    coverage_awareness,
]

# The LLM classifies into this fixed vocabulary. Unconstrained, it produced 185
# distinct type names for 198 gaps ("abnormal_finding_not_addressed",
# "abnormal_lab_followup", "abnormal_creatinine_not_addressed" = one concept,
# three labels), which breaks dedup, aggregation and the eval.
GAP_TYPES = [
    "medication_condition_conflict",
    "duplicate_prescription",
    "inappropriate_medication_elderly",
    "missing_diagnostic_workup",
    "unmonitored_condition",
    "uninvestigated_symptom",
    "abnormal_result_not_addressed",
    "undertreated_condition",
    "no_end_organ_assessment",
    "cancer_screening_gap",
    "prenatal_supplementation_gap",
    "depression_screening_gap",
    "social_barrier_to_plan",
    "cost_adherence_risk",
    "specialist_referral_gap",
]

# types that are screening/preventive -> kind='preventive', which drives both
# the goals-of-care guard and the health-maintenance bucket in deliver.py
PREVENTIVE_TYPES = {
    "cancer_screening_gap",
    "depression_screening_gap",
    "prenatal_supplementation_gap",
}

DISCOVER_SYS = """You are reviewing one primary-care encounter for CARE GAPS:
clinically meaningful things that were missed, not done, or not followed up.

Report a gap only if you can point to specific evidence in the record provided.
Do NOT compute or estimate lab values, risk scores, or thresholds you were not
given -- those are handled deterministically elsewhere.

Include gaps a standard quality-measure checklist would MISS: an abnormal result
never followed up, a medication-condition conflict, an un-investigated likely
cause, a social or practical barrier that will stop the plan from happening.

Set verifiable_from_record=false when your claim depends on history this single
encounter does not contain -- prior screening, whether a lab was ever repeated.
Those are delivered to the patient as a question, not an assertion."""

DISCOVER_SCHEMA = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": GAP_TYPES},
                    "finding": {"type": "string"},
                    "evidence": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "moderate", "low"]},
                    "topic": {
                        "type": "string",
                        "description": "'|'-delimited keywords for resolution matching, "
                        "e.g. 'sleep|apnea|snoring'. Specific, not generic.",
                    },
                    "verifiable_from_record": {"type": "boolean"},
                },
                "required": [
                    "type", "finding", "evidence", "severity", "topic",
                    "verifiable_from_record",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["gaps"],
    "additionalProperties": False,
}

_SEVERITY_PRIORITY = {"high": 0.9, "moderate": 0.6, "low": 0.3}


def discover(rec: dict, client) -> list[Candidate]:
    """LLM-primary detection -- the breadth layer, and where the demo's best
    findings come from. Measured against the deterministic rules on all 25
    encounters: 198 gaps vs 45, zero encounters with nothing found vs three,
    and zero rejected as wrong in hand review.

    Clue elevation (clues.py) is appended to the prompt: deterministic
    extraction of the patient's own words and wearable deltas. Without it the
    OSA insight on Elias fired in 1 of 5 runs; with it, 2 of 2."""
    wearable = chart.wearable_for(rec["metadata"]["patient_id"]) or {}
    cue_block = clues.contributing_factors_block(rec["transcript"], wearable.get("days"))
    ctx = _detection_context(rec)
    response = client.messages.create(
        model=DISCOVER_MODEL,
        max_tokens=4000,
        system=DISCOVER_SYS + cue_block,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": DISCOVER_SCHEMA}},
        messages=[{"role": "user", "content": ctx}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    out = []
    for i, g in enumerate(json.loads(text)["gaps"]):
        out.append(
            Candidate(
                id=f"llm-{i}-{g['type']}",
                topic=g["topic"],
                kind="preventive" if g["type"] in PREVENTIVE_TYPES else "gap",
                trigger=g["finding"],
                evidence=[{"source": "llm_discovery", "field": g["type"], "value": g["evidence"]}],
                priority=_SEVERITY_PRIORITY.get(g["severity"], 0.5),
                verifiable_from_record=g["verifiable_from_record"],
            )
        )
    return out


def _detection_context(rec: dict) -> str:
    """Everything the model needs, in one payload. The whole record fits
    comfortably -- roughly 8-15k tokens against a 1M window -- so there is
    nothing to retrieve."""
    obs = []
    for o in chart.observations(rec):
        code = o.get("code", {})
        name = code.get("text") or (code.get("coding") or [{}])[0].get("display", "")
        if "valueQuantity" in o:
            v = o["valueQuantity"]
            obs.append(f"  {name} = {v.get('value')} {v.get('unit', '')}".rstrip())
        for comp in o.get("component", []):
            cn = (comp.get("code", {}).get("coding") or [{}])[0].get("display", "")
            if "valueQuantity" in comp:
                obs.append(f"  {cn} = {comp['valueQuantity'].get('value')}")
    new_rx = []
    for m in chart.medication_requests(rec):
        mc = m.get("medicationCodeableConcept", {})
        new_rx.append(mc.get("text") or (mc.get("coding") or [{}])[0].get("display", ""))
    return (
        f"ENCOUNTER: {chart.visit_title(rec)} ({chart.encounter_class(rec)}) "
        f"at {chart.service_provider(rec)}\n"
        f"ACTIVE CONDITIONS: {', '.join(chart.condition_labels(rec))}\n"
        f"CURRENT MEDICATIONS: {', '.join(chart.medication_labels(rec)) or 'none listed'}\n"
        f"PRESCRIBED AT THIS VISIT: {', '.join(x for x in new_rx if x) or 'none'}\n\n"
        f"OBSERVATIONS THIS VISIT:\n" + "\n".join(obs) + "\n\n"
        f"CLINICAL NOTE:\n{rec['note']}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{rec['transcript']}"
    )


def run_all(rec: dict, client=None) -> list[Candidate]:
    """Deterministic generators always; LLM discovery when a client is given.
    The LLM is the primary detector -- the generators cover exact computation
    and anchor the taxonomy. Without a client this returns the deterministic
    subset only, which is what the no-cost eval ablation uses."""
    out = [c for gen in GENERATORS for c in [gen(rec)] if c is not None]
    if client is not None:
        out.extend(discover(rec, client))
    return out
