from __future__ import annotations

import json
from dataclasses import dataclass

import chart
from detect import Candidate

MODEL = "claude-sonnet-5"
K_BUDGET = 3

HOSPICE_SNF_KEYWORDS = ("hospice", "skilled nursing", "snf", "nursing facility", "nursing home")
GOALS_OF_CARE_KINDS = {"referral", "preventive"}
GOALS_OF_CARE_TRIGGER_KEYWORDS = ("screening", "specialist", "referral")


@dataclass
class DeliveredItem:
    id: str
    question: str  # spoken to the clinician -- clinical only, never coverage/cost
    evidence: list[dict]
    scope: str  # in_scope | referral | barrier
    context: str | None = None  # patient-facing subtext (e.g. coverage) -- never asked aloud


GROUND_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "grounded": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "grounded", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

PHRASE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "question": {"type": "string"},
                    "scope": {"type": "string", "enum": ["in_scope", "referral", "barrier"]},
                    "context": {"type": ["string", "null"]},
                },
                "required": ["id", "question", "scope", "context"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


def is_goals_of_care_context(rec: dict) -> bool:
    """Hospice or SNF admission -- deterministic gate on Encounter.class +
    serviceProvider/visit_title. Cheap, explainable, checked first."""
    if chart.encounter_class(rec) == "HH":
        return True
    text = (chart.visit_title(rec) + " " + chart.service_provider(rec)).lower()
    return any(k in text for k in HOSPICE_SNF_KEYWORDS)


def goals_of_care_guard(candidates: list[Candidate], rec: dict) -> list[Candidate]:
    """Step 0. Drop preventive/referral candidates outright for hospice/SNF
    patients -- suggesting cancer screening to an end-stage hospice patient
    is the output that sinks a demo in front of clinicians. Not restricted
    to candidates explicitly tagged kind='referral'/'preventive': also
    catches anything whose trigger text reads as a screening or specialist
    suggestion, since not every generator will tag kind consistently."""
    if not is_goals_of_care_context(rec):
        return candidates

    def is_preventive_or_referral(c: Candidate) -> bool:
        if c.kind in GOALS_OF_CARE_KINDS:
            return True
        return any(k in c.trigger.lower() for k in GOALS_OF_CARE_TRIGGER_KEYWORDS)

    return [c for c in candidates if not is_preventive_or_referral(c)]


def filter_survivors(candidates: list[Candidate]) -> list[Candidate]:
    """Drop anything resolve.py already closed."""
    return [c for c in candidates if c.resolved_by is None]


def ground(client, survivors: list[Candidate]) -> list[Candidate]:
    """Gate B: one batched call over all survivors. Each candidate must
    cite a specific datum from its own evidence list or it gets dropped.
    Deterministic in spirit -- thinking disabled, this is a check, not
    open-ended reasoning."""
    if not survivors:
        return []

    payload = [{"id": c.id, "trigger": c.trigger, "evidence": c.evidence} for c in survivors]
    prompt = (
        "Each candidate below was proposed by a deterministic rule, not by "
        "you -- your only job is to check whether its evidence array backs "
        "up its trigger with real, specific data (a value, a diagnosis, a "
        "coverage fact, a computed boolean), not whether the underlying "
        "clinical or eligibility reasoning is one you'd have made yourself. "
        "Mark grounded=false only when the evidence is empty, vague, or "
        "doesn't actually contain the fact the trigger claims -- not because "
        "you'd want additional data the rule didn't have access to.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=1,
        # adaptive, not disabled -- this is a borderline judgment call
        # (does the evidence really support the trigger), and disabled
        # thinking produced run-to-run flip-flops on close cases
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": GROUND_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    results = {r["id"]: r["grounded"] for r in json.loads(text)["results"]}
    return [c for c in survivors if results.get(c.id, False)]


def budget(survivors: list[Candidate], k: int = K_BUDGET) -> list[Candidate]:
    """Gate C: hard cap at k, strict priority ordering."""
    return sorted(survivors, key=lambda c: c.priority, reverse=True)[:k]


def phrase(client, survivors: list[Candidate]) -> list[DeliveredItem]:
    """One call. Patient's voice, first person, <=25 words, purely
    clinical -- a doctor cannot answer "is this covered," so coverage/cost
    never becomes part of the spoken question. Also does scope routing
    (in_scope | referral | barrier) in the same call -- no extra cost.
    Returns DeliveredItem list in survivor order."""
    if not survivors:
        return []

    payload = [{"id": c.id, "trigger": c.trigger, "evidence": c.evidence} for c in survivors]
    prompt = (
        "For each candidate below, produce:\n"
        "- question: what the patient could ask their own doctor, in the "
        "patient's voice, first person, plain language, at most 25 words. "
        "This must be a purely clinical question the doctor can actually "
        "answer -- never ask the doctor to confirm insurance coverage, "
        "copay, or cost; a doctor cannot answer that.\n"
        "- scope: 'in_scope' if the clinician in front of the patient can "
        "act on this directly, 'referral' if it genuinely needs a "
        "specialist, 'barrier' if it is a cost/access/social obstacle.\n"
        "- context: an optional one-line patient-facing note, only when the "
        "evidence includes a coverage_271 entry (copay, payer, referral "
        "requirement) -- state it plainly for the patient's own reference. "
        "null if there is no such evidence. Never fold this into the "
        "question.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1536,
        temperature=1,
        thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": PHRASE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    by_id = {q["id"]: q for q in json.loads(text)["questions"]}

    items = []
    for c in survivors:
        q = by_id.get(c.id)
        if q is None:
            continue
        items.append(
            DeliveredItem(
                id=c.id,
                question=q["question"],
                evidence=c.evidence,
                scope=q["scope"],
                context=q.get("context"),
            )
        )
    return items


def deliver(client, candidates: list[Candidate], rec: dict) -> list[DeliveredItem]:
    candidates = goals_of_care_guard(candidates, rec)
    survivors = filter_survivors(candidates)
    survivors = ground(client, survivors)
    survivors = budget(survivors)
    return phrase(client, survivors)
