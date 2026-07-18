from __future__ import annotations

import json

from detect import Candidate

MODEL = "claude-sonnet-5"
K_BUDGET = 3

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
                    "evidence": {"type": "string"},
                },
                "required": ["id", "question", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


def filter_survivors(candidates: list[Candidate]) -> list[Candidate]:
    """Drop anything resolve.py already closed."""
    return [c for c in candidates if c.resolved_by is None]


def ground(client, survivors: list[Candidate]) -> list[Candidate]:
    """Gate B: one batched call over all survivors. Each candidate must
    cite a specific datum from its own evidence list or it gets dropped.
    Deterministic, no thinking -- this is a check, not open-ended
    reasoning."""
    if not survivors:
        return []

    payload = [
        {"id": c.id, "trigger": c.trigger, "evidence": c.evidence}
        for c in survivors
    ]
    prompt = (
        "For each candidate below, decide whether its evidence genuinely "
        "supports asking the patient about it -- the evidence must cite a "
        "specific datum (a value, a diagnosis, a coverage fact), not a vague "
        "trigger with nothing to point to. Mark grounded=false for anything "
        "that doesn't hold up.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=1,
        thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": GROUND_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    results = {r["id"]: r["grounded"] for r in json.loads(text)["results"]}
    return [c for c in survivors if results.get(c.id, False)]


def budget(survivors: list[Candidate], k: int = K_BUDGET) -> list[Candidate]:
    """Gate C: hard cap at k, strict priority ordering."""
    return sorted(survivors, key=lambda c: c.priority, reverse=True)[:k]


def phrase(client, survivors: list[Candidate]) -> list[dict]:
    """One call. Patient's voice, first person, <=25 words each, each
    carrying its evidence line. Returns [{"id", "question", "evidence"}, ...]
    in the same order as `survivors`."""
    if not survivors:
        return []

    payload = [
        {"id": c.id, "trigger": c.trigger, "evidence": c.evidence}
        for c in survivors
    ]
    prompt = (
        "Turn each candidate below into a question the patient could ask "
        "their own doctor, in the patient's voice, first person, plain "
        "language, at most 25 words. Then write a one-line evidence note "
        "explaining what grounds the question (cite the specific datum).\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=1,
        thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": PHRASE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    by_id = {q["id"]: q for q in json.loads(text)["questions"]}
    return [by_id[c.id] for c in survivors if c.id in by_id]


def deliver(client, candidates: list[Candidate]) -> list[dict]:
    survivors = filter_survivors(candidates)
    survivors = ground(client, survivors)
    survivors = budget(survivors)
    return phrase(client, survivors)
