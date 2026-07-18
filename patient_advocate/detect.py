from __future__ import annotations
import dataclasses
import json
import pathlib
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
    base_priority: float | None = None
    # priority BEFORE any conversational decay. `priority` is multiplied by
    # PARTIAL_DECAY on each clinician mention, so a high-severity item raised
    # once lands exactly on a low-severity item never raised (0.9*0.5 vs 0.3
    # is close; 0.6*0.5 == 0.3 exactly). deliver.budget() breaks those ties on
    # this field so a serious concern that got discussed still outranks a
    # minor one that did not -- previously the tie fell to list order.
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
                base_priority=0.5,
            )
    return None


# DELETED, deliberately -- bp_control, a1c_threshold, cost_adherence_risk,
# transcript_clue, wearable_delta.
#
# transcript_clue / wearable_delta became prompt inputs (clues.py): Python
# finds the cue, the model does the abductive reasoning.
#
# The other three were dropped once the LLM proved it covers them:
#   - cost_adherence_risk fired on Julius in 4 of 4 runs, so the
#     reliability argument for a deterministic version is gone.
#   - bp_control / a1c_threshold would emit findings that are not gaps. On
#     Elias the model reported no_end_organ_assessment rather than "BP is
#     141/100" -- correctly, because the doctor had already diagnosed and
#     treated it. A deterministic rule would push that non-gap into the K=3
#     budget.
#   - The "they anchor the taxonomy" argument died when GAP_TYPES became a
#     fixed list independent of the generators.
#
# A deterministic generator now has to earn its place with computation the
# model genuinely cannot do (eGFR arithmetic against a cutoff, say). None of
# the deleted three qualified.
GENERATORS = [
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
    # vaccines had no type of their own, so they landed in the nearest bucket:
    # "should I also get the pneumococcal and shingles vaccines?" was filed as
    # cancer_screening_gap.
    "immunization_gap",
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
    "immunization_gap",
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
Those are delivered to the patient as a question, not an assertion.

CITE WHAT YOU USED. Every gap needs a citations array pointing at the specific
data you relied on, not a paraphrase of it. For a lab or vital, give the
observation name and its exact value. For something said in the room, quote the
speaker's own words. A downstream check verifies each gap against its citations
and DROPS gaps whose citations do not contain the fact the finding claims -- so
if your finding names a number, a citation must carry that number."""

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
                    # Structured citations, added because Gate B could not
                    # verify anything without them: every LLM candidate used
                    # to carry ONE prose evidence item, so the gate could only
                    # check that the prose sounded specific. It dropped a real
                    # platelet finding (436.75, a genuine Observation in the
                    # record) because the candidate paraphrased the transcript
                    # instead of citing the value the finding named.
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "enum": ["observation", "condition", "medication",
                                             "transcript", "coverage", "longitudinal_summary"],
                                },
                                "field": {
                                    "type": "string",
                                    "description": "Observation/condition/medication name, or "
                                    "the speaker (DR/PT/NURSE) for a transcript quote.",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "The exact value or verbatim quote. Never a "
                                    "paraphrase. If the finding names a number, put that number here.",
                                },
                            },
                            "required": ["source", "field", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "severity": {"type": "string", "enum": ["high", "moderate", "low"]},
                    "topic": {
                        "type": "string",
                        "description": "'|'-delimited keywords for resolution matching, "
                        "e.g. 'sleep|apnea|snoring'. Specific, not generic.",
                    },
                    "verifiable_from_record": {"type": "boolean"},
                },
                "required": [
                    "type", "finding", "evidence", "citations", "severity", "topic",
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

# barrier-type gaps need a resolution vocabulary scoped to actually solving
# the barrier, not generic clinical action verbs. Sending a referral or
# starting a medication doesn't address whether the patient can afford or
# attend it -- confirmed live on Elias, where an LLM-found cost_adherence_risk
# candidate (topic included "dental") false-resolved off "I'm sending a
# dental referral today" purely because "referral" is a generic plan-verb and
# "dental" happened to be one of its own topic keywords. Same trap as
# Julius's cost case, just unprotected because discover() never set
# plan_verbs on LLM-found candidates.
BARRIER_TYPES = {"cost_adherence_risk", "social_barrier_to_plan"}
BARRIER_PLAN_VERBS = (
    "call back",
    "call us",
    "assistance program",
    "sliding scale",
    "90-day supply",
    "ninety-day supply",
    "price check",
    "generic substitute",
    "social work",
    "case manager",
    "resource",
)


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
    # effort="low": this call alone was ~43s at the (implicit) default of
    # "high" -- nearly the entire end-to-end latency. Tested low vs medium
    # vs high on Elias: gap count 8/9/9, OSA insight found at all three.
    # That's one trial per setting against inherently stochastic discovery
    # (this doc's own note above: OSA fired 1-of-5 unprompted before clue
    # elevation existed), not the same rigor as the original 25-encounter
    # measurement -- if delivered gap quality/breadth regresses, raise this
    # before anything else, and ideally re-run the full comparison this was
    # measured against.
    response = client.messages.create(
        model=DISCOVER_MODEL,
        # 4000 truncated mid-JSON once citations became required: each gap now
        # carries 1-3 {source, field, value} objects with verbatim quotes, and
        # a 6-candidate encounter blew the limit on the FIRST record. A
        # truncated response is not a soft failure -- json.loads raises and the
        # whole encounter is lost.
        max_tokens=8000,
        system=DISCOVER_SYS + cue_block,
        thinking={"type": "adaptive"},
        output_config={
            "format": {"type": "json_schema", "schema": DISCOVER_SCHEMA},
            "effort": "low",
        },
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
                # citations first -- Gate B reads the evidence array in order,
                # and the chart-grounded items are what it can actually verify.
                # The prose summary is kept last for the phrasing call's context.
                evidence=g["citations"]
                + [{"source": "llm_discovery", "field": g["type"], "value": g["evidence"]}],
                priority=_SEVERITY_PRIORITY.get(g["severity"], 0.5),
                base_priority=_SEVERITY_PRIORITY.get(g["severity"], 0.5),
                verifiable_from_record=g["verifiable_from_record"],
                plan_verbs=BARRIER_PLAN_VERBS if g["type"] in BARRIER_TYPES else None,
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


CACHE_PATH = pathlib.Path(__file__).parent.parent / "data" / "detection_cache.json"

# The committed cache is a TRIAL ARTIFACT for this round of work, not a build
# input. It exists so the deterministic eval (ablation, invariants, silence
# control) runs at zero cost against a fixed candidate set, and so repeat trial
# runs don't re-spend on detection. It is NOT the demo path -- ui.run_live
# passes use_cache=False and detects live.
#
# Treat it as disposable: after any change to DISCOVER_SYS, GAP_TYPES, the
# clue-elevation block, or the model, it is stale and must be deleted and
# rewarmed. Nothing detects that automatically, so a stale cache will silently
# serve old candidates.


def _cache_load() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _cache_store(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))


def _from_cache(d: dict) -> Candidate:
    """JSON has no tuples. plan_verbs round-trips as a list, and resolve.py
    compares it against its own tuple constants, so restore the type here
    rather than leaving two shapes of the same field in circulation."""
    d = dict(d)
    if d.get("plan_verbs") is not None:
        d["plan_verbs"] = tuple(d["plan_verbs"])
    return Candidate(**d)


def run_all(rec: dict, client=None, use_cache: bool = True) -> list[Candidate]:
    """Deterministic generators always; LLM discovery when a client is given.

    Detection is CACHED per patient, for two reasons. First, it is a pre-visit
    step -- the agent reads the chart before the patient walks in, so running
    it once and reusing it is architecturally correct, not a shortcut. Second,
    the eval depends on it: LLM detection varies run to run (the OSA insight
    on Elias fired in 1 of 5 unprompted runs), so a suppression ON/OFF
    ablation over freshly-detected candidates would measure model variance
    instead of the gates. Both arms must see an identical candidate set.

    Pass use_cache=False to force a fresh detection.

    The cache is consulted BEFORE the client is required. A warmed cache is
    the whole reason the deterministic eval (ablation, invariants, the silence
    control) can run at zero cost with client=None -- checking `client` first
    would make those paths see only the deterministic generators and report
    an empty candidate set.
    """
    out = [c for gen in GENERATORS for c in [gen(rec)] if c is not None]

    pid = rec["metadata"]["patient_id"]
    cache = _cache_load() if use_cache else {}
    if pid in cache:
        out.extend(_from_cache(c) for c in cache[pid])
        return out
    if client is None:
        return out

    discovered = discover(rec, client)
    out.extend(discovered)
    if use_cache:
        cache = _cache_load()
        cache[pid] = [dataclasses.asdict(c) for c in discovered]
        _cache_store(cache)
    return out
