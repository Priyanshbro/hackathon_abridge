from __future__ import annotations

import json
from dataclasses import dataclass

import chart
from detect import Candidate

MODEL = "claude-sonnet-5"
# Phrasing is templated rewriting against a fixed schema, not a judgment
# call -- Gate B already did the hard reasoning. Haiku is faster and
# cheaper here with no quality loss for the task shape.
PHRASE_MODEL = "claude-haiku-4-5"
K_BUDGET = 3

# Hospice and SNF are NOT the same context and must not share a guard.
# Hospice is comfort-focused: preventive and referral candidates are dropped
# outright. A SNF admission is rehabilitative -- mood screening is routine
# there (PHQ-9 on admission), while cancer screening is not. So SNF gets a
# narrowed maintenance scope in split_buckets() rather than a hard drop here.
HOSPICE_KEYWORDS = ("hospice",)
SNF_KEYWORDS = ("skilled nursing", "snf", "nursing facility", "nursing home")
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


def _context_text(rec: dict) -> str:
    return (chart.visit_title(rec) + " " + chart.service_provider(rec)).lower()


def is_goals_of_care_context(rec: dict) -> bool:
    """Hospice admission -- deterministic gate on Encounter.class +
    serviceProvider/visit_title. Cheap, explainable, checked first.
    SNF deliberately excluded: see the keyword constants above."""
    if chart.encounter_class(rec) == "HH":
        return True
    return any(k in _context_text(rec) for k in HOSPICE_KEYWORDS)


def is_snf_context(rec: dict) -> bool:
    return any(k in _context_text(rec) for k in SNF_KEYWORDS)


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
        # thinking produced run-to-run flip-flops on close cases.
        # effort="low" caps how long that deliberation runs for latency --
        # Sonnet 5 has no budget_tokens (removed on this model family, 400s
        # if sent), so effort is the only depth lever available. This is a
        # real tradeoff against the flip-flop fix: if grounding inconsistency
        # reappears, raise effort before reaching for anything else.
        thinking={"type": "adaptive"},
        output_config={
            "format": {"type": "json_schema", "schema": GROUND_SCHEMA},
            "effort": "low",
        },
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    results = {r["id"]: r["grounded"] for r in json.loads(text)["results"]}
    return [c for c in survivors if results.get(c.id, False)]


def budget(survivors: list[Candidate], k: int = K_BUDGET) -> list[Candidate]:
    """Gate C: hard cap at k, strict priority ordering.

    Ties break on BASE severity, then id. Live priority is base severity times
    a decay applied once per clinician mention, so two candidates of different
    original severity land on the same number routinely (0.30 undecayed ties
    0.60-decayed-once). `sorted` is stable, so those ties were being broken by
    list order -- i.e. by the detector's output sequence, which is arbitrary.
    That is not acceptable at the K boundary, where a tie decides whether a
    real question reaches the patient: a recurrent-UTI candidate lost its slot
    to an exactly-equal chest-symptom candidate purely on position.

    Preferring higher base severity means a serious concern that was discussed
    once still outranks a minor one that was never raised."""
    return sorted(
        survivors,
        key=lambda c: (c.priority, c.base_priority or c.priority, c.id),
        reverse=True,
    )[:k]


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
        "copay, or cost; a doctor cannot answer that. But do NOT default to "
        "a vague 'what options exist' question either. When the evidence "
        "names new prescriptions alongside a cost/affordability barrier, "
        "the doctor CAN and should answer a concrete prescribing question: "
        "a cheaper generic, a different equally-effective medication, "
        "samples, or a manufacturer/pharmacy assistance program they'd "
        "recommend. Ground the specific ask in what the evidence actually "
        "names (e.g. the medications listed) rather than asking generically "
        "'what options exist.'\n"
        "ONE SPECIFIC ASK, NOT A LIST. Never enumerate unrelated organ "
        "systems or tests -- 'should we check my heart, kidneys and "
        "cholesterol?' is three questions in a trench coat and a doctor "
        "cannot answer it. Pick the SINGLE thing that today's plan makes "
        "most decision-relevant, and name it. Relevance comes from what is "
        "happening in THIS visit, not from what matters most in general: "
        "when three blood pressure drugs including an ACE inhibitor and a "
        "diuretic are being started, kidney function and potassium are the "
        "pressing check even though cholesterol is the bigger long-run "
        "risk. Naming two things together is fine ONLY when they are the "
        "same concept or one panel a patient would think of as one thing "
        "('cholesterol and blood sugar'); never bundle items that are "
        "unrelated to each other.\n"
        "PATIENT WORDS, NOT LAB CODES. Say what the patient would say: "
        "'blood sugar', not 'A1c' or 'HbA1c'; 'kidney numbers', not "
        "'creatinine' or 'eGFR'; 'blood count', not 'CBC'. Naming the test "
        "the clinic orders is the same mistake as asking for a PHQ-9 -- the "
        "patient describes what they want to know, the clinician picks the "
        "test.\n"
        "ASK 'WHEN' OR 'HOW', NOT 'WHETHER', once the clinical situation "
        "already settles that something is needed. If a monitoring test is "
        "clearly indicated by the treatment just started, the patient's "
        "useful question is the timing or the threshold -- 'when should my "
        "kidney numbers and potassium be checked after starting these?' -- "
        "not 'should we check my kidneys?', which invites a yes and ends "
        "there.\n"
        "- scope: 'in_scope' if the clinician in front of the patient can "
        "act on this directly, 'referral' if it genuinely needs a "
        "specialist, 'barrier' if it is a cost/access/social obstacle.\n"
        "MENTAL-HEALTH ITEMS: never have the patient request a screening "
        "instrument. No patient asks 'can we do a PHQ-9' or 'should I "
        "complete a depression questionnaire' -- that is the clinic's "
        "language, and this is the patient's question. Ask instead what "
        "SUPPORT OR RESOURCES they should be considering, anchored to what "
        "they actually said about their own life: 'I've been struggling "
        "since my sister died -- what mental health support should we be "
        "thinking about?' Concrete about the next step, never a vague "
        "'should we look at this'. The clinician picks the instrument.\n"
        "- context: an optional one-line patient-facing note, only when the "
        "evidence includes a coverage_271 entry (copay, payer, referral "
        "requirement) -- state it plainly for the patient's own reference. "
        "null if there is no such evidence. Never fold this into the "
        "question.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    response = client.messages.create(
        model=PHRASE_MODEL,
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


HEALTH_MAINTENANCE_BUDGET = 2

# health maintenance is PCP scope. A psychiatrist is not going to action a
# mammogram, so surfacing it at a specialist visit is noise.
#
# The rule is an EXCLUSION list, not an inclusion list. Enumerating specialty
# markers is a short, stable vocabulary; enumerating every way a chart writes
# "primary care" is not -- the previous inclusion list ("primary care",
# "family", "general", "wellness", ...) silently dropped 13 of 32 preventive
# candidates, classifying "Annual physical -- geriatric cardiometabolic" as
# specialist while "Annual physical -- new adult wellness exam" passed, purely
# on incidental wording in a free-text title. An ambulatory visit is treated as
# primary care unless something says otherwise.
SPECIALTY_MARKERS = (
    "cardiology", "oncology", "psychiatry", "dermatology", "orthopedic",
    "ophthalmology", "physical therapy", "podiatry", "rheumatology",
    "gastroenterology", "pulmonology", "neurology", "urology", "nephrology",
)

# Prenatal care is primary-care adjacent -- the obstetric team carries general
# health maintenance for a pregnant patient, so the bucket stays open.
PRENATAL_MARKERS = ("prenatal", "obstetric", "pregnancy", "antepartum")

# Dental is its own scope: a dentist carries health maintenance for DENTAL
# problems only. The bucket is not suppressed, it is narrowed -- surfacing a
# mammogram at a dental visit is noise, surfacing a periodontal recall is not.
DENTAL_MARKERS = ("dental", "dentist", "oral surgery", "gingival", "periodont")
DENTAL_TOPIC_MARKERS = ("dental", "oral", "tooth", "teeth", "gingiv", "periodont", "caries")

# A SNF admission carries mood/psychosocial screening -- PHQ-9 on admission is
# routine, and bereavement/isolation are live issues in that setting. It does
# not carry cancer screening.
SNF_TOPIC_MARKERS = (
    "depression", "phq", "mood", "anxiety", "bereavement", "isolation",
    "psychosocial", "mental health", "cognitive", "delirium",
)


def maintenance_scope(rec: dict) -> str:
    """Which health-maintenance items this encounter can carry:
    'all' (primary care or prenatal), 'dental', 'snf', or 'none'.

    Same care-context triple the scope routing uses: Encounter.class,
    visit_title, serviceProvider. No Practitioner resource exists in the
    bundle, so there is no specialty lookup -- this triple is the signal."""
    if is_snf_context(rec):
        return "snf"
    if chart.encounter_class(rec) != "AMB":
        return "none"
    text = _context_text(rec)
    if any(k in text for k in PRENATAL_MARKERS):
        return "all"
    if any(k in text for k in DENTAL_MARKERS):
        return "dental"
    if any(k in text for k in SPECIALTY_MARKERS):
        return "none"
    return "all"


def is_primary_care(rec: dict) -> bool:
    """Kept as the boolean form for callers that only need 'does this
    encounter carry general health maintenance'."""
    return maintenance_scope(rec) == "all"


def split_buckets(candidates: list[Candidate], rec: dict) -> tuple[list[Candidate], list[Candidate]]:
    """Visit questions vs health maintenance. Screening/preventive items are
    a different category from questions arising out of what happened today,
    and they must not compete with them for the K=3 budget -- they render as
    their own shorter list. Scoped by maintenance_scope(): everything at a
    primary-care or prenatal visit, dental items only at a dental visit,
    mood/psychosocial items only at a SNF admission, nothing at a specialist,
    hospice, or other inpatient encounter."""
    visit = [c for c in candidates if c.kind != "preventive"]
    preventive = [c for c in candidates if c.kind == "preventive"]

    scope = maintenance_scope(rec)
    topic_filter = {"dental": DENTAL_TOPIC_MARKERS, "snf": SNF_TOPIC_MARKERS}.get(scope)
    if scope == "none":
        maintenance = []
    elif topic_filter:
        maintenance = [
            c for c in preventive
            if any(k in (c.topic + " " + c.trigger).lower() for k in topic_filter)
        ]
    else:
        maintenance = preventive
    return visit, maintenance


def deliver(client, candidates: list[Candidate], rec: dict) -> dict:
    """Returns two lists. `visit` is the top-K questions from today's
    encounter; `health_maintenance` is the PCP-scoped screening list, capped
    separately and empty for specialist/hospice/SNF encounters."""
    candidates = goals_of_care_guard(candidates, rec)
    survivors = filter_survivors(candidates)
    survivors = ground(client, survivors)

    visit_c, maint_c = split_buckets(survivors, rec)
    visit_budgeted = budget(visit_c, K_BUDGET)
    maint_budgeted = budget(maint_c, HEALTH_MAINTENANCE_BUDGET)

    # one phrase() call for both buckets, not two -- which of the two lists
    # an id came from is already known locally, no need for the model to
    # tell us
    all_items = phrase(client, visit_budgeted + maint_budgeted)
    visit_ids = {c.id for c in visit_budgeted}
    visit = [it for it in all_items if it.id in visit_ids]
    maintenance = [it for it in all_items if it.id not in visit_ids]
    return {"visit": visit, "health_maintenance": maintenance}
