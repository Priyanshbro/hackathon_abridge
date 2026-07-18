from __future__ import annotations

import re

from detect import Candidate
from stream import Utterance

# Default resolution vocabulary for clinical candidates (bp_control, a1c,
# wearable_delta, coverage_awareness, ...). A candidate can override this
# via Candidate.plan_verbs when "addressed" means something more specific
# than these generic clinical actions -- see cost_adherence_risk.
CLINICAL_PLAN_VERBS = (
    "start",
    "order",
    "refer",
    "referral",
    "recheck",
    "we'll",
    "we will",
    "let's get",
    "let us get",
    "i'm putting in",
    "i am putting in",
)

TOPIC_WINDOW = 6  # utterances to look back when checking "was this topic raised recently"
PARTIAL_DECAY = 0.5  # multiplicative priority decay when topic is raised without a plan
CLINICIAN_SPEAKERS = {"DR", "NURSE"}


def _keywords(topic: str) -> list[str]:
    return [k.strip().lower() for k in topic.split("|") if k.strip()]


def _contains_any(text: str, phrases) -> bool:
    text = text.lower()
    return any(re.search(r"\b" + re.escape(p) + r"\b", text) for p in phrases)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _plan_verb_near_topic(text: str, keywords, plan_verbs) -> bool:
    """True when a plan verb and one of this candidate's topic keywords
    share a SENTENCE. A multi-topic recap utterance ("Gums -- dental
    referral... Cholesterol -- diet and movement, recheck later. Migraines
    -- stable pattern...") puts unrelated topics in the same utterance but
    different sentences; same-utterance co-occurrence alone let "referral"
    (about dental) or "recheck" (about cholesterol) falsely resolve a
    migraine candidate whose own sentence assigned it no plan at all."""
    return any(
        _contains_any(sentence, plan_verbs) and _contains_any(sentence, keywords)
        for sentence in _sentences(text)
    )


def resolve(candidates: list[Candidate], utterance: Utterance, history: list[Utterance]) -> None:
    """Mutate candidates in place. Called once per utterance during replay;
    `history` is every utterance seen so far this visit, including the
    current one (the caller appends before calling). Deterministic, no LLM.

    addressed: the current utterance is from a clinician (DR/NURSE), contains
      a plan-verb for this candidate (candidate.plan_verbs, or the generic
      CLINICAL_PLAN_VERBS if unset), AND the candidate's topic was mentioned
      somewhere in the last TOPIC_WINDOW utterances (including now). This
      covers same-utterance plans ("blood pressure recheck") and split ones
      (diagnosis named one turn, referral stated the next).

    partially addressed: a CLINICIAN mentions the topic in the current
      utterance but the addressed test above didn't pass -- priority decays
      but the candidate stays open. This is Julius's reassurance case:
      "these are all cheap generics" mentions cost but is not a contingency
      plan, so it must not resolve the candidate.

      Decay requires a clinician for the same reason resolution does: only
      the care team can address a concern. This was previously any speaker,
      and 24% of all decay events (34 of 141 across the cohort) came from the
      patient or family. That inverts the product -- a patient restating a
      symptom is evidence the concern is LIVE, not that it was handled.
      It killed the hero case: Elias's OSA candidate carries "sofa bed" as a
      topic keyword (the same phrase clues.py elevates to raise OSA
      suspicion), so his three mentions of the sofa bed decayed it
      0.60 -> 0.075 and the K=3 budget cut it. The clue that should surface
      the question was burying it.

    untouched: no match, no change.
    """
    recent = history[-TOPIC_WINDOW:]
    # CLINICIAN-only, same reasoning as the partial-decay fix below: a
    # patient naming their own symptom ("the stuff I keep for when a
    # headache gets really bad") is evidence the concern is LIVE, not that
    # anyone addressed it. Scanning all speakers here let an unrelated
    # clinician plan-verb utterance ("we will go line by line" on the
    # medication list) falsely resolve a migraine candidate just because
    # the patient had said "headache" one turn earlier.
    recent_text = " ".join(u.text for u in recent if u.speaker in CLINICIAN_SPEAKERS)
    text = utterance.text
    is_clinician = utterance.speaker in CLINICIAN_SPEAKERS

    for c in candidates:
        if c.resolved_by is not None:
            continue  # already closed

        keywords = _keywords(c.topic)
        plan_verbs = c.plan_verbs or CLINICAL_PLAN_VERBS

        topic_recent = _contains_any(recent_text, keywords)
        topic_now = _contains_any(text, keywords)
        has_plan_verb = is_clinician and _contains_any(text, plan_verbs)

        if has_plan_verb and topic_now and not _plan_verb_near_topic(text, keywords, plan_verbs):
            has_plan_verb = False  # same utterance, different sentence -- not a plan for THIS topic

        if has_plan_verb and topic_recent:
            c.resolved_by = f"{utterance.idx}: {utterance.text}"
            c.priority = 0.0
        elif topic_now and is_clinician:
            c.priority *= PARTIAL_DECAY
