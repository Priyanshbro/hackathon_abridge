from __future__ import annotations
from detect import Candidate
from stream import Utterance

PLAN_VERBS = ("start", "order", "refer", "recheck", "we'll", "i'm putting in", "let's get")


def resolve(candidates: list[Candidate], utterance: Utterance) -> None:
    """Mutate candidates in place, one utterance at a time (deterministic,
    no LLM). For each open candidate whose topic keyword appears in
    utterance.text:
      - if a plan-verb is also present -> addressed: set
        resolved_by = f"{utterance.idx}: {utterance.text}", full priority drop
      - if topic mentioned but no plan-verb -> partially addressed:
        leave resolved_by unset, drop priority (this is the
        cost_adherence_risk / coverage_awareness case -- reassurance
        without a plan must NOT count as resolved)
      - otherwise -> untouched, no change
    """
    raise NotImplementedError
