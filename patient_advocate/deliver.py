from __future__ import annotations
from detect import Candidate

K_BUDGET = 3


def filter_survivors(candidates: list[Candidate]) -> list[Candidate]:
    """Drop anything with resolved_by set."""
    raise NotImplementedError


def ground(client, survivors: list[Candidate]) -> list[Candidate]:
    """Gate B: one batched call over all survivors (claude-opus-4-8,
    output_config={"effort": "low"}, thinking={"type": "disabled"},
    json_schema output). Each candidate must cite a specific datum from
    its own evidence list or it gets dropped."""
    raise NotImplementedError


def budget(survivors: list[Candidate], k: int = K_BUDGET) -> list[Candidate]:
    """Gate C: hard cap at k, strict priority ordering."""
    raise NotImplementedError


def phrase(client, survivors: list[Candidate]) -> list[dict]:
    """One call, adaptive thinking. Patient's voice, first person,
    <=25 words each, each carrying its evidence line. Returns
    [{"question": str, "evidence": str}, ...]."""
    raise NotImplementedError


def deliver(client, candidates: list[Candidate]) -> list[dict]:
    survivors = filter_survivors(candidates)
    survivors = ground(client, survivors)
    survivors = budget(survivors)
    return phrase(client, survivors)
