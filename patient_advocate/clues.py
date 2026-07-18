"""Deterministic clue elevation -- the patient's own words, surfaced for the model.

MEASURED, and the reason this module exists rather than a `transcript_clue`
generator. Opus 4.8 reading Elias's full record produced the OSA insight
("morning occipital headaches ... without considering obstructive sleep apnea")
in 1 of 5 unprompted runs. Multi-sampling did not rescue it -- 3 further runs
missed it and the union simply grew to 22 gap types, i.e. more noise for a
3-question product.

Deterministically extracting the cue and injecting it as a CONTRIBUTING FACTORS
block hit 2 of 2. The reasoning stays with the model; only the guarantee that
the cue is in front of it is deterministic. This is the ddx clue-elevation
mechanism -- Python finds "could it be the running?", the model decides whether
it explains anything.

A `transcript_clue` Candidate was rejected instead: "life-context cue: 'sofa
bed'" is a weak thing to deliver to a patient, and it discards the abductive
step that makes the insight worth surfacing at all.
"""
from __future__ import annotations

import re

# phrase -> short label shown to the model. Keep these SPECIFIC: broadening to
# bare "sleep" or "work" matches everywhere and elevates nothing.
LIFESTYLE_CUES: tuple[tuple[str, str], ...] = (
    ("sofa bed", "sleeping arrangement changed"),
    ("couch", "sleeping arrangement changed"),
    ("night shift", "shift work"),
    ("newborn", "newborn at home"),
    ("ran out of", "medication lapse"),
    ("stopped refilling", "medication lapse"),
    ("laid off", "income change"),
    ("lost my job", "income change"),
    ("left it on the counter", "cost-driven abandonment"),
)

# wearable deltas worth elevating, as (field, direction, threshold, label)
WEARABLE_SIGNALS: tuple[tuple[str, str, float, str], ...] = (
    ("sleep_hours", "down", 1.0, "sleep duration fell"),
    ("awakenings", "up", 1.0, "night awakenings rose"),
    ("resting_hr", "up", 3.0, "resting heart rate rose"),
    ("steps", "down", 1500.0, "activity fell"),
)


def transcript_cues(transcript: str) -> list[str]:
    """Lifestyle/exposure cues in the patient's own words."""
    text = transcript.lower()
    out, seen = [], set()
    for phrase, label in LIFESTYLE_CUES:
        if phrase in text and label not in seen:
            seen.add(label)
            out.append(f"patient mentioned '{phrase}' ({label})")
    return out


def wearable_cues(series: list[dict] | None, window: int = 40) -> list[str]:
    """Compare the first and last `window` days. Wearable data is the
    patient's own context that never reaches the chart -- as a clue it is far
    more useful than as a threshold generator."""
    if not series or len(series) < window * 2:
        return []
    early, late = series[:window], series[-window:]

    def mean(rows, key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else None

    out = []
    for field, direction, threshold, label in WEARABLE_SIGNALS:
        a, b = mean(early, field), mean(late, field)
        if a is None or b is None:
            continue
        delta = b - a
        moved = (delta <= -threshold) if direction == "down" else (delta >= threshold)
        if moved:
            out.append(f"wearable: {label} ({a:.1f} -> {b:.1f} over 90 days)")
    return out


def contributing_factors_block(transcript: str, wearable: list[dict] | None = None) -> str:
    """The block appended to the detection prompt. Empty string when there is
    nothing to elevate -- never inject a heading with no content under it."""
    cues = transcript_cues(transcript) + wearable_cues(wearable)
    if not cues:
        return ""
    lines = "\n".join(f"  - {c}" for c in cues)
    return (
        "\n\nCONTRIBUTING FACTORS the patient raised in their own words, or that "
        "their own device recorded. Consider whether any could be a cause or "
        "contributor to a finding, and whether that possibility was "
        f"investigated:\n{lines}"
    )
