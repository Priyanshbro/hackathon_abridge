"""Generate synthetic wearable data for the Abridge hackathon cohort.

FULLY SYNTHETIC. Nothing here is real patient data. Each patient's series is
derived deterministically (seeded by patient id) from their own FHIR chart --
age, BMI, conditions -- so the wearable agrees with the clinical record instead
of being random noise.

The point of this data: it is the patient's OWN context, which never reaches
the chart. That is the raw material for patient-side abductive questions.

Output: data/wearable.json  -> {patient_id: {"days": [...], "notes": [...]}}
"""
import json
import math
import random
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "synthetic-ambient-fhir-25" / "synthetic-ambient-fhir-25.jsonl"
OUT = ROOT / "data" / "wearable.json"
DAYS = 90


def _obs(rec, needle):
    """Latest numeric observation whose display name contains `needle`."""
    for o in rec["encounter_fhir"]["related_resources"].get("Observation", []):
        code = o.get("code", {})
        name = code.get("text") or (code.get("coding") or [{}])[0].get("display", "")
        if needle.lower() in name.lower() and "valueQuantity" in o:
            return o["valueQuantity"].get("value")
    return None


def _labels(rec):
    return " | ".join(rec["patient_context"]["longitudinal_summary"]["condition_labels"]).lower()


def _age(rec, on):
    b = rec["patient_context"]["patient"]["birthDate"]
    by, bm, bd = (int(x) for x in b.split("-"))
    return on.year - by - ((on.month, on.day) < (bm, bd))


def build(rec):
    """One patient's 90-day series, ending on the encounter date."""
    pid = rec["metadata"]["patient_id"]
    rng = random.Random(pid)                       # deterministic per patient
    end = date.fromisoformat(rec["metadata"]["date"][:10])
    labels = _labels(rec)
    bmi = _obs(rec, "body mass index") or 26.0
    age = _age(rec, end)
    transcript = rec["transcript"].lower()

    # --- baselines conditioned on the chart -------------------------------
    steps = 7800 - 45 * max(0, age - 40) - 90 * max(0, bmi - 25)
    if "obesity" in labels:              steps -= 700
    if "osteoarthritis" in labels or "chronic pain" in labels: steps -= 900
    if "hospice" in labels or "palliative" in labels:          steps = min(steps, 900)
    steps = max(600, steps)

    rhr = 58 + 0.18 * max(0, bmi - 22) * 3 + 0.05 * age
    if "hypertension" in labels: rhr += 3
    if "anemia" in labels:       rhr += 4

    sleep = 7.2 - (0.4 if "stress" in labels else 0) - (0.5 if "anxiety" in labels else 0)
    hypertensive = "hypertension" in labels

    # --- a real-life disruption the CHART never records -------------------
    # Detected from the patient's own words, not hand-coded per patient.
    #
    # Two KINDS, because the downstream cue extractor watches four signals
    # (sleep_hours, awakenings, resting_hr, steps) and a sleep-only event can
    # never move the fourth. Before this, "activity fell" was unreachable by
    # construction -- a dead signal that no generated patient could trigger.
    #   sleep    -> fragmented nights, drifting resting HR
    #   activity -> sustained drop in daily steps (pain flare, job loss)
    DISRUPTIONS = [
        ("sofa bed", "sleeping arrangement changed", "sleep"),
        ("couch", "sleeping arrangement changed", "sleep"),
        ("night shift", "shift work started", "sleep"),
        ("newborn", "newborn at home", "sleep"),
        ("flare", "pain flare limiting activity", "activity"),
        ("hurts to walk", "pain limiting walking", "activity"),
        ("laid off", "job loss", "activity"),
        ("lost my job", "job loss", "activity"),
    ]
    disruption, cue, kind = None, None, None
    for phrase, label, k in DISRUPTIONS:
        if phrase in transcript:
            disruption, cue, kind = 45, label, k   # began ~45 days ago
            break

    days, notes = [], []
    for i in range(DAYS):
        d = end - timedelta(days=DAYS - 1 - i)
        weekend = d.weekday() >= 5
        s = steps * (0.82 if weekend else 1.0) * rng.uniform(0.75, 1.25)
        h = rhr + rng.uniform(-3, 3)
        sl = sleep + rng.uniform(-0.8, 0.8)
        eff = rng.uniform(0.86, 0.94)
        wake = rng.randint(0, 2)

        if disruption is not None and i >= DAYS - disruption:
            if kind == "sleep":
                # fragmented nights, drifting resting HR
                sl -= 1.3
                eff -= 0.11
                wake += rng.randint(1, 3)
                h += 4
            else:
                # activity collapse: deconditioning nudges resting HR up too,
                # but the signature is the step count. -45% clears the cue
                # extractor's 1500-step threshold at every plausible baseline
                # in this cohort (lowest non-hospice is ~3.5k).
                s *= 0.55
                h += 2

        row = {
            "date": d.isoformat(),
            "steps": int(max(0, s)),
            "resting_hr": round(h, 1),
            "sleep_hours": round(max(2.5, sl), 1),
            "sleep_efficiency": round(min(0.99, max(0.55, eff)), 2),
            "awakenings": wake,
        }
        if hypertensive and i % 3 == 0:        # home cuff, every ~3 days
            row["home_bp"] = f"{int(rng.gauss(138, 7))}/{int(rng.gauss(88, 5))}"
        days.append(row)

    if disruption is not None:
        notes.append(
            f"Sleep quality degraded ~{disruption} days ago ({cue}); "
            f"awakenings and resting heart rate both rose."
            if kind == "sleep" else
            f"Daily activity dropped ~{disruption} days ago ({cue}); "
            f"step count fell and resting heart rate drifted up."
        )
    return pid, {"days": days, "notes": notes, "kind": kind, "synthetic": True}


def verify(out):
    """Confirm the generated series actually trips the agent's cue extractor.

    The generator and clues.wearable_cues() have separate notions of "a
    signal" -- one writes deltas, the other reads them against thresholds over
    a 40-day head/tail window. Nothing keeps them in sync, so a threshold
    change or a smaller DAYS would silently produce a dataset that looks
    populated and yields zero cues. Fail loudly here instead."""
    sys.path.insert(0, str(ROOT / "patient_advocate"))
    import clues

    problems, fired = [], 0
    for pid, v in out.items():
        cues = clues.wearable_cues(v["days"])
        if v["kind"] and not cues:
            problems.append(f"{pid[:8]}: {v['kind']} event generated but no cue extracted")
        if v["kind"] == "activity" and not any("activity fell" in c for c in cues):
            problems.append(f"{pid[:8]}: activity event did not trip the steps signal")
        fired += bool(cues)
    return fired, problems


def main():
    recs = [json.loads(l) for l in open(COHORT)]
    out = dict(build(r) for r in recs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))

    kinds = Counter(v["kind"] for v in out.values() if v["kind"])
    fired, problems = verify(out)
    print(f"wrote {OUT}  ({len(out)} patients, {DAYS}d each)")
    print(f"life events: {dict(kinds)}   patients yielding >=1 cue: {fired}")
    for p in problems:
        print(f"  WARN {p}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
