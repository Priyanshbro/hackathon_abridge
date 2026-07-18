from __future__ import annotations

import anthropic

import agent
import chart
import deliver
import detect
import resolve
import stream


def all_patient_ids() -> list[str]:
    return [r["metadata"]["patient_id"] for r in chart.load_cohort()]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2


def candidates_survived(rec: dict, suppression: bool = True) -> list[detect.Candidate]:
    """Deterministic only -- detect + (optionally) resolve. No LLM calls,
    no cost. This is the suppression ON/OFF ablation: 'OFF delivers ~20
    questions per visit... ON delivers 3' (PLAN.md). Runs replay at max
    speed since nothing here needs to feel real-time."""
    candidates = detect.run_all(rec)
    if suppression:
        history: list[stream.Utterance] = []
        for utterance in stream.replay(rec["transcript"], speed=1e9):
            history.append(utterance)
            resolve.resolve(candidates, utterance, history)
    return deliver.filter_survivors(candidates)


def ablation(patient_ids: list[str]) -> dict:
    rows = []
    for pid in patient_ids:
        rec = chart.record_for(pid)
        proposed = len(detect.run_all(rec))
        on = len(candidates_survived(rec, suppression=True))
        off = len(candidates_survived(rec, suppression=False))
        rows.append(
            {"patient_id": pid, "proposed": proposed, "suppression_on": on, "suppression_off": off}
        )
    return {
        "rows": rows,
        "median_proposed": _median([r["proposed"] for r in rows]),
        "median_suppression_on": _median([r["suppression_on"] for r in rows]),
        "median_suppression_off": _median([r["suppression_off"] for r in rows]),
    }


def check_groundedness(candidates: list[detect.Candidate], delivered: list) -> dict:
    """Deterministic -- verify each delivered question's own candidate
    actually has non-empty, valued evidence. No LLM judge, no noise.
    `delivered` is a list of deliver.DeliveredItem."""
    by_id = {c.id: c for c in candidates}
    results = []
    for q in delivered:
        c = by_id.get(q.id)
        ok = c is not None and len(c.evidence) > 0 and all(
            e.get("value") not in (None, "") for e in c.evidence
        )
        results.append({"id": q.id, "grounded": ok})
    n_ok = sum(1 for r in results if r["grounded"])
    return {"results": results, "rate": n_ok / len(results) if results else 1.0}


def check_suppression_accuracy(candidates: list[detect.Candidate], sample_size: int = 5) -> dict:
    """Spot-check: for each resolved candidate, does resolved_by actually
    contain a plan-verb from that candidate's own vocabulary? Deterministic
    re-derivation using resolve.py's own matcher -- proves suppression is
    real, not lossy."""
    resolved = [c for c in candidates if c.resolved_by is not None]
    sample = resolved[:sample_size]
    checks = []
    for c in sample:
        plan_verbs = c.plan_verbs or resolve.CLINICAL_PLAN_VERBS
        text = c.resolved_by.split(": ", 1)[-1] if c.resolved_by else ""
        has_verb = resolve._contains_any(text, plan_verbs)
        checks.append({"id": c.id, "resolved_by": c.resolved_by, "plan_verb_present": has_verb})
    n_ok = sum(1 for c in checks if c["plan_verb_present"])
    return {"checks": checks, "accuracy": n_ok / len(checks) if checks else 1.0}


def run_full(patient_ids: list[str], client=None) -> list[dict]:
    """The only function here that spends API credits -- one full run per
    patient (detect + resolve + Gate B + Gate C + phrase). Run on a small
    subset first, not all 25, until the pipeline is trusted."""
    client = client or anthropic.Anthropic()
    results = []
    for pid in patient_ids:
        out = agent.run_encounter(pid, speed=1e9, client=client)
        results.append(
            {
                **out,
                "groundedness": check_groundedness(out["candidates"], out["delivered"]),
                "suppression_accuracy": check_suppression_accuracy(out["candidates"]),
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    proposed = [r["candidates_proposed"] for r in results]
    delivered = [r["candidates_delivered"] for r in results]
    return {
        "n_encounters": len(results),
        "median_proposed": _median(proposed),
        "median_delivered": _median(delivered),
        "max_delivered": max(delivered) if delivered else 0,
        "mean_groundedness_rate": (
            sum(r["groundedness"]["rate"] for r in results) / len(results) if results else 1.0
        ),
    }
