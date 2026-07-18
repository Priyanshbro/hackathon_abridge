from __future__ import annotations
import anthropic

import chart
import detect
import deliver
import resolve
import stream


def run_encounter(patient_id: str, speed: float = 8.0, client=None) -> dict:
    """Orchestrate one encounter end to end:
      Before: detect.run_all(rec) -> candidates
      During: stream.replay(transcript, speed) -> resolve.resolve() per utterance
      After:  deliver.deliver(client, candidates, rec) -> {visit, health_maintenance}

    Returns a dict shaped for eval.py:
      {"patient_id", "candidates_proposed", "candidates_delivered",
       "candidates": [...], "delivered": [...]}
    """
    client = client or anthropic.Anthropic()
    rec = chart.record_for(patient_id)
    candidates = detect.run_all(rec, client=client)

    history: list[stream.Utterance] = []
    for utterance in stream.replay(rec["transcript"], speed=speed):
        history.append(utterance)
        resolve.resolve(candidates, utterance, history)

    out = deliver.deliver(client, candidates, rec)
    delivered = out["visit"]
    maintenance = out["health_maintenance"]

    return {
        "patient_id": patient_id,
        "candidates_proposed": len(candidates),
        "candidates_delivered": len(delivered),
        "candidates": candidates,
        "delivered": delivered,
        "health_maintenance": maintenance,
    }
