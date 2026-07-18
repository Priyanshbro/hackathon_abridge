from __future__ import annotations

import anthropic
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

import chart
import deliver
import detect
import resolve
import stream

MAX_TRANSCRIPT_LINES = 14
SPEAKER_STYLE = {"DR": "bold cyan", "NURSE": "cyan", "PT": "white", "FAMILY": "dim white"}


def build_layout() -> Layout:
    layout = Layout()
    layout.split_row(
        Layout(name="transcript", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["right"].split_column(
        Layout(name="candidates"),
        Layout(name="delivered"),
    )
    return layout


def render_transcript(lines: list[Text]) -> Panel:
    visible = lines[-MAX_TRANSCRIPT_LINES:]
    return Panel(Group(*visible) if visible else Text(""), title="transcript", border_style="blue")


def render_candidates(candidates: list[detect.Candidate], order: dict[str, int]) -> Panel:
    rows = []
    for c in sorted(candidates, key=lambda c: order[c.id]):
        if c.resolved_by is not None:
            rows.append(Text(f"[X] {c.topic}", style="strike dim"))
        else:
            rows.append(Text(f"[ ] {c.topic}  (p={c.priority:.2f})", style="yellow"))
    if not rows:
        rows = [Text("no candidates yet", style="dim")]
    return Panel(Group(*rows), title="open candidates", border_style="yellow")


def render_delivered(items: list[deliver.DeliveredItem] | None) -> Panel:
    if items is None:
        return Panel(Text("...listening", style="dim"), title="delivered", border_style="green")
    if not items:
        return Panel(Text("nothing survived suppression", style="dim"), title="delivered", border_style="green")
    rows = []
    for it in items:
        rows.append(Text(f"* {it.question}", style="bold green"))
        if it.context:
            rows.append(Text(f"  {it.context}", style="dim green"))
        rows.append(Text(""))
    return Panel(Group(*rows), title=f"delivered ({len(items)})", border_style="green")


def run_live(patient_id: str, speed: float = 8.0, client=None, console: Console | None = None) -> dict:
    """Same three building blocks as agent.py (detect -> resolve -> deliver),
    but hooked into a rich.Live display per utterance instead of returning
    only a final result. Kept separate from agent.run_encounter() because a
    live UI needs to render intermediate state; agent.py stays the headless
    orchestrator eval.py drives."""
    client = client or anthropic.Anthropic()
    rec = chart.record_for(patient_id)
    candidates = detect.run_all(rec)

    # stable display order, captured once -- resolve.py zeroes priority on
    # resolution, so sorting live by priority would make a candidate jump to
    # the bottom the instant it strikes through. Order stays fixed; only
    # the [ ]/[X] state changes.
    order = {c.id: i for i, c in enumerate(sorted(candidates, key=lambda c: c.priority, reverse=True))}

    layout = build_layout()
    lines: list[Text] = []
    layout["transcript"].update(render_transcript(lines))
    layout["candidates"].update(render_candidates(candidates, order))
    layout["delivered"].update(render_delivered(None))

    with Live(layout, console=console, refresh_per_second=8, screen=False) as live:
        history: list[stream.Utterance] = []
        for utterance in stream.replay(rec["transcript"], speed=speed):
            history.append(utterance)
            resolve.resolve(candidates, utterance, history)

            style = SPEAKER_STYLE.get(utterance.speaker, "white")
            lines.append(Text(f"{utterance.speaker}: {utterance.text}", style=style))
            layout["transcript"].update(render_transcript(lines))
            layout["candidates"].update(render_candidates(candidates, order))
            live.refresh()

        delivered = deliver.deliver(client, candidates, rec)
        layout["delivered"].update(render_delivered(delivered))
        live.refresh()

    return {
        "patient_id": patient_id,
        "candidates_proposed": len(candidates),
        "candidates_delivered": len(delivered),
        "candidates": candidates,
        "delivered": delivered,
    }


if __name__ == "__main__":
    import sys

    pid = sys.argv[1] if len(sys.argv) > 1 else "4b4735a2-ee12-ec86-041f-3ba4d5c81ec9"
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    run_live(pid, speed=speed)
