from __future__ import annotations
import re
import time
from dataclasses import dataclass


@dataclass
class Utterance:
    idx: int
    speaker: str  # DR | PT | NURSE | FAMILY
    text: str
    t: float  # seconds from start


_LINE_RE = re.compile(r"^(DR|PT|NURSE|FAMILY):\s*(.*)$")


def replay(transcript: str, speed: float = 8.0):
    """Split transcript on newlines, parse `SPK:` prefix, yield Utterance,
    sleep(dur/speed) between each so the visit compresses in real time."""
    idx = 0
    t = 0.0
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        speaker, text = m.groups()
        dur = max(len(text.split()) / 2.5, 1.0)  # ~150wpm speech estimate
        yield Utterance(idx=idx, speaker=speaker, text=text, t=t)
        time.sleep(dur / speed)
        t += dur
        idx += 1
