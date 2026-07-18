"""
Local web UI for the patient advocate agent -- same pipeline as ui.py
(chart -> detect -> resolve -> deliver), streamed to a browser over SSE
instead of rendered with rich.Live in a terminal. Built because the
terminal's fixed height truncates long delivered lists; a real page
scrolls instead.

Run: python3 webui.py [patient_id] [port]
Then open http://localhost:<port>/
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import pathlib
import queue
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import anthropic

import chart
import deliver
import detect
import resolve
import stream

DEFAULT_PATIENT = "4b4735a2-ee12-ec86-041f-3ba4d5c81ec9"  # Elias
DEFAULT_PORT = 8787

ENV_LOCAL_PATH = pathlib.Path(__file__).parent / ".env.local"


def _load_api_key() -> str | None:
    """ANTHROPIC_API_KEY if already set, otherwise parsed directly out of
    .env.local. Some process launchers (dev-server preview harnesses in
    particular) sandbox the spawned shell's environment and don't reliably
    propagate `source .env.local` through to the child process -- reading
    the file ourselves sidesteps that entirely rather than fighting shell
    env-inheritance quirks."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    if not ENV_LOCAL_PATH.exists():
        return None
    match = re.search(r'ANTHROPIC_API_KEY="([^"]+)"', ENV_LOCAL_PATH.read_text())
    return match.group(1) if match else None


def _client() -> anthropic.Anthropic:
    key = _load_api_key()
    if key:
        return anthropic.Anthropic(api_key=key)
    return anthropic.Anthropic()  # let the SDK raise its normal error if nothing is configured

PATIENTS = dict(
    sorted(
        ((r["metadata"]["patient_id"], chart.patient_name(r)) for r in chart.load_cohort()),
        key=lambda kv: kv[1],
    )
)


def _candidate_dict(c: detect.Candidate) -> dict:
    return {
        "id": c.id,
        "topic": c.topic,
        "kind": c.kind,
        "trigger": c.trigger,
        "priority": c.priority,
        "resolved": c.resolved_by is not None,
        "resolved_by": c.resolved_by,
    }


def _item_dict(it: deliver.DeliveredItem) -> dict:
    return dataclasses.asdict(it)


# Detection reads only the chart + coverage_271 -- never the transcript --
# so in a real deployment it's the part that could run the moment a visit is
# scheduled, well before the patient is in the room. The demo mirrors that:
# the moment the page loads, every patient's detection gets warmed into
# detection_cache.json in the background, so by the time someone clicks
# "Run visit" only the transcript (genuinely unavailable pre-visit) and
# grounding are live. Picking "Live detection" in the dropdown still forces
# a fresh discover() call regardless of what's warmed, for demoing the
# reasoning stream itself.
_warm_state_lock = threading.Lock()
_warm_state = {"total": len(PATIENTS), "done": 0, "pending": set(PATIENTS)}
_warm_started = False
_warm_lock = threading.Lock()


def _warm_one(pid: str, client) -> None:
    try:
        detect.run_all(chart.record_for(pid), client=client, use_cache=True)
    except Exception:
        pass  # best-effort prep -- a live "Run visit" click will just detect fresh for this patient
    finally:
        with _warm_state_lock:
            _warm_state["done"] += 1
            _warm_state["pending"].discard(pid)


def _prewarm_all_patients() -> None:
    client = _client()
    # max_workers=4: patients already in detection_cache.json return instantly
    # and cost nothing, but any that need a fresh discover() call are a real
    # ~20-40s Opus call -- capped concurrency avoids bursting 25 of those at
    # once against rate limits.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda pid: _warm_one(pid, client), PATIENTS))


def ensure_prewarm_started() -> None:
    global _warm_started
    with _warm_lock:
        if _warm_started:
            return
        _warm_started = True
    threading.Thread(target=_prewarm_all_patients, daemon=True).start()


def _stream_worker(fn):
    """Run fn(on_event) on a background thread and yield each event the
    moment it's queued, instead of after fn() returns. detect.discover()
    and deliver.ground() call on_event synchronously from inside a blocking
    API call -- without a thread, those events could only be handed back to
    this generator after the whole call finished, defeating the point of a
    live reasoning feed. fn's return value comes back through `yield from`,
    which surfaces a generator's `return` value as its StopIteration
    result; any exception raised in the worker is re-raised here instead."""
    q: queue.Queue = queue.Queue()
    box = {}

    def on_event(ev):
        q.put(("event", ev))

    def worker():
        try:
            box["result"] = fn(on_event)
        except Exception as exc:  # noqa: BLE001 -- re-raised on the generator's thread below
            box["error"] = exc
        finally:
            q.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        kind, payload = q.get()
        if kind == "done":
            break
        yield payload
    if "error" in box:
        raise box["error"]
    return box["result"]


def run_encounter_events(patient_id: str, speed: float = 8.0, use_cache: bool = True, client=None):
    """Same three phases as ui.run_live(), but yields JSON-serializable
    event dicts instead of updating a rich Layout. Runs in real wall-clock
    time (stream.replay sleeps per utterance), so events arrive at the
    same pace a terminal viewer would see."""
    client = client or _client()
    rec = chart.record_for(patient_id)

    yield {"type": "status", "message": "Reading chart, detecting candidates..."}
    candidates = yield from _stream_worker(
        lambda on_event: detect.run_all(rec, client=client, use_cache=use_cache, on_event=on_event)
    )
    order = {c.id: i for i, c in enumerate(sorted(candidates, key=lambda c: c.priority, reverse=True))}

    yield {
        "type": "init",
        "patient_name": PATIENTS.get(patient_id, patient_id),
        "visit_title": chart.visit_title(rec),
        "order": order,
        "candidates": [_candidate_dict(c) for c in candidates],
    }

    history: list[stream.Utterance] = []
    for utterance in stream.replay(rec["transcript"], speed=speed):
        history.append(utterance)
        resolve.resolve(candidates, utterance, history)
        yield {
            "type": "utterance",
            "idx": utterance.idx,
            "speaker": utterance.speaker,
            "text": utterance.text,
            "candidates": [_candidate_dict(c) for c in candidates],
        }

    yield {"type": "status", "message": "Visit complete -- grounding and phrasing..."}
    out = yield from _stream_worker(
        lambda on_event: deliver.deliver(client, candidates, rec, on_event=on_event)
    )
    yield {
        "type": "delivered",
        "visit": [_item_dict(i) for i in out["visit"]],
        "health_maintenance": [_item_dict(i) for i in out["health_maintenance"]],
    }
    yield {"type": "done"}


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Patient Advocate</title>
<style>
  :root {
    --bg: #F4F1EA;
    --panel: #FBFAF6;
    --panel-alt: #FFFFFF;
    --border: #E4DFD3;
    --text: #2A2620;
    --text-muted: #7A7367;
    --accent: #C15F3C;
    --accent-soft: #EFE0D8;
    --resolved: #B7B2A4;
    --maint-accent: #6E7B6A;
    --maint-soft: #E4E9E0;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.55;
  }
  h1, .serif { font-family: Georgia, "Iowan Old Style", Charter, serif; }

  header {
    display: flex; align-items: center; gap: 16px;
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  header h1 {
    font-size: 20px; font-weight: 500; margin: 0;
    color: var(--text);
  }
  header .sub { color: var(--text-muted); font-size: 13px; }
  header .controls { margin-left: auto; display: flex; gap: 10px; align-items: center; }
  select, button {
    font: inherit;
    padding: 7px 14px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--panel-alt);
    color: var(--text);
  }
  button {
    background: var(--accent);
    color: #fff;
    border: none;
    font-weight: 500;
    cursor: pointer;
  }
  button:hover { opacity: 0.92; }
  button:disabled { opacity: 0.5; cursor: default; }

  .status {
    padding: 8px 28px;
    color: var(--text-muted);
    font-size: 13px;
    font-style: italic;
    min-height: 34px;
    display: flex;
    align-items: center;
  }
  #warm-status {
    padding-top: 0;
    min-height: 0;
    font-size: 12px;
  }
  #warm-status:empty { display: none; }

  main {
    display: grid;
    grid-template-columns: 1.1fr 1fr 1fr;
    gap: 20px;
    padding: 0 28px 28px;
    height: calc(100vh - 130px);
  }
  .col { display: flex; flex-direction: column; gap: 20px; min-height: 0; }

  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .card h2 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    color: var(--text-muted);
    margin: 0 0 12px;
  }
  .card-scroll { overflow-y: auto; min-height: 0; flex: 1; }

  #transcript-card { flex: 1; }
  #transcript { display: flex; flex-direction: column; gap: 10px; }
  .line { padding-right: 4px; }
  .line .speaker {
    font-weight: 600;
    margin-right: 6px;
  }
  .line.DR .speaker { color: var(--accent); }
  .line.NURSE .speaker { color: var(--accent); opacity: 0.8; }
  .line.PT .speaker { color: var(--text); }
  .line.FAMILY .speaker { color: var(--text-muted); }
  .line .text { color: var(--text); }
  .line.PT .text, .line.FAMILY .text { color: var(--text); }

  #candidates-card { flex: 1; }
  .candidate {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid var(--border);
    transition: opacity 0.4s ease;
  }
  .candidate:last-child { border-bottom: none; }
  .candidate .dot {
    width: 9px; height: 9px; border-radius: 50%;
    margin-top: 5px; flex-shrink: 0;
    background: var(--accent);
  }
  .candidate.resolved .dot { background: var(--resolved); }
  .candidate .topic {
    font-size: 13.5px;
    color: var(--text);
  }
  .candidate.resolved .topic {
    color: var(--text-muted);
    text-decoration: line-through;
  }
  .candidate .priority {
    margin-left: auto;
    font-size: 11px;
    color: var(--text-muted);
    white-space: nowrap;
    padding-top: 1px;
  }

  #delivered-card { flex: 1.3; }
  .section-label {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text-muted); font-weight: 600;
    margin: 14px 0 8px;
  }
  .section-label:first-child { margin-top: 0; }
  .item {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
  }
  .item .question {
    font-size: 14px;
    color: var(--text);
  }
  .item .context {
    margin-top: 6px;
    font-size: 12.5px;
    color: var(--text-muted);
    border-left: 2px solid var(--accent-soft);
    padding-left: 8px;
  }
  .item.maint { border-left: 3px solid var(--maint-accent); }
  .item.maint .context { border-left-color: var(--maint-soft); }
  .empty-note { color: var(--text-muted); font-style: italic; font-size: 13px; }

  #reasoning-card { flex: 1; }
  #reasoning-card h2 { display: flex; align-items: center; gap: 7px; }
  .live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--resolved);
  }
  .live-dot.active {
    background: var(--accent);
    animation: pulse 1.1s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
  }
  #reasoning {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
    line-height: 1.6;
  }
  .reasoning-line {
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
  }
  .reasoning-line:last-child { border-bottom: none; }
  .reasoning-line.thinking {
    color: var(--text);
    font-style: italic;
    white-space: pre-wrap;
  }
  .reasoning-line .tag {
    display: inline-block;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 2px 7px;
    border-radius: 5px;
    margin-right: 7px;
  }
  .reasoning-line .tag.found { background: var(--accent-soft); color: var(--accent); }
  .reasoning-line .tag.pass { background: var(--maint-soft); color: var(--maint-accent); }
  .reasoning-line .tag.fail { background: var(--border); color: var(--text-muted); }
  .reasoning-line .meta { color: var(--text-muted); }

  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
</style>
</head>
<body>

<header>
  <h1 class="serif">Patient Advocate</h1>
  <span class="sub" id="visit-title"></span>
  <div class="controls">
    <select id="patient-select">
      __PATIENT_OPTIONS__
    </select>
    <select id="speed-select">
      <option value="1000000">Instant</option>
      <option value="8" selected>8x (demo pace)</option>
      <option value="2">2x (slow)</option>
    </select>
    <button id="run-btn">Run visit</button>
  </div>
</header>

<div class="status" id="status"></div>
<div class="status" id="warm-status"></div>

<main>
  <div class="col">
    <div class="card" id="transcript-card">
      <h2>Transcript</h2>
      <div class="card-scroll" id="transcript"></div>
    </div>
  </div>
  <div class="col">
    <div class="card" id="candidates-card">
      <h2>Open candidates</h2>
      <div class="card-scroll" id="candidates"></div>
    </div>
    <div class="card" id="delivered-card">
      <h2>Delivered</h2>
      <div class="card-scroll" id="delivered">
        <div class="empty-note">Run a visit to see delivered questions.</div>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card" id="reasoning-card">
      <h2>Agent reasoning <span class="live-dot" id="reasoning-live"></span></h2>
      <div class="card-scroll" id="reasoning">
        <div class="empty-note">Live model reasoning appears here during grounding, and during detection for any patient whose chart hasn't been pre-warmed yet.</div>
      </div>
    </div>
  </div>
</main>

<script>
let order = {};

function renderCandidates(list) {
  const el = document.getElementById('candidates');
  const sorted = [...list].sort((a, b) => (order[a.id] ?? 0) - (order[b.id] ?? 0));
  el.innerHTML = sorted.map(c => `
    <div class="candidate ${c.resolved ? 'resolved' : ''}">
      <div class="dot"></div>
      <div class="topic">${c.topic.split('|')[0]}</div>
      ${c.resolved ? '' : `<div class="priority">p=${c.priority.toFixed(2)}</div>`}
    </div>
  `).join('');
}

function renderDelivered(visit, maintenance) {
  const el = document.getElementById('delivered');
  if (!visit.length && !maintenance.length) {
    el.innerHTML = '<div class="empty-note">Nothing survived suppression.</div>';
    return;
  }
  let html = '<div class="section-label">This visit</div>';
  html += visit.map(it => `
    <div class="item">
      <div class="question">${it.question}</div>
      ${it.context ? `<div class="context">${it.context}</div>` : ''}
    </div>
  `).join('') || '<div class="empty-note">Nothing in scope for a K=3 ask right now.</div>';

  if (maintenance.length) {
    html += '<div class="section-label">Health maintenance</div>';
    html += maintenance.map(it => `
      <div class="item maint">
        <div class="question">${it.question}</div>
        ${it.context ? `<div class="context">${it.context}</div>` : ''}
      </div>
    `).join('');
  }
  el.innerHTML = html;
}

function appendLine(speaker, text) {
  const el = document.getElementById('transcript');
  const div = document.createElement('div');
  div.className = 'line ' + speaker;
  div.innerHTML = `<span class="speaker">${speaker}:</span><span class="text">${text}</span>`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

// `thinkingEl` accumulates streamed thinking_delta chunks into one growing
// paragraph; it's reset to null on any non-thinking reasoning event so the
// next thinking burst (e.g. the next model call) starts a fresh paragraph
// instead of running on from the previous one.
let thinkingEl = null;

function appendThinkingDelta(text) {
  const el = document.getElementById('reasoning');
  if (!thinkingEl) {
    thinkingEl = document.createElement('div');
    thinkingEl.className = 'reasoning-line thinking';
    el.appendChild(thinkingEl);
  }
  thinkingEl.textContent += text;
  el.scrollTop = el.scrollHeight;
}

function appendReasoning(html) {
  thinkingEl = null;
  const el = document.getElementById('reasoning');
  const div = document.createElement('div');
  div.className = 'reasoning-line';
  div.innerHTML = html;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function run() {
  const patientId = document.getElementById('patient-select').value;
  const speed = document.getElementById('speed-select').value;
  const btn = document.getElementById('run-btn');
  btn.disabled = true;

  document.getElementById('transcript').innerHTML = '';
  document.getElementById('candidates').innerHTML = '';
  document.getElementById('delivered').innerHTML = '<div class="empty-note">Running...</div>';
  document.getElementById('reasoning').innerHTML = '';
  document.getElementById('status').textContent = '';
  document.getElementById('reasoning-live').classList.add('active');
  thinkingEl = null;

  const es = new EventSource(`/events?patient_id=${patientId}&speed=${speed}`);

  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'status') {
      document.getElementById('status').textContent = ev.message;
    } else if (ev.type === 'init') {
      order = ev.order;
      document.getElementById('visit-title').textContent = ev.visit_title;
      document.getElementById('status').textContent = '';
      renderCandidates(ev.candidates);
    } else if (ev.type === 'utterance') {
      appendLine(ev.speaker, ev.text);
      renderCandidates(ev.candidates);
    } else if (ev.type === 'delivered') {
      renderDelivered(ev.visit, ev.health_maintenance);
    } else if (ev.type === 'thinking') {
      appendThinkingDelta(ev.text);
    } else if (ev.type === 'candidate_found') {
      appendReasoning(`<span class="tag found">found</span>${ev.trigger} <span class="meta">(${ev.kind}, p=${ev.priority.toFixed(2)})</span>`);
    } else if (ev.type === 'ground_result') {
      appendReasoning(`<span class="tag ${ev.grounded ? 'pass' : 'fail'}">${ev.grounded ? 'grounded' : 'dropped'}</span>${ev.reason}`);
    } else if (ev.type === 'done') {
      document.getElementById('status').textContent = '';
      document.getElementById('reasoning-live').classList.remove('active');
      es.close();
      btn.disabled = false;
    } else if (ev.type === 'error') {
      document.getElementById('status').textContent = 'Error: ' + ev.message;
      document.getElementById('reasoning-live').classList.remove('active');
      es.close();
      btn.disabled = false;
    }
  };
  es.onerror = () => {
    btn.disabled = false;
    document.getElementById('reasoning-live').classList.remove('active');
    es.close();
  };
}

document.getElementById('run-btn').addEventListener('click', run);

function pollWarmStatus() {
  fetch('/warm_status').then(r => r.json()).then(s => {
    const el = document.getElementById('warm-status');
    if (s.done >= s.total) {
      el.textContent = '';
      return;
    }
    el.textContent = `Pre-warming chart + insurance cache: ${s.done}/${s.total} patients ready...`;
    setTimeout(pollWarmStatus, 1500);
  }).catch(() => {});
}
pollWarmStatus();
</script>
</body>
</html>
"""


def _build_index_html() -> str:
    options = "\n".join(
        f'<option value="{pid}"{" selected" if pid == DEFAULT_PATIENT else ""}>{name}</option>'
        for pid, name in PATIENTS.items()
    )
    return INDEX_HTML.replace("__PATIENT_OPTIONS__", options)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet -- SSE polling would otherwise spam the console

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            ensure_prewarm_started()
            body = _build_index_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/warm_status":
            with _warm_state_lock:
                body = json.dumps({"total": _warm_state["total"], "done": _warm_state["done"]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/events":
            qs = parse_qs(parsed.query)
            patient_id = qs.get("patient_id", [DEFAULT_PATIENT])[0]
            speed = float(qs.get("speed", ["8"])[0])

            # protocol_version defaults to HTTP/1.0 on BaseHTTPRequestHandler
            # (never overridden here), which closes the connection when the
            # handler returns -- exactly right for a one-shot stream. Sending
            # "Connection: keep-alive" on top of that was a mismatched signal:
            # it told the browser to expect HTTP/1.1 framing (chunked
            # transfer) that this server never actually sends, corrupting the
            # stream mid-message. Let HTTP/1.0's natural close-on-done stand.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                client = _client()
                for event in run_encounter_events(patient_id, speed=speed, client=client):
                    payload = f"data: {json.dumps(event)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001 -- surface any pipeline error to the browser
                try:
                    payload = f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return

        self.send_response(404)
        self.end_headers()


def main():
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Patient Advocate UI running at http://localhost:{port}/")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
