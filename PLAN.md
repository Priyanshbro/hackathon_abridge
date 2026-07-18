# Build plan — patient advocate agent

**Thesis:** a patient-side agent that surfaces the questions the patient should be
asking — grounded in their chart, deterministic care-gap computation, and abductive
clue-elevation from their own life context.

**The agent never interrupts the clinician.** It listens to the whole visit, watches
which of its concerns the doctor addresses unprompted, and hands the patient only what
is left over at the end. A device that chirps mid-encounter derails a focused history
and would never be deployed; a short list handed over at the door is exactly the moment
that already exists in care (the after-visit summary).

**So the hard problem is not "call an LLM." It is *what survives*.** A visit generates
~20 candidate concerns across ~90 utterances. Most get addressed by a good clinician
without prompting. Everything below exists to deliver **3 questions that genuinely were
not answered**, instead of 20 that mostly were.

---

## Three phases, one engine

| Phase | What happens | Output |
|---|---|---|
| **Before** | generators run on chart + wearable | prep questions the patient brings in |
| **During** | agent listens silently; marks candidates *resolved* as the doctor covers them | nothing shown — this is the suppression work |
| **After** | surviving candidates ranked, grounded, phrased | the delivered short list |

The *During* phase produces no output. That is the point, and it is what makes the
*After* list short.

---

## File layout

```
patient_advocate/
  stream.py      Utterance type + replay adapter + whisper adapter
  chart.py       cohort loader + FHIR/wearable accessors
  detect.py      deterministic candidate generators   <- item 2
  resolve.py     resolution tracking + suppression
  deliver.py     end-of-visit grounding, ranking, phrasing
  agent.py       orchestration                        <- item 1
  eval.py        all-25 run, metrics, suppression ON/OFF  <- item 4
  ui.py          rich live display                    <- item 3
  make_wearable.py
```

---

## The one abstraction that matters

Everything downstream consumes `Utterance`. The agent cannot tell replay from live audio.

```python
@dataclass
class Utterance:
    idx: int
    speaker: str      # DR | PT | NURSE | FAMILY
    text: str
    t: float          # seconds from start
```

Two adapters, same output type:
- `replay(transcript, speed=8.0)` — split on newline, parse `SPK:`, sleep(dur/speed)
- `whisper_stream(wav)` — segments from faster_whisper; recover speaker by matching
  segment start-time against the known TTS utterance boundaries (Whisper has no
  diarization; in production Abridge's ASR supplies it)

Build `replay` first. Add `whisper_stream` last, only if ahead of schedule.

**Because nothing is delivered mid-stream, replay can run fast (8–20x).** A 10-minute
visit compresses to under a minute of demo. This is a direct benefit of end-of-visit
delivery.

---

## Item 2 — deterministic layer (`detect.py`, 1.0h)

Pure Python. No LLM. Each generator emits candidates with **structured evidence**,
which is what makes groundedness checkable later.

```python
@dataclass
class Candidate:
    id: str
    topic: str            # for resolution matching
    kind: str             # gap | score | drug_disease | clue | wearable
    trigger: str          # human-readable why
    evidence: list[dict]  # [{source, field, value, date}]  <- must be verifiable
    priority: float
    resolved_by: str|None # utterance idx + span, set during the visit
```

Generators (each 10–20 lines; do not gold-plate):

| # | Generator | Fires when |
|---|---|---|
| 1 | `bp_control` | BP >=140/90 -> stage 2 |
| 2 | `egfr` | CKD-EPI 2021 from creatinine+age+sex; flag <60, or diuretic started with no documented eGFR |
| 3 | `ascvd_inputs_missing` | cardiometabolic condition set but no lipid panel -> risk not computable |
| 4 | `statin_gap` | condition set implies statin consideration + no statin in med labels |
| 5 | `a1c_threshold` | 5.7–6.4 prediabetes, >=6.5 diabetes |
| 6 | `phq2_escalation` | PHQ-2 >=3 -> PHQ-9 required (hard rule) |
| 7 | `drug_disease` | e.g. thiazide + prediabetes (HCTZ raises glucose) |
| 8 | `wearable_delta` | sleep / RHR / steps shift beyond threshold in window |
| 9 | `transcript_clue` | lifestyle/exposure cue in patient's own words (sofa bed, shift work, new med) |
| 10 | `cost_adherence_risk` | prior cost/coverage disruption + new prescriptions today + no written cost contingency |

Rule: **generators propose, the model never invents.** The LLM only selects and phrases.

### Build steps — `cost_adherence_risk` (~20 lines, the second demo case)

Four small functions. The nuance in step 3 is what keeps it from being naive.

1. **`cost_barrier_evidence(rec) -> list[dict]`** — collect proof the patient has a
   cost/coverage problem, from two sources:
   - *transcript regex*: `insurance switched|stopped refilling|ran out of|couldn't
     afford|left it on the counter|\$\d+` near pharmacy/refill context
   - *structured*: PRAPARE `Primary insurance` in {Medicaid, None/uninsured};
     condition labels containing `Unemployed`, `Housing unsatisfactory`
   Each hit becomes an evidence dict with source + verbatim span, so groundedness
   stays deterministically checkable.
   **Do not require Medicaid/uninsured status** — Julius is on private insurance and
   his risk comes from the transcript disclosure. Gate on evidence, not category.

2. **`new_prescriptions(rec) -> list[str]`** — `MedicationRequest` resources on this
   encounter. This is the *exposure*: more new scripts = higher abandonment risk.

3. **`cost_contingency_present(...) -> bool`** — the important distinction.
   Do **not** test whether cost was *mentioned*. Julius's doctor did mention it
   ("all cheap generics, a few dollars a month") — a naive keyword trigger would
   wrongly mark it resolved. Test whether a **contingency** exists: an instruction for
   what to do *if* the price differs (call back, alternative, assistance program,
   90-day supply, pharmacy price check). Reassurance is not a plan.

4. **Assemble the Candidate.** Priority scales with:
   `prior_abandonment_event (bool) x n_new_prescriptions x (not contingency_present)`.

Pass the patient's own verbatim quote into the phrasing call so the question echoes
their language rather than clinical register.

---

## Item 1 — listen, resolve, deliver (`resolve.py` + `deliver.py` + `agent.py`, 2.5h)

### During the visit — `resolve.py` (deterministic, free, no LLM)

For each incoming utterance, test every open candidate for **resolution**:

- **addressed**: the clinician stated a plan, order, or explanation covering the topic
- **partially addressed**: topic raised but no actionable plan -> stays open, priority
  drops (Julius's cost case lives here — reassurance without contingency)
- **untouched**: never came up -> stays open at full priority

Match on topic keywords plus a plan-verb test (`start`, `order`, `refer`, `recheck`,
`we'll`, `I'm putting in`). Record `resolved_by` = utterance index + the verbatim span,
so the eval can show *why* something was suppressed.

**This is the whole product.** A good clinician resolves most candidates unprompted;
the residue is what the patient actually needs.

### At the end — `deliver.py`

Triggered on stream end, or on a closing cue (`any questions`, `see you`, `follow up`).

1. **Filter** — drop resolved candidates.
2. **Gate B — grounding** (one batched LLM call over all survivors): each question must
   cite a specific datum or it dies. Batched, so it is one call, not N.
3. **Gate C — budget**: hard cap `K = 3`, strict priority ordering.
4. **Phrase** — one call, patient's voice, first person, <=25 words each, each carrying
   its evidence line.

### LLM usage — now trivially cheap

End-of-visit delivery removes all latency pressure. **Two calls per encounter**, both
after the conversation ends:

- **Gate B (grounding, batched):** `claude-opus-4-8`, `output_config={"effort":"low"}`,
  `thinking={"type":"disabled"}`, json_schema output.
- **Phrasing:** `claude-opus-4-8`, adaptive thinking.

No streaming inference, no per-utterance calls, no latency risk during the demo.
This is a real simplification over mid-visit delivery — say so.

### Deliberate non-goal
No urgent-interrupt path. If something were genuinely time-critical you would want one,
and that is the obvious v2 — but interrupting is the thing clinicians reject, so the
default must be silence. Good Q&A answer; do not build it.

---

## Item 4 — eval (`eval.py`, 0.75h) — BUILD THIS BEFORE THE UI

Runs headless over all 25 encounters. Metrics:

| Metric | How | Why it matters |
|---|---|---|
| candidates proposed vs delivered | count | the headline: ~20 proposed -> 3 delivered |
| **suppression accuracy** | for each suppressed candidate, does `resolved_by` actually contain a plan for that topic? spot-check a sample | proves suppression is real, not lossy |
| **groundedness rate** | **deterministic** — verify each delivered question's evidence refs resolve to a real field/value | headline number, no judge noise |
| questions / encounter | count | median 2–3, never >5 |

**The money chart: suppression ON vs OFF.** Same run, flip resolution tracking.
OFF delivers ~20 questions per visit, most of them already answered during the visit —
which is precisely the useless product. ON delivers 3.
Cheap, honest, within-run A/B; almost nobody at a hackathon ships one.

Groundedness is checkable **deterministically** because candidates carry structured
evidence refs. That is the whole reason for the `evidence` field.

---

## Item 3 — UI (`ui.py`, 0.75h)

`rich` Live layout, three zones:
- **left**: utterances streaming past (fast)
- **right top**: open candidates, greying out and striking through as they resolve —
  *this is the visible suppression, and it is the best thing in the demo*
- **right bottom**: at end of stream, the 3 delivered questions with evidence lines

Terminal, not web — faster to build, screen-records well, no web stack.

---

## Timebox (submissions due 5:00 PM)

| Time | Work |
|---|---|
| 10:30–11:00 | repo + `chart.py` accessors, load cohort + wearable |
| 11:00–12:00 | `detect.py` — the 10 generators (item 2) |
| 12:00–12:30 | `stream.py` replay + candidate state |
| 12:30–13:00 | lunch / buffer |
| 13:00–14:30 | `resolve.py` + `deliver.py` + `agent.py` (item 1) |
| 14:30–15:15 | `eval.py` + suppression ON/OFF ablation (item 4) |
| 15:15–16:00 | `ui.py` (item 3) |
| 16:00–16:30 | wearable rewrite + TTS excerpt |
| 16:30–16:50 | record 1-min video, submit |

**There is no slack. Cut order if behind:**
1. UI degrades to coloured stdout (costs nothing in judging)
2. Whisper loop drops entirely — demo the replay
3. Wearable drops to a static note
4. **Never cut the eval.** It is the differentiator.

Eval is scheduled before UI on purpose: it runs headless, so if the day collapses
you want eval + stdout, not a pretty UI with no numbers.

---

## The two demo cases

Two patients, two **structurally different** generators, one engine. That contrast is
the proof the system generalizes rather than being tuned to a single record.

### Hero — Elias Wisozk (record 6), the wearable-clue path
54M, new BP 141/100, morning occipital headaches, BMI 30.2, A1c 6.28, started on HCTZ.
- he says it himself: sleeps on his sister's sofa bed, *"I blamed the sofa bed, honestly"*
- wearable corroborates: sleep 6.8 -> 5.4h, awakenings 1.1 -> 3.1, resting HR 72 -> 76
- the doctor started an antihypertensive and never asked about sleep
- poor sleep / undiagnosed apnea is a genuine reversible contributor to new hypertension
- **survives to delivery** because the topic never came up at all
- **question:** "Could my blood pressure and the headaches be about how badly I've slept
  since I moved in with my sister?"

### Second — Julius Renner (record 12), the cost-adherence path
36M, unemployed, unsatisfactory housing, hypertension diagnosed previously and
**never treated**.
- verbatim: *"I ran out of everything months ago and stopped refilling when the
  insurance switched over to my wife's plan"*
- verbatim: *"The pharmacy said it would be forty bucks and I just... left it on the counter"*
- today he receives **four** prescriptions (lisinopril, amlodipine, HCTZ, acetaminophen)
- the doctor offers reassurance ("cheap generics") but no contingency
- **survives to delivery as *partially addressed*** — the strongest possible
  demonstration that resolution tracking is not keyword matching
- **question:** "Last time I left the pharmacy because it was forty dollars. What will
  these four cost me on my wife's plan, and what do I do if the counter price isn't
  what you expect?"

Cost-driven non-adherence is one of the largest real drivers of uncontrolled
hypertension, so this question plausibly decides whether the treatment happens at all.

---

## Demo script (3 min)

1. **20s** — the problem: patients don't know what to ask; the chart knows things
   they don't, and they know things the chart doesn't. And no one wants an AI
   interrupting their doctor.
2. **60s** — live: Elias's visit replaying at speed. **Candidates light up on the right
   and strike through as the doctor addresses them.** At the end, 3 questions remain.
   Land on the sleep one.
3. **30s** — same engine, different patient: Julius. His cost concern survives as
   *partially addressed* — the doctor said "cheap generics" but never said what to do
   if the counter price is wrong, and that is exactly what failed him last time.
4. **35s** — how: deterministic generators propose, resolution tracking suppresses,
   every delivered question traces to a datum. 20 proposed, 3 delivered.
5. **25s** — the eval: suppression ON vs OFF across 25 encounters, groundedness rate.
6. **10s** — scale: same engine over a population -> "what did patients like me
   wish they'd asked."

## Say out loud, unprompted
- the agent never interrupts the clinician — that is a design decision, not a limitation
- the wearable is synthetic, generated by us
- any cohort statistic is synthetic, illustrative of the mechanism, not evidence
- Whisper has no diarization; speaker mapping is by timestamp
